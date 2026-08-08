import "server-only";

import { Chess } from "chess.js";
import { getD1 } from "../db";
import { HistoryError, getPublicGame } from "./history";
import type {
  LiveGame,
  LiveGamePhase,
  LiveGameResult,
  LiveGameSource,
} from "./live-game-types";

const LIVE_GAME_TTL_MS = 24 * 60 * 60 * 1_000;
const MAX_LIVE_PLIES = 1_000;

type StoredLiveGame = Omit<LiveGame, "moves"> & {
  moves: string;
  publisherTokenHash: string;
  expiresAt: number;
};

export type OpenLiveGameInput = {
  id: string;
  publisherToken: string;
  source: LiveGameSource;
  tournamentId?: string | null;
  tournamentPairKey?: string | null;
  tournamentGameIndex?: number | null;
  runnerId?: string | null;
  whiteName: string;
  blackName: string;
  whiteModelReference?: string | null;
  blackModelReference?: string | null;
  openingName?: string | null;
};

export type PublishLiveGameInput = {
  publisherToken: string;
  revision: number;
  phase: LiveGamePhase;
  status: string;
  moves: string[];
  lastMoveMs?: number | null;
  result?: LiveGameResult | null;
};

export async function openLiveGame(input: OpenLiveGameInput): Promise<LiveGame> {
  assertGameId(input.id);
  assertPublisherToken(input.publisherToken);
  if (input.source !== "arena" && input.source !== "tournament") {
    throw new HistoryError(400, "A live game has an unknown source.");
  }
  const source = input.source;
  const tournamentId = cleanOptional(input.tournamentId, 80);
  const pairKey = cleanOptional(input.tournamentPairKey, 120);
  const gameIndex = input.tournamentGameIndex ?? null;
  if (source === "tournament") {
    if (!tournamentId || !pairKey || typeof gameIndex !== "number" || !Number.isInteger(gameIndex) || gameIndex < 0) {
      throw new HistoryError(400, "A tournament broadcast requires its scheduled game slot.");
    }
    await requireCurrentRunner(tournamentId, input.runnerId);
  } else if (tournamentId || pairKey || gameIndex !== null) {
    throw new HistoryError(400, "A regular arena broadcast cannot claim a tournament slot.");
  }

  const now = Date.now();
  const tokenHash = await sha256(input.publisherToken);
  const db = await getD1();
  const statements = [
    db.prepare("DELETE FROM live_games WHERE expires_at < ?").bind(now),
  ];
  if (tournamentId) {
    statements.push(
      db.prepare(`UPDATE live_games SET phase = 'finished', status = 'Broadcast replaced',
          updated_at = ?, expires_at = ?
        WHERE tournament_id = ? AND phase != 'finished'`)
        .bind(now, now + LIVE_GAME_TTL_MS, tournamentId),
    );
  }
  statements.push(
    db.prepare(`INSERT INTO live_games (
        id, publisher_token_hash, source, tournament_id, tournament_pair_key,
        tournament_game_index, white_name, black_name, white_model_reference,
        black_model_reference, opening_name, phase, status, moves, last_move_ms,
        result, revision, started_at, updated_at, expires_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'starting', 'Preparing game…', '[]',
        NULL, NULL, 0, ?, ?, ?)
      ON CONFLICT(id) DO NOTHING`).bind(
      input.id,
      tokenHash,
      source,
      tournamentId,
      pairKey,
      gameIndex,
      cleanRequired(input.whiteName, "White player", 120),
      cleanRequired(input.blackName, "Black player", 120),
      cleanOptional(input.whiteModelReference, 300),
      cleanOptional(input.blackModelReference, 300),
      cleanOptional(input.openingName, 160),
      now,
      now,
      now + LIVE_GAME_TTL_MS,
    ),
  );
  await db.batch(statements);

  const created = await getLiveGame(input.id);
  if (!created) throw new HistoryError(409, "That game already has a different broadcaster.");
  const stored = await getStoredLiveGame(input.id);
  if (!stored || stored.publisherTokenHash !== tokenHash) {
    throw new HistoryError(409, "That game already has a different broadcaster.");
  }
  return created;
}

export async function publishLiveGame(
  id: string,
  input: PublishLiveGameInput,
): Promise<LiveGame> {
  assertGameId(id);
  assertPublisherToken(input.publisherToken);
  if (!Number.isInteger(input.revision) || input.revision < 1) {
    throw new HistoryError(400, "A broadcast update requires a positive revision.");
  }
  const phase = assertPhase(input.phase);
  const moves = validateMoves(input.moves);
  const result = assertResult(input.result ?? null);
  const lastMoveMs = input.lastMoveMs ?? null;
  if (lastMoveMs !== null && (!Number.isFinite(lastMoveMs) || lastMoveMs < 0)) {
    throw new HistoryError(400, "Last-move time must be non-negative.");
  }

  const now = Date.now();
  const tokenHash = await sha256(input.publisherToken);
  const update = await (await getD1()).prepare(`UPDATE live_games SET
      phase = ?, status = ?, moves = ?, last_move_ms = ?, result = ?, revision = ?,
      updated_at = ?, expires_at = ?
    WHERE id = ? AND publisher_token_hash = ? AND revision < ?`)
    .bind(
      phase,
      cleanRequired(input.status, "Broadcast status", 240),
      JSON.stringify(moves),
      lastMoveMs === null ? null : Math.round(lastMoveMs),
      result,
      input.revision,
      now,
      now + LIVE_GAME_TTL_MS,
      id,
      tokenHash,
      input.revision,
    )
    .run();
  if ((update.meta?.changes ?? 0) === 0) {
    const stored = await getStoredLiveGame(id);
    if (!stored || stored.publisherTokenHash !== tokenHash) {
      throw new HistoryError(403, "This browser cannot update that live game.");
    }
    if (stored.revision >= input.revision) return toPublicLiveGame(stored);
    throw new HistoryError(409, "The live game could not be updated.");
  }
  const published = await getLiveGame(id);
  if (!published) throw new HistoryError(404, "That live game no longer exists.");
  return published;
}

export async function getLiveGame(id: string): Promise<LiveGame | null> {
  const stored = await getStoredLiveGame(id);
  if (!stored || stored.expiresAt <= Date.now()) return null;
  return toPublicLiveGame(stored);
}

export async function getTournamentLiveGame(tournamentId: string): Promise<LiveGame | null> {
  const stored = await (await getD1()).prepare(`${LIVE_GAME_SELECT}
      WHERE tournament_id = ? AND phase != 'finished' AND expires_at > ?
      ORDER BY updated_at DESC LIMIT 1`)
    .bind(tournamentId, Date.now())
    .first<StoredLiveGame>();
  return stored ? toPublicLiveGame(stored) : null;
}

export async function getLiveGameResponse(id: string) {
  const [live, completed] = await Promise.all([getLiveGame(id), getPublicGame(id)]);
  return {
    live,
    completed: completed ? {
      id: completed.id,
      whiteName: completed.whiteName,
      blackName: completed.blackName,
      pgn: completed.pgn,
      result: completed.result,
      termination: completed.termination,
      recordedAt: completed.recordedAt,
    } : null,
  };
}

const LIVE_GAME_SELECT = `SELECT
    id, publisher_token_hash AS publisherTokenHash, source,
    tournament_id AS tournamentId, tournament_pair_key AS tournamentPairKey,
    tournament_game_index AS tournamentGameIndex,
    white_name AS whiteName, black_name AS blackName,
    white_model_reference AS whiteModelReference,
    black_model_reference AS blackModelReference,
    opening_name AS openingName, phase, status, moves,
    last_move_ms AS lastMoveMs, result, revision,
    started_at AS startedAt, updated_at AS updatedAt, expires_at AS expiresAt
  FROM live_games`;

async function getStoredLiveGame(id: string): Promise<StoredLiveGame | null> {
  return await (await getD1()).prepare(`${LIVE_GAME_SELECT} WHERE id = ?`)
    .bind(id)
    .first<StoredLiveGame>() ?? null;
}

function toPublicLiveGame(stored: StoredLiveGame): LiveGame {
  let moves: string[] = [];
  try {
    const parsed = JSON.parse(stored.moves) as unknown;
    if (Array.isArray(parsed) && parsed.every((move) => typeof move === "string")) moves = parsed;
  } catch {
    // A malformed ephemeral row should render as an empty board, not break the
    // permanent game history or tournament standings.
  }
  return {
    id: stored.id,
    source: stored.source,
    tournamentId: stored.tournamentId,
    tournamentPairKey: stored.tournamentPairKey,
    tournamentGameIndex: stored.tournamentGameIndex,
    whiteName: stored.whiteName,
    blackName: stored.blackName,
    whiteModelReference: stored.whiteModelReference,
    blackModelReference: stored.blackModelReference,
    openingName: stored.openingName,
    phase: stored.phase,
    status: stored.status,
    moves,
    lastMoveMs: stored.lastMoveMs,
    result: stored.result,
    revision: stored.revision,
    startedAt: stored.startedAt,
    updatedAt: stored.updatedAt,
  };
}

async function requireCurrentRunner(tournamentId: string, runnerId: unknown): Promise<void> {
  if (typeof runnerId !== "string" || !runnerId) {
    throw new HistoryError(401, "A tournament broadcast requires the pinned runner.");
  }
  const row = await (await getD1()).prepare(
    "SELECT id FROM tournaments WHERE id = ? AND runner_id = ? AND status = 'running'",
  ).bind(tournamentId, runnerId).first<{ id: string }>();
  if (!row) throw new HistoryError(403, "Only the pinned runner can broadcast this tournament.");
}

function validateMoves(value: unknown): string[] {
  if (!Array.isArray(value) || value.length > MAX_LIVE_PLIES) {
    throw new HistoryError(400, "A live game has too many moves.");
  }
  const game = new Chess();
  const moves: string[] = [];
  for (const candidate of value) {
    if (typeof candidate !== "string" || candidate.length > 24) {
      throw new HistoryError(400, "A live game contains an invalid move.");
    }
    let move;
    try {
      move = game.move(candidate);
    } catch {
      move = null;
    }
    if (!move) throw new HistoryError(400, `The live move “${candidate}” is not legal.`);
    moves.push(move.san);
  }
  return moves;
}

function assertGameId(value: unknown): asserts value is string {
  if (typeof value !== "string" || !/^[0-9a-f]{8}-[0-9a-f-]{27}$/i.test(value)) {
    throw new HistoryError(400, "A live game requires a valid id.");
  }
}

function assertPublisherToken(value: unknown): asserts value is string {
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/i.test(value)) {
    throw new HistoryError(400, "A live game requires a valid publisher token.");
  }
}

function assertPhase(value: unknown): LiveGamePhase {
  if (value === "starting" || value === "playing" || value === "paused" || value === "finished") return value;
  throw new HistoryError(400, "A live game has an unknown phase.");
}

function assertResult(value: unknown): LiveGameResult | null {
  if (value === null || value === "1-0" || value === "0-1" || value === "1/2-1/2") return value;
  throw new HistoryError(400, "A live game has an invalid result.");
}

function cleanRequired(value: unknown, label: string, max: number): string {
  const cleaned = typeof value === "string" ? value.trim().slice(0, max) : "";
  if (!cleaned) throw new HistoryError(400, `${label} is required.`);
  return cleaned;
}

function cleanOptional(value: unknown, max: number): string | null {
  if (typeof value !== "string") return null;
  return value.trim().slice(0, max) || null;
}

async function sha256(value: string): Promise<string> {
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
