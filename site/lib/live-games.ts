import "server-only";

import { Chess } from "chess.js";
import { getD1 } from "../db";
import { HistoryError, getPublicGame } from "./history";
import {
  normalizeLiveGameEventBatch,
  type LiveGameEventBatch,
} from "./live-game-events.mjs";
import type {
  LiveGame,
  LiveGamePhase,
  LiveGameResult,
  LiveGameSource,
} from "./live-game-types";

const LIVE_GAME_TTL_MS = 60 * 60 * 1_000;
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
  whiteMoveTimeLimitMs?: number | null;
  blackMoveTimeLimitMs?: number | null;
  openingName?: string | null;
};

export type PublishLiveGameInput = {
  publisherToken: string;
  revision: number;
  phase: LiveGamePhase;
  status: string;
  moves: string[];
  lastMoveMs?: number | null;
  activeTurnColor?: "w" | "b" | null;
  activeTurnElapsedMs?: number | null;
  result?: LiveGameResult | null;
  eventBatch?: unknown;
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
  const whiteModelReference = cleanOptional(input.whiteModelReference, 300);
  const blackModelReference = cleanOptional(input.blackModelReference, 300);
  const whiteMoveTimeLimitMs = whiteModelReference
    ? cleanMoveTimeLimit(input.whiteMoveTimeLimitMs, "White move time limit")
    : null;
  const blackMoveTimeLimitMs = blackModelReference
    ? cleanMoveTimeLimit(input.blackMoveTimeLimitMs, "Black move time limit")
    : null;

  const now = Date.now();
  const tokenHash = await sha256(input.publisherToken);
  const db = await getD1();
  const statements = [
    db.prepare("DELETE FROM live_game_event_batches WHERE expires_at < ?").bind(now),
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
        black_model_reference, white_move_time_limit_ms, black_move_time_limit_ms,
        opening_name, phase, status, moves, last_move_ms, active_turn_color,
        active_turn_elapsed_ms, result, revision, event_seq, started_at, updated_at,
        expires_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'starting', 'Preparing game…', '[]',
        NULL, NULL, NULL, NULL, 0, 0, ?, ?, ?)
      ON CONFLICT(id) DO NOTHING`).bind(
      input.id,
      tokenHash,
      source,
      tournamentId,
      pairKey,
      gameIndex,
      cleanRequired(input.whiteName, "White player", 120),
      cleanRequired(input.blackName, "Black player", 120),
      whiteModelReference,
      blackModelReference,
      whiteMoveTimeLimitMs,
      blackMoveTimeLimitMs,
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
  const status = cleanRequired(input.status, "Broadcast status", 240);
  const lastMoveMs = input.lastMoveMs ?? null;
  if (lastMoveMs !== null && (!Number.isFinite(lastMoveMs) || lastMoveMs < 0)) {
    throw new HistoryError(400, "Last-move time must be non-negative.");
  }
  let activeTurnColor = input.activeTurnColor ?? null;
  let activeTurnElapsedMs = input.activeTurnElapsedMs ?? null;
  if (activeTurnColor !== null && activeTurnColor !== "w" && activeTurnColor !== "b") {
    throw new HistoryError(400, "The active turn has an invalid colour.");
  }
  if (activeTurnElapsedMs !== null && (!Number.isFinite(activeTurnElapsedMs) || activeTurnElapsedMs < 0)) {
    throw new HistoryError(400, "The active-turn time must be non-negative.");
  }
  if ((activeTurnColor === null) !== (activeTurnElapsedMs === null)) {
    throw new HistoryError(400, "The active turn requires both a colour and elapsed time.");
  }
  if (phase === "finished") {
    activeTurnColor = null;
    activeTurnElapsedMs = null;
  }

  const tokenHash = await sha256(input.publisherToken);
  const stored = await getStoredLiveGame(id);
  if (!stored || stored.publisherTokenHash !== tokenHash) {
    throw new HistoryError(403, "This browser cannot update that live game.");
  }
  if (stored.revision >= input.revision) return toPublicLiveGame(stored);

  const eventBatch = input.eventBatch === undefined
    ? null
    : normalizeLiveGameEventBatch(input.eventBatch);
  if (input.eventBatch !== undefined && !eventBatch) {
    throw new HistoryError(400, "The live-game event batch is invalid.");
  }
  if (eventBatch && eventBatch.firstSeq !== stored.eventSeq + 1) {
    if (eventBatch.lastSeq <= stored.eventSeq) return toPublicLiveGame(stored);
    throw new HistoryError(409, "The live-game event stream has a sequence gap.");
  }

  const normalizedUpdate = {
    phase,
    status,
    moves,
    lastMoveMs: lastMoveMs === null ? null : Math.round(lastMoveMs),
    activeTurnColor,
    activeTurnElapsedMs: activeTurnElapsedMs === null ? null : Math.round(activeTurnElapsedMs),
    result,
  };
  if (eventBatch) {
    for (const event of eventBatch.events) {
      if (event.payload.type === "game.updated") event.payload.update = normalizedUpdate;
    }
  }

  const now = Date.now();
  const expiresAt = now + LIVE_GAME_TTL_MS;
  const nextEventSeq = eventBatch?.lastSeq ?? stored.eventSeq;
  const db = await getD1();
  const statements = [];
  if (eventBatch) {
    statements.push(db.prepare(`INSERT INTO live_game_event_batches (
        game_id, batch_index, first_seq, last_seq, events, expires_at
      ) VALUES (?, ?, ?, ?, ?, ?)
      ON CONFLICT(game_id, batch_index) DO NOTHING`).bind(
      id,
      eventBatch.batchIndex,
      eventBatch.firstSeq,
      eventBatch.lastSeq,
      JSON.stringify(eventBatch),
      expiresAt,
    ));
  }
  statements.push(db.prepare(`UPDATE live_games SET
      phase = ?, status = ?, moves = ?, last_move_ms = ?, active_turn_color = ?,
      active_turn_elapsed_ms = ?, result = ?, revision = ?, event_seq = ?,
      updated_at = ?, expires_at = ?
    WHERE id = ? AND publisher_token_hash = ? AND revision < ? AND event_seq = ?`)
    .bind(
      phase,
      status,
      JSON.stringify(moves),
      normalizedUpdate.lastMoveMs,
      normalizedUpdate.activeTurnColor,
      normalizedUpdate.activeTurnElapsedMs,
      result,
      input.revision,
      nextEventSeq,
      now,
      expiresAt,
      id,
      tokenHash,
      input.revision,
      stored.eventSeq,
    ));
  await db.batch(statements);

  const published = await getStoredLiveGame(id);
  if (!published) throw new HistoryError(404, "That live game no longer exists.");
  if (published.revision < input.revision || published.eventSeq < nextEventSeq) {
    throw new HistoryError(409, "The live game could not be updated.");
  }
  return toPublicLiveGame(published);
}

export async function getLiveGame(id: string): Promise<LiveGame | null> {
  const stored = await getStoredLiveGame(id);
  if (!stored || stored.expiresAt <= Date.now()) return null;
  return toPublicLiveGame(stored);
}

export async function getTournamentLiveGame(tournamentId: string): Promise<LiveGame | null> {
  const stored = await (await getD1()).prepare(`${LIVE_GAME_SELECT}
      WHERE tournament_id = ? AND expires_at > ?
      ORDER BY CASE WHEN phase != 'finished' THEN 0 ELSE 1 END,
        updated_at DESC LIMIT 1`)
    .bind(tournamentId, Date.now())
    .first<StoredLiveGame>();
  return stored ? toPublicLiveGame(stored) : null;
}

export async function getLiveGameEventBatches(
  id: string,
  afterSeq: number,
): Promise<LiveGameEventBatch[]> {
  assertGameId(id);
  const { results } = await (await getD1()).prepare(`SELECT events
      FROM live_game_event_batches
      WHERE game_id = ? AND last_seq > ? AND expires_at > ?
      ORDER BY batch_index LIMIT 32`)
    .bind(id, Math.max(0, Math.floor(afterSeq)), Date.now())
    .all<{ events: string }>();
  const batches: LiveGameEventBatch[] = [];
  for (const row of results) {
    try {
      const batch = normalizeLiveGameEventBatch(JSON.parse(row.events));
      if (batch) batches.push(batch);
    } catch {
      // Malformed ephemeral telemetry must not break the permanent game record.
    }
  }
  return batches;
}

export async function getLiveGameResponse(id: string, afterSeq?: number) {
  const [live, completed, batches] = await Promise.all([
    getLiveGame(id),
    getPublicGame(id),
    afterSeq === undefined ? Promise.resolve([]) : getLiveGameEventBatches(id, afterSeq),
  ]);
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
    batches,
  };
}

const LIVE_GAME_SELECT = `SELECT
    id, publisher_token_hash AS publisherTokenHash, source,
    tournament_id AS tournamentId, tournament_pair_key AS tournamentPairKey,
    tournament_game_index AS tournamentGameIndex,
    white_name AS whiteName, black_name AS blackName,
    white_model_reference AS whiteModelReference,
    black_model_reference AS blackModelReference,
    white_move_time_limit_ms AS whiteMoveTimeLimitMs,
    black_move_time_limit_ms AS blackMoveTimeLimitMs,
    opening_name AS openingName, phase, status, moves,
    last_move_ms AS lastMoveMs, active_turn_color AS activeTurnColor,
    active_turn_elapsed_ms AS activeTurnElapsedMs, result, revision,
    event_seq AS eventSeq,
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
    whiteMoveTimeLimitMs: stored.whiteMoveTimeLimitMs,
    blackMoveTimeLimitMs: stored.blackMoveTimeLimitMs,
    openingName: stored.openingName,
    phase: stored.phase,
    status: stored.status,
    moves,
    lastMoveMs: stored.lastMoveMs,
    activeTurnColor: stored.activeTurnColor,
    activeTurnElapsedMs: stored.activeTurnElapsedMs,
    result: stored.result,
    revision: stored.revision,
    eventSeq: stored.eventSeq,
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

function cleanMoveTimeLimit(value: unknown, label: string): number | null {
  if (value === undefined || value === null) return null;
  if (typeof value !== "number" || !Number.isFinite(value) || !Number.isInteger(value) || value <= 0 || value > 600_000) {
    throw new HistoryError(400, `${label} must be a positive integer no greater than 600000 ms.`);
  }
  return value;
}

async function sha256(value: string): Promise<string> {
  const bytes = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, "0")).join("");
}
