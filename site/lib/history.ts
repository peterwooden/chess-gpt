import "server-only";

import { Chess } from "chess.js";
import { getD1, getRuntimeEnvironment } from "../db";
import type { ChatGPTUser } from "../app/chatgpt-auth";
import { modelPageHref, resolveHuggingFaceReference } from "../app/arena/hugging-face-reference.mjs";
import { parsePackageManifest } from "../app/arena/package-manifest.mjs";
import { buildModelDirectoryQuery, buildModelProfileQuery, MODEL_VERSIONS_SQL } from "./model-catalog-query.mjs";

const ARENA_VERSION = "history-v1";
const MAX_PGN_BYTES = 64_000;
const PAGE_SIZE = 24;

export type ParticipantInput =
  | { kind: "human" }
  | { kind: "model"; reference: string };

export type SaveGameInput = {
  id: string;
  pgn: string;
  playedAt?: string;
  white: ParticipantInput;
  black: ParticipantInput;
};

export type PublicPlayer = {
  id: string;
  kind: "human" | "model";
  displayName: string;
  playerCode: string;
  createdAt: number;
  lastPlayedAt: number;
  repository: string | null;
  commitSha: string | null;
  manifestSha256: string | null;
};

export type PublicGame = {
  id: string;
  whitePlayerId: string | null;
  blackPlayerId: string | null;
  whiteName: string;
  blackName: string;
  whiteModelReference: string | null;
  blackModelReference: string | null;
  result: "1-0" | "0-1" | "1/2-1/2";
  termination: string;
  pgn: string;
  moveCount: number;
  arenaVersion: string;
  playedAt: number;
  recordedAt: number;
};

type StoredParticipant = {
  id: string | null;
  name: string;
};

export type ParticipantProfile = {
  id: string;
  name: string;
};

export type ModelParticipantProfile = ParticipantProfile & {
  reference: string;
  repository: string;
  commitSha: string;
  profileHref: string;
};

type DirectoryRow = PublicPlayer & {
  games: number;
  wins: number;
  draws: number;
  losses: number;
};

export type PublicModel = {
  repository: string;
  displayName: string;
  firstSeenAt: number;
  lastPlayedAt: number;
  latestCommitSha: string;
  latestFirstSeenAt: number;
  versions: number;
  games: number;
  wins: number;
  draws: number;
  losses: number;
};

export type PublicModelVersion = {
  playerId: string;
  displayName: string;
  repository: string;
  commitSha: string;
  manifestSha256: string;
  firstSeenAt: number;
  lastPlayedAt: number;
  games: number;
  wins: number;
  draws: number;
  losses: number;
};

export async function saveCompletedGame(
  input: SaveGameInput,
  user: ChatGPTUser | null,
): Promise<PublicGame> {
  if (!/^[0-9a-f]{8}-[0-9a-f-]{27}$/i.test(input.id)) throw new HistoryError(400, "Invalid game id.");
  if (typeof input.pgn !== "string" || new TextEncoder().encode(input.pgn).byteLength > MAX_PGN_BYTES) {
    throw new HistoryError(400, "The PGN is missing or too large.");
  }
  const seats = [input.white, input.black];
  if (!seats.every(isParticipantInput)) throw new HistoryError(400, "Both game participants are required.");
  const modelCount = seats.filter((seat) => seat.kind === "model").length;
  if (modelCount < 1 || modelCount > 2) throw new HistoryError(400, "Arena games require one or two models.");
  if (seats.filter((seat) => seat.kind === "human").length > 1) {
    throw new HistoryError(400, "The arena supports at most one human player.");
  }

  const validated = validateCompletedPgn(input.pgn);
  const db = await getD1();
  const existing = await getPublicGame(input.id);
  if (existing) return existing;

  const [white, black] = await Promise.all([
    resolveParticipant(input.white, user),
    resolveParticipant(input.black, user),
  ]);
  const recordedAt = Date.now();
  const requestedPlayedAt = input.playedAt ? Date.parse(input.playedAt) : Number.NaN;
  const playedAt = Number.isFinite(requestedPlayedAt) && requestedPlayedAt <= recordedAt
    ? requestedPlayedAt
    : recordedAt;
  const canonicalPgn = canonicalizePgn(
    validated.game,
    white.name,
    black.name,
    validated.result,
    validated.termination,
  );

  await db.batch([
    db.prepare(`INSERT INTO games (
      id, white_player_id, black_player_id, white_name, black_name, result,
      termination, pgn, move_count, arena_version, played_at, recorded_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(id) DO NOTHING`).bind(
      input.id,
      white.id,
      black.id,
      white.name,
      black.name,
      validated.result,
      validated.termination,
      canonicalPgn,
      Math.ceil(validated.game.history().length / 2),
      ARENA_VERSION,
      playedAt,
      recordedAt,
    ),
    ...(white.id ? [db.prepare("UPDATE players SET last_played_at = ? WHERE id = ?").bind(recordedAt, white.id)] : []),
    ...(black.id ? [db.prepare("UPDATE players SET last_played_at = ? WHERE id = ?").bind(recordedAt, black.id)] : []),
  ]);
  const saved = await getPublicGame(input.id);
  if (!saved) throw new HistoryError(500, "The completed game could not be saved.");
  return saved;
}

export async function getPublicGame(id: string): Promise<PublicGame | null> {
  const row = await (await getD1()).prepare(`SELECT
      g.id, g.white_player_id AS whitePlayerId, g.black_player_id AS blackPlayerId,
      g.white_name AS whiteName, g.black_name AS blackName, g.result, g.termination,
      g.pgn, g.move_count AS moveCount, g.arena_version AS arenaVersion,
      g.played_at AS playedAt, g.recorded_at AS recordedAt,
      CASE WHEN wmv.player_id IS NULL THEN NULL ELSE wmv.repository || '@' || wmv.commit_sha END AS whiteModelReference,
      CASE WHEN bmv.player_id IS NULL THEN NULL ELSE bmv.repository || '@' || bmv.commit_sha END AS blackModelReference
    FROM games g
    LEFT JOIN model_versions wmv ON wmv.player_id = g.white_player_id
    LEFT JOIN model_versions bmv ON bmv.player_id = g.black_player_id
    WHERE g.id = ?`).bind(id).first<PublicGame>();
  return row ?? null;
}

export async function listPlayers(options: {
  kind: "human" | "model";
  search?: string;
  sort?: "recent" | "name" | "games";
  cursor?: string;
}) {
  const sort = options.sort ?? "recent";
  const search = options.search?.trim().slice(0, 100) ?? "";
  const cursor = decodeCursor(options.cursor);
  const where = ["p.kind = ?"];
  const bindings: unknown[] = [options.kind];
  if (search) {
    where.push(`(p.display_name LIKE ? COLLATE NOCASE OR p.player_code LIKE ? COLLATE NOCASE
      OR mv.repository LIKE ? COLLATE NOCASE OR mv.commit_sha LIKE ? COLLATE NOCASE)`);
    const pattern = `%${search}%`;
    bindings.push(pattern, pattern, pattern, pattern);
  }
  let having = "";
  let order = "p.last_played_at DESC, p.id DESC";
  if (cursor && sort === "recent") {
    where.push("(p.last_played_at < ? OR (p.last_played_at = ? AND p.id < ?))");
    bindings.push(Number(cursor.value), Number(cursor.value), cursor.id);
  } else if (cursor && sort === "name") {
    where.push("(lower(p.display_name) > ? OR (lower(p.display_name) = ? AND p.id > ?))");
    bindings.push(cursor.value, cursor.value, cursor.id);
  }
  if (sort === "name") order = "lower(p.display_name) ASC, p.id ASC";
  if (sort === "games") {
    order = "COUNT(g.id) DESC, p.id DESC";
    if (cursor) {
      having = "HAVING (COUNT(g.id) < ? OR (COUNT(g.id) = ? AND p.id < ?))";
      bindings.push(Number(cursor.value), Number(cursor.value), cursor.id);
    }
  }
  bindings.push(PAGE_SIZE + 1);
  const result = await (await getD1()).prepare(`${directorySelectSql()}
    WHERE ${where.join(" AND ")}
    GROUP BY p.id
    ${having}
    ORDER BY ${order}
    LIMIT ?`).bind(...bindings).all<DirectoryRow>();
  const rows = result.results.slice(0, PAGE_SIZE);
  const last = rows.at(-1);
  const value = last ? sort === "recent" ? last.lastPlayedAt : sort === "name" ? last.displayName.toLocaleLowerCase() : last.games : null;
  return {
    players: rows,
    nextCursor: result.results.length > PAGE_SIZE && last ? encodeCursor(value, last.id) : null,
  };
}

export async function listModels(options: {
  search?: string;
  sort?: "recent" | "name" | "games" | "versions";
  cursor?: string;
}) {
  const sort = options.sort ?? "recent";
  const search = options.search?.trim().slice(0, 100) ?? "";
  const cursor = decodeCursor(options.cursor);
  const query = buildModelDirectoryQuery({ search, sort, cursor, limit: PAGE_SIZE + 1 });
  const result = await (await getD1()).prepare(query.sql).bind(...query.bindings).all<PublicModel>();
  const rows = result.results.slice(0, PAGE_SIZE);
  const last = rows.at(-1);
  const value = last
    ? sort === "recent" ? last.lastPlayedAt
      : sort === "name" ? last.displayName.toLocaleLowerCase()
        : last[sort]
    : null;
  return {
    models: rows,
    nextCursor: result.results.length > PAGE_SIZE && last ? encodeCursor(value, last.repository) : null,
  };
}

export async function getModelProfile(repository: string): Promise<PublicModel | null> {
  const model = await (await getD1()).prepare(buildModelProfileQuery()).bind(repository).first<PublicModel>();
  return model ?? null;
}

export async function listModelVersions(repository: string): Promise<PublicModelVersion[]> {
  const result = await (await getD1()).prepare(MODEL_VERSIONS_SQL).bind(repository).all<PublicModelVersion>();
  return result.results;
}

export async function getPlayerProfile(id: string) {
  const player = await (await getD1()).prepare(`${directorySelectSql()}
    WHERE p.id = ?
    GROUP BY p.id`).bind(id).first<DirectoryRow>();
  return player ?? null;
}

export async function listPlayerGames(playerId: string, cursorValue?: string) {
  const cursor = decodeCursor(cursorValue);
  const where = ["(g.white_player_id = ? OR g.black_player_id = ?)"];
  const bindings: unknown[] = [playerId, playerId];
  if (cursor) {
    where.push("(g.recorded_at < ? OR (g.recorded_at = ? AND g.id < ?))");
    bindings.push(Number(cursor.value), Number(cursor.value), cursor.id);
  }
  bindings.push(PAGE_SIZE + 1);
  const result = await (await getD1()).prepare(`SELECT
      g.id, g.white_player_id AS whitePlayerId, g.black_player_id AS blackPlayerId,
      g.white_name AS whiteName, g.black_name AS blackName, g.result,
      g.move_count AS moveCount, g.played_at AS playedAt, g.recorded_at AS recordedAt
    FROM games g
    WHERE ${where.join(" AND ")}
    ORDER BY g.recorded_at DESC, g.id DESC
    LIMIT ?`).bind(...bindings).all<Omit<PublicGame, "termination" | "pgn" | "arenaVersion" | "whiteModelReference" | "blackModelReference">>();
  const games = result.results.slice(0, PAGE_SIZE);
  const last = games.at(-1);
  return {
    games,
    nextCursor: result.results.length > PAGE_SIZE && last ? encodeCursor(last.recordedAt, last.id) : null,
  };
}

async function resolveParticipant(input: ParticipantInput, user: ChatGPTUser | null): Promise<StoredParticipant> {
  if (input.kind === "human") return user ? getOrCreateHuman(user) : { id: null, name: "Anonymous" };
  return getOrCreateModel(input.reference);
}

export async function ensureHumanPlayer(user: ChatGPTUser): Promise<ParticipantProfile> {
  return getOrCreateHuman(user);
}

export async function ensureModelPlayer(reference: string): Promise<ModelParticipantProfile> {
  return getOrCreateModel(reference);
}

async function getOrCreateHuman(user: ChatGPTUser): Promise<ParticipantProfile> {
  const secret = (await getRuntimeEnvironment()).PLAYER_ID_HMAC_SECRET
    ?? process.env.PLAYER_ID_HMAC_SECRET;
  if (!secret) throw new HistoryError(500, "Player identity is not configured.");
  const digest = await hmacHex(secret, user.email.trim().toLocaleLowerCase());
  const identityKey = `human:${digest}`;
  const existing = await findPlayer(identityKey);
  if (existing) return existing;
  const id = crypto.randomUUID();
  const playerCode = publicPlayerCode(id);
  const name = sanitizeName(user.fullName) || `Player ${playerCode}`;
  const now = Date.now();
  await (await getD1()).prepare(`INSERT INTO players
    (id, kind, identity_key, display_name, player_code, created_at, last_played_at)
    VALUES (?, 'human', ?, ?, ?, ?, ?)
    ON CONFLICT(identity_key) DO NOTHING`).bind(id, identityKey, name, playerCode, now, now).run();
  return (await findPlayer(identityKey)) ?? { id, name };
}

async function getOrCreateModel(reference: string): Promise<ModelParticipantProfile> {
  if (typeof reference !== "string" || reference.length > 300) throw new HistoryError(400, "Invalid model reference.");
  const resolved = await resolveHuggingFaceReference(reference);
  const identityKey = `model:${resolved.reference}`;
  const existing = await findPlayer(identityKey);
  if (existing) return modelParticipantProfile(existing, resolved.reference, resolved.repository, resolved.revision);
  const manifestResponse = await fetch(resolved.manifestUrl, { headers: { Accept: "application/json" } });
  if (!manifestResponse.ok) throw new HistoryError(400, "The pinned model manifest could not be loaded.");
  const bytes = new Uint8Array(await manifestResponse.arrayBuffer());
  if (bytes.byteLength > 1_000_000) throw new HistoryError(400, "The model manifest is too large.");
  let manifestName: string;
  try {
    manifestName = parsePackageManifest(bytes).manifest.name;
  } catch (error) {
    throw new HistoryError(400, error instanceof Error ? error.message : "The pinned model manifest is invalid.");
  }
  const name = sanitizeName(manifestName);
  if (!name) throw new HistoryError(400, "The pinned model manifest has no valid name.");
  const manifestSha256 = await sha256Hex(bytes);
  const id = crypto.randomUUID();
  const playerCode = publicPlayerCode(id);
  const now = Date.now();
  const db = await getD1();
  await db.batch([
    db.prepare(`INSERT INTO models (repository, display_name, first_seen_at)
      VALUES (?, ?, ?)
      ON CONFLICT(repository) DO UPDATE SET display_name = excluded.display_name`).bind(
        resolved.repository,
        name,
        now,
      ),
    db.prepare(`INSERT INTO players
      (id, kind, identity_key, display_name, player_code, created_at, last_played_at)
      VALUES (?, 'model', ?, ?, ?, ?, ?)
      ON CONFLICT(identity_key) DO NOTHING`).bind(id, identityKey, name, playerCode, now, now),
    db.prepare(`INSERT INTO model_versions (player_id, repository, commit_sha, manifest_sha256, first_seen_at)
      SELECT id, ?, ?, ?, ? FROM players WHERE identity_key = ?
      ON CONFLICT(player_id) DO NOTHING`).bind(
        resolved.repository,
        resolved.revision,
        manifestSha256,
        now,
        identityKey,
      ),
  ]);
  const player = (await findPlayer(identityKey)) ?? { id, name };
  return modelParticipantProfile(player, resolved.reference, resolved.repository, resolved.revision);
}

function modelParticipantProfile(
  player: StoredParticipant,
  reference: string,
  repository: string,
  commitSha: string,
): ModelParticipantProfile {
  if (!player.id) throw new HistoryError(500, "The model profile has no persistent identity.");
  return { ...player, id: player.id, reference, repository, commitSha, profileHref: modelPageHref(reference) };
}

async function findPlayer(identityKey: string): Promise<StoredParticipant | null> {
  const row = await (await getD1()).prepare("SELECT id, display_name AS name FROM players WHERE identity_key = ?")
    .bind(identityKey).first<StoredParticipant>();
  return row ?? null;
}

function validateCompletedPgn(pgn: string): {
  game: Chess;
  result: PublicGame["result"];
  termination: string;
} {
  const game = new Chess();
  try {
    game.loadPgn(pgn);
  } catch {
    throw new HistoryError(400, "The PGN is invalid.");
  }
  if (game.history().length === 0) throw new HistoryError(400, "A game must contain at least one move.");
  const headers = game.getHeaders();
  let result: PublicGame["result"];
  if (game.isCheckmate()) result = game.turn() === "w" ? "0-1" : "1-0";
  else if (game.isDraw()) result = "1/2-1/2";
  else if ((headers.Result === "1-0" || headers.Result === "0-1") && /^forfeit:/i.test(headers.Termination ?? "")) {
    result = headers.Result;
  } else {
    throw new HistoryError(400, "Only completed games can be saved.");
  }
  if (headers.Result && headers.Result !== "*" && headers.Result !== result) {
    throw new HistoryError(400, "The PGN result does not match the final position.");
  }
  return { game, result, termination: terminationForGame(game, headers.Termination) };
}

function canonicalizePgn(
  game: Chess,
  whiteName: string,
  blackName: string,
  result: PublicGame["result"],
  termination: string,
): string {
  const canonical = new Chess();
  for (const san of game.history()) canonical.move(san);
  canonical.setHeader("Event", "ChessGPT Arena");
  canonical.setHeader("White", whiteName);
  canonical.setHeader("Black", blackName);
  canonical.setHeader("Result", result);
  canonical.setHeader("Termination", termination);
  return canonical.pgn({ maxWidth: 0 });
}

function terminationForGame(game: Chess, reported?: string): string {
  if (!game.isGameOver() && reported && /^forfeit:/i.test(reported)) return sanitizeName(reported) || "forfeit";
  if (game.isCheckmate()) return "checkmate";
  if (game.isStalemate()) return "stalemate";
  if (game.isThreefoldRepetition()) return "threefold repetition";
  if (game.isInsufficientMaterial()) return "insufficient material";
  if (game.isDrawByFiftyMoves()) return "fifty-move rule";
  return "draw";
}

function directorySelectSql() {
  return `SELECT
      p.id, p.kind, p.display_name AS displayName, p.player_code AS playerCode,
      p.created_at AS createdAt, p.last_played_at AS lastPlayedAt,
      mv.repository, mv.commit_sha AS commitSha, mv.manifest_sha256 AS manifestSha256,
      COUNT(g.id) AS games,
      COALESCE(SUM(CASE WHEN (g.seat = 'w' AND g.result = '1-0')
        OR (g.seat = 'b' AND g.result = '0-1') THEN 1 ELSE 0 END), 0) AS wins,
      COALESCE(SUM(CASE WHEN g.result = '1/2-1/2' THEN 1 ELSE 0 END), 0) AS draws,
      COALESCE(SUM(CASE WHEN (g.seat = 'w' AND g.result = '0-1')
        OR (g.seat = 'b' AND g.result = '1-0') THEN 1 ELSE 0 END), 0) AS losses
    FROM players p
    LEFT JOIN model_versions mv ON mv.player_id = p.id
    LEFT JOIN (
      SELECT id, result, white_player_id AS player_id, 'w' AS seat FROM games WHERE white_player_id IS NOT NULL
      UNION ALL
      SELECT id, result, black_player_id AS player_id, 'b' AS seat FROM games WHERE black_player_id IS NOT NULL
    ) g ON g.player_id = p.id`;
}

function isParticipantInput(value: unknown): value is ParticipantInput {
  if (!isRecord(value)) return false;
  return value.kind === "human" || (value.kind === "model" && typeof value.reference === "string");
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value);
}

function sanitizeName(value: unknown): string {
  return typeof value === "string"
    ? value.replace(/[\\\r\n]+/g, " ").replaceAll('"', "'").trim().slice(0, 160)
    : "";
}

function publicPlayerCode(id: string): string {
  return id.replaceAll("-", "").slice(0, 8).toLocaleUpperCase();
}

async function hmacHex(secret: string, value: string): Promise<string> {
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(secret),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const bytes = new Uint8Array(await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(value)));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest("SHA-256", bytes));
  return Array.from(digest, (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function encodeCursor(value: unknown, id: string): string {
  return encodeURIComponent(JSON.stringify({ value, id }));
}

function decodeCursor(cursor?: string): { value: string; id: string } | null {
  if (!cursor) return null;
  try {
    const value = JSON.parse(decodeURIComponent(cursor));
    if ((typeof value.value === "string" || typeof value.value === "number") && typeof value.id === "string") {
      return { value: String(value.value), id: value.id };
    }
  } catch {
    return null;
  }
  return null;
}

export class HistoryError extends Error {
  constructor(public readonly status: number, message: string) {
    super(message);
  }
}
