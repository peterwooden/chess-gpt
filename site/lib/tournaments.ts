import "server-only";

import { getD1, getRuntimeEnvironment } from "../db";
import type { ChatGPTUser } from "../app/chatgpt-auth";
import { resolveHuggingFaceReference } from "../app/arena/hugging-face-reference.mjs";
import { ensureHumanPlayer, ensureModelPlayer, HistoryError } from "./history";
import { buildSchedule, remainingGames, standings } from "./tournament-schedule.mjs";

export type TournamentStatus = "registration" | "running" | "completed";

export type TournamentConfig = {
  name: string;
  gamesPerPair: number;
  moveTimeLimitMs: number;
  maxPlies: number;
  residentBudgetBytes: number;
  maxAttemptsPerGame: number;
};

export type RunnerChange = {
  at: number;
  fromRunnerId: string | null;
  fromLabel: string | null;
  toRunnerId: string;
  toLabel: string;
  authorisedBy: string;
};

export type Tournament = {
  id: string;
  name: string;
  status: TournamentStatus;
  gamesPerPair: number;
  moveTimeLimitMs: number;
  maxPlies: number;
  residentBudgetBytes: number;
  maxAttemptsPerGame: number;
  createdByPlayerId: string;
  createdAt: number;
  startedAt: number | null;
  completedAt: number | null;
  runnerId: string | null;
  runnerLabel: string | null;
  runnerMetadata: string | null;
  runnerHeartbeatAt: number | null;
  runnerChanges: string;
};

export type TournamentEntry = {
  id: string;
  tournamentId: string;
  ownerPlayerId: string;
  modelPlayerId: string;
  reference: string;
  manifestSha256: string;
  displayName: string;
  packageBytes: number;
  verifiedAt: number | null;
  smokeMoveCount: number | null;
  smokeMedianMs: number | null;
  smokeP95Ms: number | null;
  createdAt: number;
  updatedAt: number;
};

export type RecordedTournamentGame = {
  id: string;
  pairKey: string;
  gameIndex: number;
  whitePlayerId: string;
  blackPlayerId: string;
  whiteName: string;
  blackName: string;
  result: "1-0" | "0-1" | "1/2-1/2";
  termination: string;
  moveCount: number;
  moveList: string;
  recordedAt: number;
};

/**
 * How long a runner heartbeat stays valid. A second tab would not merely
 * duplicate work: it would contend for CPU and silently corrupt every move's
 * time budget, so the lease is deliberately short.
 */
export const RUNNER_LEASE_MS = 60_000;

const MAX_GAMES_PER_PAIR = 5_000;
const MAX_PLIES_CEILING = 1_000;

export async function isAdministrator(user: ChatGPTUser | null): Promise<boolean> {
  if (!user) return false;
  const configured = (await getRuntimeEnvironment()).TOURNAMENT_ADMIN_EMAILS
    ?? process.env.TOURNAMENT_ADMIN_EMAILS
    ?? "";
  const allowed = configured
    .split(",")
    .map((value) => value.trim().toLocaleLowerCase())
    .filter(Boolean);
  return allowed.includes(user.email.trim().toLocaleLowerCase());
}

async function requireAdministrator(user: ChatGPTUser | null): Promise<ChatGPTUser> {
  if (!user) throw new HistoryError(401, "Sign in to manage tournaments.");
  if (!await isAdministrator(user)) {
    throw new HistoryError(403, "Only a tournament administrator can do that.");
  }
  return user;
}

export async function listTournaments(): Promise<Tournament[]> {
  const { results } = await (await getD1())
    .prepare(`${TOURNAMENT_SELECT} ORDER BY t.created_at DESC, t.id DESC LIMIT 100`)
    .all<Tournament>();
  return results ?? [];
}

export async function getTournament(id: string): Promise<Tournament | null> {
  const row = await (await getD1())
    .prepare(`${TOURNAMENT_SELECT} WHERE t.id = ?`)
    .bind(id)
    .first<Tournament>();
  return row ?? null;
}

export async function createTournament(
  config: TournamentConfig,
  user: ChatGPTUser | null,
): Promise<Tournament> {
  const administrator = await requireAdministrator(user);
  const validated = validateConfig(config);
  const owner = await ensureHumanPlayer(administrator);
  const id = crypto.randomUUID();
  const now = Date.now();

  await (await getD1()).prepare(`INSERT INTO tournaments (
      id, name, status, games_per_pair, move_time_limit_ms, max_plies,
      resident_budget_bytes, max_attempts_per_game, created_by_player_id,
      created_at, runner_changes
    ) VALUES (?, ?, 'registration', ?, ?, ?, ?, ?, ?, ?, '[]')`).bind(
    id,
    validated.name,
    validated.gamesPerPair,
    validated.moveTimeLimitMs,
    validated.maxPlies,
    validated.residentBudgetBytes,
    validated.maxAttemptsPerGame,
    owner.id,
    now,
  ).run();

  const created = await getTournament(id);
  if (!created) throw new HistoryError(500, "The tournament could not be created.");
  return created;
}

export async function setTournamentStatus(
  id: string,
  status: TournamentStatus,
  user: ChatGPTUser | null,
): Promise<Tournament> {
  await requireAdministrator(user);
  const tournament = await requireTournament(id);

  const allowed: Record<TournamentStatus, TournamentStatus[]> = {
    registration: ["running"],
    running: ["completed", "registration"],
    completed: ["running"],
  };
  if (!allowed[tournament.status].includes(status)) {
    throw new HistoryError(400, `A ${tournament.status} tournament cannot become ${status}.`);
  }

  if (status === "running" && tournament.status === "registration") {
    const entries = await listEntries(id);
    if (entries.length < 2) {
      throw new HistoryError(400, "A tournament needs at least two entries to run.");
    }
    const unverified = entries.filter((entry) => entry.verifiedAt === null);
    if (unverified.length > 0) {
      throw new HistoryError(
        400,
        `${unverified.length} entr${unverified.length === 1 ? "y has" : "ies have"} not passed the registration smoke test.`,
      );
    }
  }

  const now = Date.now();
  await (await getD1()).prepare(`UPDATE tournaments
      SET status = ?,
          started_at = COALESCE(started_at, CASE WHEN ? = 'running' THEN ? END),
          completed_at = CASE WHEN ? = 'completed' THEN ? ELSE NULL END
      WHERE id = ?`).bind(status, status, now, status, now, id).run();

  return await requireTournament(id);
}

export async function listEntries(tournamentId: string): Promise<TournamentEntry[]> {
  const { results } = await (await getD1()).prepare(`SELECT
      id, tournament_id AS tournamentId, owner_player_id AS ownerPlayerId,
      model_player_id AS modelPlayerId, reference, manifest_sha256 AS manifestSha256,
      display_name AS displayName, package_bytes AS packageBytes,
      verified_at AS verifiedAt, smoke_move_count AS smokeMoveCount,
      smoke_median_ms AS smokeMedianMs, smoke_p95_ms AS smokeP95Ms,
      created_at AS createdAt, updated_at AS updatedAt
    FROM tournament_entries WHERE tournament_id = ? ORDER BY id`)
    .bind(tournamentId)
    .all<TournamentEntry>();
  return results ?? [];
}

export type RegisterEntryInput = {
  reference: string;
  packageBytes: number;
  smokeMoveCount: number;
  smokeMedianMs: number;
  smokeP95Ms: number;
};

/**
 * Register a verified entry.
 *
 * The reference is resolved to a full commit SHA here and stored, so the runner
 * loads exactly what was registered. Re-resolving a branch at play time would
 * let someone publish new weights after registration closed.
 */
export async function registerEntry(
  tournamentId: string,
  input: RegisterEntryInput,
  user: ChatGPTUser | null,
): Promise<TournamentEntry> {
  if (!user) throw new HistoryError(401, "Sign in to register a model.");
  const tournament = await requireTournament(tournamentId);
  if (tournament.status !== "registration") {
    throw new HistoryError(409, "This tournament is no longer accepting entries.");
  }
  if (!Number.isInteger(input.packageBytes) || input.packageBytes <= 0) {
    throw new HistoryError(400, "A verified entry must report its package size.");
  }
  if (!Number.isInteger(input.smokeMoveCount) || input.smokeMoveCount < 1) {
    throw new HistoryError(400, "A verified entry must report its smoke-test move count.");
  }

  const resolved = await resolveHuggingFaceReference(input.reference);
  const [owner, model] = await Promise.all([
    ensureHumanPlayer(user),
    ensureModelPlayer(resolved.reference),
  ]);

  const existing = await (await getD1())
    .prepare("SELECT id, owner_player_id AS ownerPlayerId FROM tournament_entries WHERE tournament_id = ? AND model_player_id = ?")
    .bind(tournamentId, model.id)
    .first<{ id: string; ownerPlayerId: string }>();
  if (existing && existing.ownerPlayerId !== owner.id) {
    throw new HistoryError(409, "Another competitor already registered that model version.");
  }

  // The runner re-verifies this digest before play, so an entry that no longer
  // matches what was registered is refused rather than quietly substituted.
  const version = await (await getD1())
    .prepare("SELECT manifest_sha256 AS manifestSha256 FROM model_versions WHERE player_id = ?")
    .bind(model.id)
    .first<{ manifestSha256: string }>();
  if (!version) throw new HistoryError(500, "The registered model version has no recorded manifest digest.");

  const now = Date.now();
  const id = existing?.id ?? crypto.randomUUID();
  await (await getD1()).prepare(`INSERT INTO tournament_entries (
      id, tournament_id, owner_player_id, model_player_id, reference,
      manifest_sha256, display_name, package_bytes, verified_at,
      smoke_move_count, smoke_median_ms, smoke_p95_ms, created_at, updated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(tournament_id, model_player_id) DO UPDATE SET
      reference = excluded.reference,
      manifest_sha256 = excluded.manifest_sha256,
      display_name = excluded.display_name,
      package_bytes = excluded.package_bytes,
      verified_at = excluded.verified_at,
      smoke_move_count = excluded.smoke_move_count,
      smoke_median_ms = excluded.smoke_median_ms,
      smoke_p95_ms = excluded.smoke_p95_ms,
      updated_at = excluded.updated_at`).bind(
    id,
    tournamentId,
    owner.id,
    model.id,
    resolved.reference,
    version.manifestSha256,
    model.name,
    input.packageBytes,
    now,
    input.smokeMoveCount,
    Math.round(input.smokeMedianMs),
    Math.round(input.smokeP95Ms),
    now,
    now,
  ).run();

  const entries = await listEntries(tournamentId);
  const saved = entries.find((entry) => entry.modelPlayerId === model.id);
  if (!saved) throw new HistoryError(500, "The entry could not be registered.");
  return saved;
}

export async function withdrawEntry(
  tournamentId: string,
  entryId: string,
  user: ChatGPTUser | null,
): Promise<void> {
  if (!user) throw new HistoryError(401, "Sign in to withdraw a model.");
  const tournament = await requireTournament(tournamentId);
  if (tournament.status !== "registration") {
    throw new HistoryError(409, "Entries are frozen once a tournament leaves registration.");
  }
  const owner = await ensureHumanPlayer(user);
  const administrator = await isAdministrator(user);
  const entry = await (await getD1())
    .prepare("SELECT owner_player_id AS ownerPlayerId FROM tournament_entries WHERE id = ? AND tournament_id = ?")
    .bind(entryId, tournamentId)
    .first<{ ownerPlayerId: string }>();
  if (!entry) throw new HistoryError(404, "That entry does not exist.");
  if (entry.ownerPlayerId !== owner.id && !administrator) {
    throw new HistoryError(403, "You can only withdraw your own entry.");
  }
  await (await getD1())
    .prepare("DELETE FROM tournament_entries WHERE id = ? AND tournament_id = ?")
    .bind(entryId, tournamentId)
    .run();
}

export type ClaimRunnerInput = {
  runnerId: string;
  runnerLabel: string;
  runnerMetadata: Record<string, unknown>;
  /** Set by an administrator to move a pinned tournament to a different machine. */
  override?: boolean;
};

/**
 * Take or renew the runner lease.
 *
 * A tournament split across machines is not a clean result under a wall-clock
 * limit, so a different machine is refused unless an administrator overrides,
 * and every override is recorded permanently and shown beside the standings.
 */
export async function claimRunner(
  tournamentId: string,
  input: ClaimRunnerInput,
  user: ChatGPTUser | null,
): Promise<Tournament> {
  if (!user) throw new HistoryError(401, "Sign in to run a tournament.");
  const tournament = await requireTournament(tournamentId);
  if (tournament.status !== "running") {
    throw new HistoryError(409, "Only a running tournament can be played.");
  }
  if (typeof input.runnerId !== "string" || !/^[0-9a-f-]{8,64}$/i.test(input.runnerId)) {
    throw new HistoryError(400, "A runner needs a valid identifier.");
  }

  const now = Date.now();
  const label = (input.runnerLabel || "Unnamed machine").slice(0, 120);
  const leaseHeld = tournament.runnerHeartbeatAt !== null
    && now - tournament.runnerHeartbeatAt < RUNNER_LEASE_MS;

  if (tournament.runnerId && tournament.runnerId !== input.runnerId) {
    if (leaseHeld) {
      throw new HistoryError(
        409,
        `“${tournament.runnerLabel ?? "Another machine"}” is already running this tournament.`,
      );
    }
    if (!input.override) {
      throw new HistoryError(
        409,
        `This tournament is pinned to “${tournament.runnerLabel ?? "another machine"}”. An administrator must authorise a different machine, and the change is recorded against the results.`,
      );
    }
    await requireAdministrator(user);
    const changes = parseRunnerChanges(tournament.runnerChanges);
    changes.push({
      at: now,
      fromRunnerId: tournament.runnerId,
      fromLabel: tournament.runnerLabel,
      toRunnerId: input.runnerId,
      toLabel: label,
      authorisedBy: user.email,
    });
    await (await getD1()).prepare("UPDATE tournaments SET runner_changes = ? WHERE id = ?")
      .bind(JSON.stringify(changes), tournamentId).run();
  }

  await (await getD1()).prepare(`UPDATE tournaments
      SET runner_id = ?, runner_label = ?, runner_metadata = ?, runner_heartbeat_at = ?
      WHERE id = ?`).bind(
    input.runnerId,
    label,
    JSON.stringify(input.runnerMetadata ?? {}),
    now,
    tournamentId,
  ).run();

  return await requireTournament(tournamentId);
}

export async function heartbeat(
  tournamentId: string,
  runnerId: string,
): Promise<{ ok: boolean }> {
  const result = await (await getD1())
    .prepare("UPDATE tournaments SET runner_heartbeat_at = ? WHERE id = ? AND runner_id = ?")
    .bind(Date.now(), tournamentId, runnerId)
    .run();
  return { ok: (result.meta?.changes ?? 0) > 0 };
}

export async function listTournamentGames(
  tournamentId: string,
): Promise<RecordedTournamentGame[]> {
  const { results } = await (await getD1()).prepare(`SELECT
      id, tournament_pair_key AS pairKey, tournament_game_index AS gameIndex,
      white_player_id AS whitePlayerId, black_player_id AS blackPlayerId,
      white_name AS whiteName, black_name AS blackName,
      result, termination, move_count AS moveCount, pgn AS moveList,
      recorded_at AS recordedAt
    FROM games WHERE tournament_id = ? ORDER BY recorded_at, id`)
    .bind(tournamentId)
    .all<RecordedTournamentGame>();
  return results ?? [];
}

/**
 * Everything the runner needs to start or resume: the entries, the remaining
 * schedule, and the attempt counters that stop a package which reliably crashes
 * the runner from stalling the tournament forever.
 */
export async function getRunnerPlan(tournamentId: string) {
  const tournament = await requireTournament(tournamentId);
  const [entries, played] = await Promise.all([
    listEntries(tournamentId),
    listTournamentGames(tournamentId),
  ]);
  const schedule = buildSchedule(entries, tournament.gamesPerPair, {
    order: totalPackageBytes(entries) > tournament.residentBudgetBytes
      ? "pair-major"
      : "interleaved",
  });
  const { results } = await (await getD1())
    .prepare("SELECT pair_key AS pairKey, game_index AS gameIndex, attempts FROM tournament_game_attempts WHERE tournament_id = ?")
    .bind(tournamentId)
    .all<{ pairKey: string; gameIndex: number; attempts: number }>();
  const attempts = results ?? [];
  const exhausted = new Set(
    attempts
      .filter((row) => row.attempts >= tournament.maxAttemptsPerGame)
      .map((row) => `${row.pairKey}#${row.gameIndex}`),
  );

  const outstanding = remainingGames(schedule, played)
    .filter((game) => !exhausted.has(`${game.pairKey}#${game.gameIndex}`));

  return {
    tournament,
    entries,
    scheduledCount: schedule.length,
    playedCount: played.length,
    abandonedCount: exhausted.size,
    remaining: outstanding,
    attempts,
  };
}

/** Increment and return the attempt count for a scheduled game. */
export async function recordAttempt(
  tournamentId: string,
  pairKey: string,
  gameIndex: number,
  lastError: string | null,
): Promise<number> {
  const now = Date.now();
  await (await getD1()).prepare(`INSERT INTO tournament_game_attempts (
      tournament_id, pair_key, game_index, attempts, last_error, updated_at
    ) VALUES (?, ?, ?, 1, ?, ?)
    ON CONFLICT(tournament_id, pair_key, game_index) DO UPDATE SET
      attempts = attempts + 1,
      last_error = excluded.last_error,
      updated_at = excluded.updated_at`)
    .bind(tournamentId, pairKey, gameIndex, lastError?.slice(0, 500) ?? null, now)
    .run();
  const row = await (await getD1())
    .prepare("SELECT attempts FROM tournament_game_attempts WHERE tournament_id = ? AND pair_key = ? AND game_index = ?")
    .bind(tournamentId, pairKey, gameIndex)
    .first<{ attempts: number }>();
  return row?.attempts ?? 1;
}

export async function getTournamentResults(tournamentId: string) {
  const tournament = await requireTournament(tournamentId);
  const [entries, games] = await Promise.all([
    listEntries(tournamentId),
    listTournamentGames(tournamentId),
  ]);
  const byModelPlayer = new Map(entries.map((entry) => [entry.modelPlayerId, entry.id]));
  const scored = games.flatMap((game) => {
    const whiteEntryId = byModelPlayer.get(game.whitePlayerId);
    const blackEntryId = byModelPlayer.get(game.blackPlayerId);
    if (!whiteEntryId || !blackEntryId) return [];
    return [{
      whiteEntryId,
      blackEntryId,
      result: game.result,
      pairKey: game.pairKey,
      moveList: movesOnly(game.moveList),
    }];
  });

  return {
    tournament,
    entries,
    games,
    runnerChanges: parseRunnerChanges(tournament.runnerChanges),
    ...standings(entries, scored),
  };
}

const TOURNAMENT_SELECT = `SELECT
      t.id, t.name, t.status, t.games_per_pair AS gamesPerPair,
      t.move_time_limit_ms AS moveTimeLimitMs, t.max_plies AS maxPlies,
      t.resident_budget_bytes AS residentBudgetBytes,
      t.max_attempts_per_game AS maxAttemptsPerGame,
      t.created_by_player_id AS createdByPlayerId, t.created_at AS createdAt,
      t.started_at AS startedAt, t.completed_at AS completedAt,
      t.runner_id AS runnerId, t.runner_label AS runnerLabel,
      t.runner_metadata AS runnerMetadata,
      t.runner_heartbeat_at AS runnerHeartbeatAt,
      t.runner_changes AS runnerChanges
    FROM tournaments t`;

async function requireTournament(id: string): Promise<Tournament> {
  const tournament = await getTournament(id);
  if (!tournament) throw new HistoryError(404, "That tournament does not exist.");
  return tournament;
}

function validateConfig(config: TournamentConfig): TournamentConfig {
  const name = String(config.name ?? "").trim().slice(0, 120);
  if (!name) throw new HistoryError(400, "A tournament needs a name.");
  const positive = (value: unknown, label: string, ceiling: number) => {
    if (!Number.isInteger(value) || (value as number) < 1 || (value as number) > ceiling) {
      throw new HistoryError(400, `${label} must be a whole number between 1 and ${ceiling}.`);
    }
    return value as number;
  };
  return {
    name,
    gamesPerPair: positive(config.gamesPerPair, "Games per pair", MAX_GAMES_PER_PAIR),
    moveTimeLimitMs: positive(config.moveTimeLimitMs, "The move time limit", 600_000),
    maxPlies: positive(config.maxPlies, "The ply cap", MAX_PLIES_CEILING),
    residentBudgetBytes: positive(config.residentBudgetBytes, "The resident budget", 8_000_000_000),
    maxAttemptsPerGame: positive(config.maxAttemptsPerGame, "Attempts per game", 20),
  };
}

function totalPackageBytes(entries: ReadonlyArray<TournamentEntry>): number {
  return entries.reduce((total, entry) => total + entry.packageBytes, 0);
}

function parseRunnerChanges(raw: string | null): RunnerChange[] {
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed as RunnerChange[] : [];
  } catch {
    return [];
  }
}

/** Strip PGN headers so identical games compare equal regardless of metadata. */
function movesOnly(pgn: string): string {
  return pgn.replace(/\[[^\]]*\]\s*/g, "").replace(/\s+/g, " ").trim();
}
