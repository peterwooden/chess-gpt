import { index, integer, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";

export const players = sqliteTable("players", {
  id: text("id").primaryKey(),
  kind: text("kind", { enum: ["human", "model"] }).notNull(),
  identityKey: text("identity_key").notNull(),
  displayName: text("display_name").notNull(),
  playerCode: text("player_code").notNull(),
  createdAt: integer("created_at", { mode: "timestamp_ms" }).notNull(),
  lastPlayedAt: integer("last_played_at", { mode: "timestamp_ms" }).notNull(),
}, (table) => [
  uniqueIndex("players_identity_key_unique").on(table.identityKey),
  uniqueIndex("players_player_code_unique").on(table.playerCode),
  index("players_kind_last_played_idx").on(table.kind, table.lastPlayedAt, table.id),
  index("players_display_name_idx").on(table.displayName, table.id),
]);

export const models = sqliteTable("models", {
  repository: text("repository").primaryKey(),
  displayName: text("display_name").notNull(),
  firstSeenAt: integer("first_seen_at", { mode: "timestamp_ms" }).notNull(),
}, (table) => [
  index("models_display_name_idx").on(table.displayName, table.repository),
  index("models_first_seen_idx").on(table.firstSeenAt, table.repository),
]);

export const modelVersions = sqliteTable("model_versions", {
  playerId: text("player_id").primaryKey().references(() => players.id),
  repository: text("repository").notNull().references(() => models.repository),
  commitSha: text("commit_sha").notNull(),
  manifestSha256: text("manifest_sha256").notNull(),
  firstSeenAt: integer("first_seen_at", { mode: "timestamp_ms" }).notNull(),
}, (table) => [
  uniqueIndex("model_versions_repository_commit_unique").on(table.repository, table.commitSha),
  index("model_versions_repository_idx").on(table.repository),
  index("model_versions_commit_idx").on(table.commitSha),
]);

export const tournaments = sqliteTable("tournaments", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  status: text("status", { enum: ["registration", "running", "completed"] }).notNull(),
  gamesPerPair: integer("games_per_pair").notNull(),
  moveTimeLimitMs: integer("move_time_limit_ms").notNull(),
  residentBudgetBytes: integer("resident_budget_bytes").notNull(),
  maxAttemptsPerGame: integer("max_attempts_per_game").notNull(),
  createdByPlayerId: text("created_by_player_id").notNull().references(() => players.id),
  createdAt: integer("created_at", { mode: "timestamp_ms" }).notNull(),
  startedAt: integer("started_at", { mode: "timestamp_ms" }),
  completedAt: integer("completed_at", { mode: "timestamp_ms" }),
  runnerId: text("runner_id"),
  runnerLabel: text("runner_label"),
  runnerMetadata: text("runner_metadata"),
  runnerHeartbeatAt: integer("runner_heartbeat_at", { mode: "timestamp_ms" }),
  runnerChanges: text("runner_changes").notNull().default("[]"),
  // Whether this tournament samples openings from the arena book. Off means
  // every game starts from the standard position, as under the 2026-07-31
  // protocol.
  openingBook: integer("opening_book", { mode: "boolean" }).notNull().default(false),
  // JSON array of sampled openings, set when an opening-book tournament leaves
  // registration. NULL when the book is off or the tournament predates it.
  openings: text("openings"),
}, (table) => [
  index("tournaments_status_created_idx").on(table.status, table.createdAt, table.id),
  index("tournaments_created_idx").on(table.createdAt, table.id),
]);

export const tournamentEntries = sqliteTable("tournament_entries", {
  id: text("id").primaryKey(),
  tournamentId: text("tournament_id").notNull().references(() => tournaments.id),
  ownerPlayerId: text("owner_player_id").notNull().references(() => players.id),
  modelPlayerId: text("model_player_id").notNull().references(() => players.id),
  reference: text("reference").notNull(),
  manifestSha256: text("manifest_sha256").notNull(),
  displayName: text("display_name").notNull(),
  packageBytes: integer("package_bytes").notNull(),
  verifiedAt: integer("verified_at", { mode: "timestamp_ms" }),
  smokeMoveCount: integer("smoke_move_count"),
  smokeMedianMs: integer("smoke_median_ms"),
  smokeP95Ms: integer("smoke_p95_ms"),
  createdAt: integer("created_at", { mode: "timestamp_ms" }).notNull(),
  updatedAt: integer("updated_at", { mode: "timestamp_ms" }).notNull(),
}, (table) => [
  uniqueIndex("tournament_entries_model_unique").on(table.tournamentId, table.modelPlayerId),
  index("tournament_entries_tournament_idx").on(table.tournamentId, table.id),
  index("tournament_entries_owner_idx").on(table.tournamentId, table.ownerPlayerId),
]);

/**
 * Persisted attempt counters for scheduled games. Without these a page reload
 * would reset the counter, so a package that reliably crashes the runner could
 * stall a tournament forever.
 */
export const tournamentGameAttempts = sqliteTable("tournament_game_attempts", {
  tournamentId: text("tournament_id").notNull().references(() => tournaments.id),
  pairKey: text("pair_key").notNull(),
  gameIndex: integer("game_index").notNull(),
  attempts: integer("attempts").notNull(),
  lastError: text("last_error"),
  updatedAt: integer("updated_at", { mode: "timestamp_ms" }).notNull(),
}, (table) => [
  uniqueIndex("tournament_game_attempts_key").on(table.tournamentId, table.pairKey, table.gameIndex),
]);

/**
 * Ephemeral, public snapshots for games that are still being played. The
 * permanent `games` row remains authoritative once a game finishes.
 */
export const liveGames = sqliteTable("live_games", {
  id: text("id").primaryKey(),
  publisherTokenHash: text("publisher_token_hash").notNull(),
  source: text("source", { enum: ["arena", "tournament"] }).notNull(),
  tournamentId: text("tournament_id").references(() => tournaments.id),
  tournamentPairKey: text("tournament_pair_key"),
  tournamentGameIndex: integer("tournament_game_index"),
  whiteName: text("white_name").notNull(),
  blackName: text("black_name").notNull(),
  whiteModelReference: text("white_model_reference"),
  blackModelReference: text("black_model_reference"),
  openingName: text("opening_name"),
  phase: text("phase", { enum: ["starting", "playing", "paused", "finished"] }).notNull(),
  status: text("status").notNull(),
  moves: text("moves").notNull(),
  lastMoveMs: integer("last_move_ms"),
  result: text("result", { enum: ["1-0", "0-1", "1/2-1/2"] }),
  revision: integer("revision").notNull(),
  startedAt: integer("started_at", { mode: "timestamp_ms" }).notNull(),
  updatedAt: integer("updated_at", { mode: "timestamp_ms" }).notNull(),
  expiresAt: integer("expires_at", { mode: "timestamp_ms" }).notNull(),
}, (table) => [
  index("live_games_tournament_updated_idx").on(table.tournamentId, table.updatedAt),
  index("live_games_expires_idx").on(table.expiresAt),
]);

export const games = sqliteTable("games", {
  id: text("id").primaryKey(),
  whitePlayerId: text("white_player_id").references(() => players.id),
  blackPlayerId: text("black_player_id").references(() => players.id),
  whiteName: text("white_name").notNull(),
  blackName: text("black_name").notNull(),
  result: text("result", { enum: ["1-0", "0-1", "1/2-1/2"] }).notNull(),
  termination: text("termination").notNull(),
  pgn: text("pgn").notNull(),
  moveCount: integer("move_count").notNull(),
  arenaVersion: text("arena_version").notNull(),
  tournamentId: text("tournament_id").references(() => tournaments.id),
  tournamentPairKey: text("tournament_pair_key"),
  tournamentGameIndex: integer("tournament_game_index"),
  playedAt: integer("played_at", { mode: "timestamp_ms" }).notNull(),
  recordedAt: integer("recorded_at", { mode: "timestamp_ms" }).notNull(),
}, (table) => [
  index("games_white_recorded_idx").on(table.whitePlayerId, table.recordedAt, table.id),
  index("games_black_recorded_idx").on(table.blackPlayerId, table.recordedAt, table.id),
  index("games_recorded_idx").on(table.recordedAt, table.id),
  // NULL tournament ids stay distinct in SQLite, so casual arena games never collide.
  uniqueIndex("games_tournament_schedule_unique")
    .on(table.tournamentId, table.tournamentPairKey, table.tournamentGameIndex),
  index("games_tournament_idx").on(table.tournamentId, table.recordedAt, table.id),
]);
