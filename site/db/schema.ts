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
  playedAt: integer("played_at", { mode: "timestamp_ms" }).notNull(),
  recordedAt: integer("recorded_at", { mode: "timestamp_ms" }).notNull(),
}, (table) => [
  index("games_white_recorded_idx").on(table.whitePlayerId, table.recordedAt, table.id),
  index("games_black_recorded_idx").on(table.blackPlayerId, table.recordedAt, table.id),
  index("games_recorded_idx").on(table.recordedAt, table.id),
]);
