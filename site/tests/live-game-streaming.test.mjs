import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { DatabaseSync } from "node:sqlite";

test("the live-game migration stores one recoverable snapshot per game", async () => {
  const migration = await readFile(
    new URL("../drizzle/0006_futuristic_wonder_man.sql", import.meta.url),
    "utf8",
  );
  const db = new DatabaseSync(":memory:");
  db.exec("CREATE TABLE tournaments (id TEXT PRIMARY KEY)");
  db.exec(migration);
  db.prepare(`INSERT INTO live_games (
      id, publisher_token_hash, source, white_name, black_name, phase, status,
      moves, revision, started_at, updated_at, expires_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
    .run(
      "game-1",
      "secret-hash",
      "arena",
      "White",
      "Black",
      "playing",
      "White to move",
      '["e4","e5"]',
      2,
      1,
      2,
      3,
    );

  assert.deepEqual(
    { ...db.prepare("SELECT source, moves, revision FROM live_games WHERE id = 'game-1'").get() },
    { source: "arena", moves: '["e4","e5"]', revision: 2 },
  );
  const indexes = db.prepare("SELECT name FROM sqlite_schema WHERE type = 'index'").all()
    .map((row) => row.name);
  assert.ok(indexes.includes("live_games_tournament_updated_idx"));
  assert.ok(indexes.includes("live_games_expires_idx"));
});

test("live event batches are ordered per game and expire after one hour", async () => {
  const initial = await readFile(
    new URL("../drizzle/0006_futuristic_wonder_man.sql", import.meta.url),
    "utf8",
  );
  const events = await readFile(
    new URL("../drizzle/0007_wandering_bedlam.sql", import.meta.url),
    "utf8",
  );
  const liveGames = await readFile(new URL("../lib/live-games.ts", import.meta.url), "utf8");
  const db = new DatabaseSync(":memory:");
  db.exec("PRAGMA foreign_keys = ON");
  db.exec("CREATE TABLE tournaments (id TEXT PRIMARY KEY)");
  db.exec(initial);
  db.exec(events);
  db.prepare(`INSERT INTO live_games (
      id, publisher_token_hash, source, white_name, black_name, phase, status,
      moves, revision, started_at, updated_at, expires_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
      "game-1", "secret-hash", "arena", "White", "Black", "playing",
      "White to move", "[]", 1, 1, 2, 3,
    );
  db.prepare(`INSERT INTO live_game_event_batches (
      game_id, batch_index, first_seq, last_seq, events, expires_at
    ) VALUES (?, ?, ?, ?, ?, ?)`).run("game-1", 1, 1, 2, "[]", 3);

  assert.deepEqual(
    { ...db.prepare("SELECT event_seq FROM live_games WHERE id = 'game-1'").get() },
    { event_seq: 0 },
  );
  assert.match(liveGames, /LIVE_GAME_TTL_MS = 60 \* 60 \* 1_000/);
  db.prepare("DELETE FROM live_games WHERE id = ?").run("game-1");
  assert.equal(db.prepare("SELECT COUNT(*) AS count FROM live_game_event_batches").get().count, 0);
});

test("live snapshots preserve per-side move clocks for remote spectators", async () => {
  const initial = await readFile(
    new URL("../drizzle/0006_futuristic_wonder_man.sql", import.meta.url),
    "utf8",
  );
  const clocks = await readFile(
    new URL("../drizzle/0008_past_cerebro.sql", import.meta.url),
    "utf8",
  );
  const db = new DatabaseSync(":memory:");
  db.exec("CREATE TABLE tournaments (id TEXT PRIMARY KEY)");
  db.exec(initial);
  db.exec(clocks);

  const columns = db.prepare("PRAGMA table_info(live_games)").all().map((row) => row.name);
  assert.ok(columns.includes("white_move_time_limit_ms"));
  assert.ok(columns.includes("black_move_time_limit_ms"));
  assert.ok(columns.includes("active_turn_color"));
  assert.ok(columns.includes("active_turn_elapsed_ms"));
});

test("regular arena games make streaming opt-in and unlisted", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");
  const publisher = await readFile(new URL("../app/arena/live-game-publisher.ts", import.meta.url), "utf8");
  const viewer = await readFile(new URL("../app/watch/[id]/live-game-viewer.tsx", import.meta.url), "utf8");

  assert.match(arena, /useState\(false\)/);
  assert.match(arena, /Create an unlisted live link/);
  assert.match(arena, /source: "arena"/);
  assert.match(arena, /Open spectator view/);
  assert.match(publisher, /publisherToken/);
  assert.match(publisher, /\/watch\//);
  assert.match(viewer, /if \(!livePhase \|\| livePhase === "finished"\) return/);
});

test("spectators poll ordered batches because Sites does not flush SSE responses", async () => {
  const viewer = await readFile(new URL("../app/watch/[id]/live-game-viewer.tsx", import.meta.url), "utf8");

  assert.doesNotMatch(viewer, /new EventSource/);
  assert.doesNotMatch(viewer, /pollFallback/);
  assert.match(viewer, /POLL_INTERVAL_MS = 500/);
  assert.match(viewer, /Live updates · 500 ms polling/);
  assert.match(viewer, /if \(refreshing \|\| stopped\) return/);
  assert.match(viewer, /for \(const batch of next\.batches\) consumeBatch\(batch\)/);
  assert.match(viewer, /window\.setInterval\(\(\) => void refresh\(\), POLL_INTERVAL_MS\)/);
});

test("tournament games broadcast automatically without affecting permanent scoring", async () => {
  const runner = await readFile(
    new URL("../app/tournaments/[id]/run/tournament-runner.tsx", import.meta.url),
    "utf8",
  );
  const results = await readFile(
    new URL("../app/api/tournaments/[id]/results/route.ts", import.meta.url),
    "utf8",
  );

  assert.match(runner, /source: "tournament"/);
  assert.match(runner, /await broadcaster\.publish/);
  assert.match(runner, /await save\(gameId/);
  assert.match(results, /getTournamentLiveGame/);
  assert.match(results, /scheduledCount/);
});

test("the tournament broadcast puts ranked progress beside active game tiles", async () => {
  const broadcast = await readFile(
    new URL("../app/tournaments/[id]/tournament-broadcast.tsx", import.meta.url),
    "utf8",
  );
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(broadcast, /className="tournament-broadcast-dashboard"/);
  assert.match(broadcast, /className="tournament-broadcast-rail"/);
  assert.match(broadcast, /className="tournament-game-grid"/);
  assert.match(broadcast, /className="tournament-live-card"/);
  assert.match(broadcast, /role="grid" aria-label="Current tournament position"/);
  assert.match(styles, /\.tournament-broadcast-dashboard\s*\{[^}]*grid-template-columns:\s*minmax\(18rem,\s*22rem\) minmax\(0,\s*1fr\)/s);
  assert.match(styles, /@media \(max-width:\s*900px\)\s*\{[\s\S]*?\.tournament-broadcast-dashboard\s*\{[^}]*grid-template-columns:\s*1fr/s);
});

test("tournament boards inherit the chess-square palette outside the arena page", async () => {
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const root = styles.match(/:root\s*\{([^}]*)\}/s)?.[1] ?? "";

  assert.match(root, /--board-light:\s*#eadfca/);
  assert.match(root, /--board-dark:\s*#386451/);
  assert.match(styles, /\.board-square\.light\s*\{[^}]*background:\s*var\(--board-light\)/s);
  assert.match(styles, /\.board-square\.dark\s*\{[^}]*background:\s*var\(--board-dark\)/s);
});

test("the tournament page only renders controls for its current phase", async () => {
  const [page, watcher] = await Promise.all([
    readFile(new URL("../app/tournaments/[id]/page.tsx", import.meta.url), "utf8"),
    readFile(
      new URL("../app/tournaments/[id]/tournament-phase-watcher.tsx", import.meta.url),
      "utf8",
    ),
  ]);

  assert.match(
    page,
    /tournament\.status === "registration"\s*&&\s*\([\s\S]*?<RegisterEntryPanel/,
  );
  assert.match(
    page,
    /tournament\.status !== "registration"\s*&&\s*\([\s\S]*?<TournamentBroadcast/,
  );
  assert.match(page, /<TournamentPhaseWatcher[\s\S]*?status=\{tournament\.status\}/);
  assert.match(watcher, /next\.status !== status/);
  assert.match(watcher, /window\.location\.reload\(\)/);
});

test("the tournament spectator follows each game with optional thinking", async () => {
  const broadcast = await readFile(
    new URL("../app/tournaments/[id]/tournament-broadcast.tsx", import.meta.url),
    "utf8",
  );

  assert.match(broadcast, /key=\{state\.liveGame\.id\}/);
  assert.match(
    broadcast,
    /fetch\(\s*`\/api\/live-games\/\$\{encodeURIComponent\(game\.id\)\}\?after=\$\{cursor\.current\}`/,
  );
  assert.match(broadcast, /<ThinkingOverlay enabled=\{showThinking\}/);
  assert.match(broadcast, /type="checkbox"\s+checked=\{showThinking\}/);
  assert.match(broadcast, /const POLL_INTERVAL_MS = 500/);
  const liveGames = await readFile(new URL("../lib/live-games.ts", import.meta.url), "utf8");
  assert.match(
    liveGames,
    /getTournamentLiveGame[\s\S]*?ORDER BY CASE WHEN phase != 'finished' THEN 0 ELSE 1 END,[\s\S]*?LIMIT 1/,
  );
});

test("tournament spectator reads are public across computers", async () => {
  const [resultsRoute, liveRoute] = await Promise.all([
    readFile(
      new URL("../app/api/tournaments/[id]/results/route.ts", import.meta.url),
      "utf8",
    ),
    readFile(new URL("../app/api/live-games/[id]/route.ts", import.meta.url), "utf8"),
  ]);

  assert.match(resultsRoute, /export async function GET/);
  assert.match(resultsRoute, /getTournamentLiveGame\(id\)/);
  assert.doesNotMatch(resultsRoute, /getChatGPTUser|Authorization/);
  assert.match(liveRoute, /export async function GET/);
  assert.match(liveRoute, /getLiveGameResponse\(id, after\)/);
  assert.doesNotMatch(liveRoute, /getChatGPTUser|Authorization/);
});

test("arena and spectator boards share player material, flip, and move-clock chrome", async () => {
  const [arena, viewer, broadcast, playerStrip, publisher, liveGames, runner, styles] = await Promise.all([
    readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/watch/[id]/live-game-viewer.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/tournaments/[id]/tournament-broadcast.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/arena/player-strip.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/arena/live-game-publisher.ts", import.meta.url), "utf8"),
    readFile(new URL("../lib/live-games.ts", import.meta.url), "utf8"),
    readFile(new URL("../app/tournaments/[id]/run/tournament-runner.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);

  for (const surface of [arena, viewer, broadcast]) {
    assert.match(surface, /<PlayerStrip/);
    assert.match(surface, /onFlip=/);
  }
  assert.match(arena, /boardFlipped/);
  assert.match(viewer, /setOrientation/);
  assert.match(broadcast, /setOrientation/);
  assert.match(playerStrip, /className="captured-pieces"/);
  assert.match(playerStrip, /className="material-lead"/);
  assert.match(playerStrip, /role="timer"/);
  assert.match(playerStrip, /aria-label="Flip board"/);
  assert.match(styles, /\.move-clock\s*\{[^}]*conic-gradient/s);
  assert.match(styles, /\.board-flip-button\s*\{/);
  assert.match(publisher, /activeTurnElapsedMs/);
  assert.match(liveGames, /whiteMoveTimeLimitMs/);
  assert.match(runner, /whiteMoveTimeLimitMs:\s*current\.tournament\.moveTimeLimitMs/);
  assert.match(viewer, /orientation === "w" \? blackSummary : whiteSummary/);
  assert.match(broadcast, /orientation === "w" \? blackSummary : whiteSummary/);
});
