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

test("regular arena games make streaming opt-in and unlisted", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");
  const publisher = await readFile(new URL("../app/arena/live-game-publisher.ts", import.meta.url), "utf8");

  assert.match(arena, /useState\(false\)/);
  assert.match(arena, /Create an unlisted live link/);
  assert.match(arena, /source: "arena"/);
  assert.match(arena, /Open spectator view/);
  assert.match(publisher, /publisherToken/);
  assert.match(publisher, /\/watch\//);
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

