import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { DatabaseSync } from "node:sqlite";

test("the model catalog migration groups legacy checkpoints without changing game identities", async () => {
  const initial = await readFile(new URL("../drizzle/0000_cold_raza.sql", import.meta.url), "utf8");
  const migration = await readFile(new URL("../drizzle/0001_first-class-models.sql", import.meta.url), "utf8");
  const db = new DatabaseSync(":memory:");
  db.exec(initial);
  db.exec(`
    INSERT INTO players VALUES
      ('v1', 'model', 'model:alice/model@aaa', 'Old name', 'VERSION1', 1000, 3000),
      ('v2', 'model', 'model:alice/model@bbb', 'New name', 'VERSION2', 2000, 4000);
    INSERT INTO model_versions VALUES
      ('v1', 'alice/model', 'aaa', 'digest-a'),
      ('v2', 'alice/model', 'bbb', 'digest-b');
    INSERT INTO games VALUES
      ('game-1', 'v1', 'v2', 'Old name', 'New name', '1-0', 'checkmate', '1. e4', 1, 'history-v1', 2500, 2500);
  `);
  db.exec(migration);

  assert.deepEqual(
    { ...db.prepare("SELECT repository, display_name, first_seen_at FROM models").get() },
    { repository: "alice/model", display_name: "New name", first_seen_at: 1000 },
  );
  assert.deepEqual(
    db.prepare("SELECT player_id, commit_sha, first_seen_at FROM model_versions ORDER BY first_seen_at").all().map((row) => ({ ...row })),
    [
      { player_id: "v1", commit_sha: "aaa", first_seen_at: 1000 },
      { player_id: "v2", commit_sha: "bbb", first_seen_at: 2000 },
    ],
  );
  assert.deepEqual(
    { ...db.prepare("SELECT white_player_id, black_player_id FROM games WHERE id = 'game-1'").get() },
    { white_player_id: "v1", black_player_id: "v2" },
  );
});
