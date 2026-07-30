import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { DatabaseSync } from "node:sqlite";
import { buildModelDirectoryQuery, MODEL_VERSIONS_SQL } from "../lib/model-catalog-query.mjs";
import { selectModelVersion } from "../app/models/model-page.mjs";

async function catalogDatabase() {
  const db = new DatabaseSync(":memory:");
  db.exec(await readFile(new URL("../drizzle/0000_cold_raza.sql", import.meta.url), "utf8"));
  db.exec(await readFile(new URL("../drizzle/0001_first-class-models.sql", import.meta.url), "utf8"));
  db.exec(`
    INSERT INTO models VALUES
      ('alice/model', 'Alice', 1000), ('bob/model', 'Bob', 1500), ('carol/model', 'Carol', 3000);
    INSERT INTO players VALUES
      ('a1','model','a1','Alice v1','A1',1000,5000), ('a2','model','a2','Alice v2','A2',2000,6000),
      ('b1','model','b1','Bob','B1',1500,5500), ('c1','model','c1','Carol','C1',3000,3000),
      ('h1','human','h1','Human','H1',500,5000);
    INSERT INTO model_versions VALUES
      ('a1','alice/model','aaa','digest-a',1000), ('a2','alice/model','bbb','digest-b',2000),
      ('b1','bob/model','ccc','digest-c',1500), ('c1','carol/model','ddd','digest-d',3000);
    INSERT INTO games VALUES
      ('g1','a1','h1','Alice v1','Human','1-0','checkmate','1. e4',1,'history-v1',4000,4000),
      ('g2','a2','b1','Alice v2','Bob','1/2-1/2','draw','1. d4',1,'history-v1',5000,5000);
  `);
  return db;
}

function rows(db, query) {
  return db.prepare(query.sql).all(...query.bindings).map((row) => ({ ...row }));
}

test("catalog queries aggregate, search, sort, and cursor through repository models", async () => {
  const db = await catalogDatabase();
  const byGames = rows(db, buildModelDirectoryQuery({ sort: "games", limit: 4 }));
  assert.deepEqual(byGames.map((row) => [row.repository, row.games, row.versions]), [
    ["alice/model", 2, 2], ["bob/model", 1, 1], ["carol/model", 0, 1],
  ]);
  assert.equal(byGames[0].wins, 1);
  assert.equal(byGames[0].draws, 1);

  const searched = rows(db, buildModelDirectoryQuery({ search: "bbb", limit: 4 }));
  assert.deepEqual(searched.map((row) => row.repository), ["alice/model"]);

  const firstPage = rows(db, buildModelDirectoryQuery({ sort: "name", limit: 2 }));
  assert.deepEqual(firstPage.map((row) => row.displayName), ["Alice", "Bob"]);
  const secondPage = rows(db, buildModelDirectoryQuery({
    sort: "name",
    cursor: { value: "bob", id: "bob/model" },
    limit: 2,
  }));
  assert.deepEqual(secondPage.map((row) => row.displayName), ["Carol"]);
});

test("version queries and page selection are chronological and exact", async () => {
  const db = await catalogDatabase();
  const versions = db.prepare(MODEL_VERSIONS_SQL).all("alice/model").map((row) => ({ ...row }));
  assert.deepEqual(versions.map((version) => version.commitSha), ["bbb", "aaa"]);
  assert.equal(selectModelVersion(versions)?.commitSha, "bbb");
  assert.equal(selectModelVersion(versions, "aaa")?.commitSha, "aaa");
  assert.equal(selectModelVersion(versions, "missing"), null);
});
