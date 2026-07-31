import assert from "node:assert/strict";
import test from "node:test";

import {
  buildSchedule,
  pairKey,
  pairings,
  remainingGames,
  standings,
} from "../lib/tournament-schedule.mjs";

const THREE = [{ id: "b" }, { id: "c" }, { id: "a" }];

test("a pairing key does not depend on argument order", () => {
  assert.equal(pairKey("a", "b"), pairKey("b", "a"));
});

test("pairings are every unordered pair, deterministically ordered", () => {
  assert.deepEqual(pairings(THREE).map((pair) => pair.key), ["a:b", "a:c", "b:c"]);
  assert.deepEqual(pairings([...THREE].reverse()).map((pair) => pair.key), ["a:b", "a:c", "b:c"]);
});

test("the schedule is a pure function of entries and games per pair", () => {
  const first = buildSchedule(THREE, 4);
  const second = buildSchedule([...THREE].reverse(), 4);
  assert.deepEqual(first, second);
  assert.equal(first.length, 3 * 4);
});

test("interleaving keeps every pairing within one game of the others", () => {
  const schedule = buildSchedule(THREE, 10);

  // Stop the run anywhere and check the pairings stay balanced.
  for (const stop of [1, 4, 7, 11, 19, 25]) {
    const counts = new Map();
    for (const game of schedule.slice(0, stop)) {
      counts.set(game.pairKey, (counts.get(game.pairKey) ?? 0) + 1);
    }
    const played = [...counts.values()];
    const missing = pairings(THREE).length - counts.size;
    const low = missing > 0 ? 0 : Math.min(...played);
    assert.ok(
      Math.max(...played) - low <= 1,
      `pairings diverged by more than one game after ${stop} games`,
    );
  }
});

test("pair-major ordering starves later pairings, which is why it is not the default", () => {
  const schedule = buildSchedule(THREE, 10, { order: "pair-major" });
  const firstTen = new Set(schedule.slice(0, 10).map((game) => game.pairKey));
  assert.deepEqual([...firstTen], ["a:b"]);
});

test("colors alternate so a partial run stays balanced", () => {
  const schedule = buildSchedule([{ id: "a" }, { id: "b" }], 4);
  assert.deepEqual(
    schedule.map((game) => game.whiteEntryId),
    ["a", "b", "a", "b"],
  );
  assert.deepEqual(
    schedule.map((game) => game.blackEntryId),
    ["b", "a", "b", "a"],
  );
});

test("resume is the set difference of recorded games", () => {
  const schedule = buildSchedule(THREE, 3);
  const recorded = [
    { pairKey: "a:b", gameIndex: 0 },
    { pairKey: "b:c", gameIndex: 1 },
  ];

  const remaining = remainingGames(schedule, recorded);

  assert.equal(remaining.length, schedule.length - 2);
  assert.ok(!remaining.some((game) => game.pairKey === "a:b" && game.gameIndex === 0));
  assert.ok(!remaining.some((game) => game.pairKey === "b:c" && game.gameIndex === 1));
});

test("resume out of order still yields the right remaining set", () => {
  const schedule = buildSchedule(THREE, 2);
  const recorded = schedule.slice().reverse().slice(0, 4);
  const remaining = remainingGames(schedule, recorded);
  assert.equal(remaining.length, schedule.length - 4);
});

test("resuming a complete tournament leaves nothing to play", () => {
  const schedule = buildSchedule(THREE, 2);
  assert.deepEqual(remainingGames(schedule, schedule), []);
});

test("a rejected games-per-pair is caught rather than silently producing nothing", () => {
  assert.throws(() => buildSchedule(THREE, 0), /positive integer/);
  assert.throws(() => buildSchedule(THREE, 1.5), /positive integer/);
});

const ENTRIES = [
  { id: "a", displayName: "Alpha" },
  { id: "b", displayName: "Beta" },
  { id: "c", displayName: "Gamma" },
];

test("standings score wins, draws and losses from both seats", () => {
  const { table } = standings(ENTRIES, [
    { pairKey: "a:b", whiteEntryId: "a", blackEntryId: "b", result: "1-0" },
    { pairKey: "a:b", whiteEntryId: "b", blackEntryId: "a", result: "1/2-1/2" },
    { pairKey: "a:c", whiteEntryId: "c", blackEntryId: "a", result: "0-1" },
  ]);

  const alpha = table.find((row) => row.entryId === "a");
  assert.equal(alpha.points, 2.5);
  assert.equal(alpha.wins, 2);
  assert.equal(alpha.draws, 1);
  assert.equal(alpha.losses, 0);
  assert.equal(alpha.games, 3);
  assert.equal(table[0].entryId, "a");
});

test("equal points share rank one rather than being separated", () => {
  const { table, shared } = standings(ENTRIES.slice(0, 2), [
    { pairKey: "a:b", whiteEntryId: "a", blackEntryId: "b", result: "1-0" },
    { pairKey: "a:b", whiteEntryId: "b", blackEntryId: "a", result: "1-0" },
  ]);

  assert.equal(shared, true);
  assert.deepEqual(table.map((row) => row.rank), [1, 1]);
  assert.deepEqual(table.map((row) => row.points), [1, 1]);
});

test("a decisive tournament does not report a shared title", () => {
  const { shared, table } = standings(ENTRIES.slice(0, 2), [
    { pairKey: "a:b", whiteEntryId: "a", blackEntryId: "b", result: "1-0" },
    { pairKey: "a:b", whiteEntryId: "b", blackEntryId: "a", result: "0-1" },
  ]);

  assert.equal(shared, false);
  assert.deepEqual(table.map((row) => row.rank), [1, 2]);
});

test("distinct game counts expose a deterministic pairing", () => {
  const replay = "e4 e5 Nf3";
  const { distinctGamesByPair } = standings(ENTRIES, [
    { pairKey: "a:b", whiteEntryId: "a", blackEntryId: "b", result: "1-0", moveList: replay },
    { pairKey: "a:b", whiteEntryId: "a", blackEntryId: "b", result: "1-0", moveList: replay },
    { pairKey: "a:b", whiteEntryId: "a", blackEntryId: "b", result: "1-0", moveList: replay },
    { pairKey: "a:c", whiteEntryId: "a", blackEntryId: "c", result: "1-0", moveList: "d4" },
    { pairKey: "a:c", whiteEntryId: "a", blackEntryId: "c", result: "0-1", moveList: "c4" },
  ]);

  assert.equal(distinctGamesByPair["a:b"], 1);
  assert.equal(distinctGamesByPair["a:c"], 2);
});

test("games referencing an unknown entry are ignored rather than crashing", () => {
  const { table } = standings(ENTRIES, [
    { pairKey: "a:z", whiteEntryId: "a", blackEntryId: "z", result: "1-0" },
  ]);
  assert.equal(table.find((row) => row.entryId === "a").games, 0);
});
