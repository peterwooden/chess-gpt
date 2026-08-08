import assert from "node:assert/strict";
import test from "node:test";
import { Chess } from "chess.js";
import {
  OPENING_BOOK,
  openingPool,
  sampleOpenings,
  openingForSlot,
  parseOpenings,
} from "../lib/opening-book.mjs";

/** The same PRNG the model worker uses, so draws are reproducible in tests. */
function mulberry32(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

test("every book line is legal from the standard start", () => {
  for (const line of OPENING_BOOK) {
    const game = new Chess();
    for (const san of line.moves) {
      let move = null;
      try {
        move = game.move(san);
      } catch {
        move = null;
      }
      assert.ok(move, `${line.name}: “${san}” is not legal after ${game.history().join(" ") || "the start"}`);
      assert.equal(move.san, san, `${line.name}: “${san}” is not canonical SAN (expected “${move.san}”)`);
    }
    assert.ok(!game.isGameOver(), `${line.name} ends the game inside the book`);
  }
});

test("book lines are even length so white is always to move after the book", () => {
  for (const line of OPENING_BOOK) {
    assert.equal(line.moves.length % 2, 0, `${line.name} has odd length ${line.moves.length}`);
    assert.ok(line.moves.length >= 6, `${line.name} is shorter than the minimum truncation`);
  }
});

test("the pool truncates at varying ply depths without duplicate positions", () => {
  const pool = openingPool();
  const keys = new Set(pool.map((opening) => opening.moves.join(" ")));
  assert.equal(keys.size, pool.length);
  const depths = new Set(pool.map((opening) => opening.moves.length));
  assert.ok(depths.size >= 3, "expected truncations at several ply depths");
  for (const depth of depths) assert.equal(depth % 2, 0);
});

test("sampling is deterministic for a given random source and unique until the pool cycles", () => {
  const first = sampleOpenings(40, mulberry32(7));
  const second = sampleOpenings(40, mulberry32(7));
  assert.deepEqual(first, second);

  const keys = new Set(first.map((opening) => opening.moves.join(" ")));
  assert.equal(keys.size, 40, "a draw within the pool size must not repeat an opening");

  const poolSize = openingPool().length;
  const oversized = sampleOpenings(poolSize + 5, mulberry32(3));
  assert.equal(oversized.length, poolSize + 5);
  assert.deepEqual(oversized[poolSize], oversized[0], "an oversized draw cycles the shuffled pool");
});

test("games 2i and 2i+1 share opening i", () => {
  const openings = sampleOpenings(5, mulberry32(1));
  for (let gameIndex = 0; gameIndex < 10; gameIndex += 1) {
    assert.deepEqual(openingForSlot(openings, gameIndex), openings[Math.floor(gameIndex / 2)]);
  }
});

test("parseOpenings tolerates rows from before opening sampling", () => {
  assert.equal(parseOpenings(null), null);
  assert.equal(parseOpenings(undefined), null);
  assert.equal(parseOpenings(""), null);
  assert.equal(parseOpenings("not json"), null);
  assert.equal(parseOpenings("[]"), null);
  const openings = sampleOpenings(3, mulberry32(2));
  assert.deepEqual(parseOpenings(JSON.stringify(openings)), openings);
});
