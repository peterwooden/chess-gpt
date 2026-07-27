import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  buildGameReview,
  classifyWinningChanceLoss,
  moveAccuracy,
  winPercentFromCentiPawns,
} from "../app/arena/stockfish-review.mjs";

test("winning chances are symmetric and centipawn values are capped", () => {
  assert.equal(winPercentFromCentiPawns(0), 50);
  assert.ok(Math.abs(winPercentFromCentiPawns(400) + winPercentFromCentiPawns(-400) - 100) < 1e-9);
  assert.equal(winPercentFromCentiPawns(10_000), winPercentFromCentiPawns(1000));
});

test("Lichess-style judgements use winning-chance loss thresholds", () => {
  assert.equal(classifyWinningChanceLoss(9.99), null);
  assert.equal(classifyWinningChanceLoss(10), "inaccuracy");
  assert.equal(classifyWinningChanceLoss(20), "mistake");
  assert.equal(classifyWinningChanceLoss(30), "blunder");
});

test("a move that preserves or improves winning chances is 100 percent accurate", () => {
  assert.equal(moveAccuracy(40, 40), 100);
  assert.equal(moveAccuracy(40, 50), 100);
  assert.ok(moveAccuracy(70, 40) < moveAccuracy(70, 60));
});

test("game review scores losses from the player who made each move", () => {
  const review = buildGameReview([
    { whiteScore: 0, bestMoveSan: "e4" },
    { whiteScore: -400, bestMoveSan: "e5" },
    { whiteScore: -400, bestMoveSan: "Nf3" },
    { whiteScore: -400, bestMoveSan: "Nc6" },
    { whiteScore: 0, bestMoveSan: null },
  ]);

  assert.equal(review.moves[0].judgement, "blunder");
  assert.equal(review.moves[1].judgement, null);
  assert.equal(review.moves[3].judgement, "blunder");
  assert.equal(review.players.w.counts.blunder, 1);
  assert.equal(review.players.b.counts.blunder, 1);
  assert.ok(review.players.w.accuracy < 100);
  assert.ok(review.players.b.accuracy < 100);
});

test("the published engine is the pinned npm distribution", async () => {
  const published = await readFile(new URL("../public/stockfish/stockfish-18-lite-single.wasm", import.meta.url));
  const installed = await readFile(new URL("../node_modules/stockfish/bin/stockfish-18-lite-single.wasm", import.meta.url));
  const digest = (bytes) => createHash("sha256").update(bytes).digest("hex");

  assert.equal(digest(published), digest(installed));
});
