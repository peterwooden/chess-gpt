import assert from "node:assert/strict";
import test from "node:test";

import {
  createThinkingCommandLimiter,
  normalizeThinkingCommand,
} from "../lib/thinking-events.mjs";

test("thinking drawing commands receive the agreed defaults", () => {
  assert.deepEqual(
    normalizeThinkingCommand({ type: "highlightSquare", square: "e4" }),
    { type: "highlightSquare", square: "e4", intensity: 1, fadeMs: 500 },
  );
  assert.deepEqual(
    normalizeThinkingCommand({
      type: "drawArrow",
      from: "g1",
      to: "f3",
      intensity: 0.4,
      fadeMs: 1200,
      side: "opponent",
    }),
    {
      type: "drawArrow",
      from: "g1",
      to: "f3",
      intensity: 0.4,
      fadeMs: 1200,
      side: "opponent",
    },
  );
});

test("thinking commands reject invalid cosmetic data without throwing", () => {
  assert.equal(normalizeThinkingCommand({ type: "highlightSquare", square: "e9" }), null);
  assert.equal(normalizeThinkingCommand({ type: "drawArrow", from: "e2", to: "e2" }), null);
  assert.equal(normalizeThinkingCommand({ type: "drawArrow", from: "e2", to: "e4", side: "white" }), null);
  assert.equal(normalizeThinkingCommand({ type: "highlightSquare", square: "e4", intensity: 1.1 }), null);
  assert.equal(normalizeThinkingCommand({ type: "highlightSquare", square: "e4", fadeMs: -1 }), null);
  assert.equal(normalizeThinkingCommand({ type: "clearArrow", from: "a1", to: "a9" }), null);
  assert.deepEqual(normalizeThinkingCommand({ type: "clearAll" }), { type: "clearAll" });
});

test("the package worker accepts at most 128 commands in each 500 ms window", () => {
  let now = 1000;
  const limiter = createThinkingCommandLimiter(() => now);
  for (let index = 0; index < 128; index += 1) {
    assert.equal(limiter.accept(), true);
  }
  assert.equal(limiter.accept(), false);
  now = 1499;
  assert.equal(limiter.accept(), false);
  now = 1500;
  assert.equal(limiter.accept(), true);
});
