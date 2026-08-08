import assert from "node:assert/strict";
import test from "node:test";

import {
  annotationOpacity,
  arrowSide,
  thinkingArrowShape,
  thinkingArrowPoints,
} from "../lib/thinking-visuals.mjs";

test("annotations decay by newer annotation count and reach zero at sixty-four", () => {
  assert.equal(annotationOpacity(1, 0), 0.8);
  assert.equal(annotationOpacity(1, 32), 0.4);
  assert.equal(annotationOpacity(1, 63) > 0, true);
  assert.equal(annotationOpacity(1, 64), 0);
  assert.equal(annotationOpacity(0.5, 0), 0.4);
});

test("knight arrows take an orthogonal L-shaped path", () => {
  assert.deepEqual(thinkingArrowPoints("e4", "f6", "w"), [
    { x: 56.25, y: 56.25 },
    { x: 56.25, y: 31.25 },
    { x: 68.75, y: 31.25 },
  ]);
  assert.deepEqual(thinkingArrowPoints("e2", "e4", "w"), [
    { x: 56.25, y: 81.25 },
    { x: 56.25, y: 56.25 },
  ]);
});

test("the arrow shaft stops at the arrowhead base instead of painting underneath it", () => {
  assert.deepEqual(thinkingArrowShape("e2", "e4", "w"), {
    shaft: [
      { x: 56.25, y: 81.25 },
      { x: 56.25, y: 60.25 },
    ],
    head: [
      { x: 56.25, y: 56.25 },
      { x: 59.45, y: 60.25 },
      { x: 53.05, y: 60.25 },
    ],
  });
});

test("arrows distinguish the thinking side from its opponent", () => {
  assert.equal(arrowSide("w", "w"), "own");
  assert.equal(arrowSide("w", "b"), "opponent");
  assert.equal(arrowSide("b", null), "own");
});
