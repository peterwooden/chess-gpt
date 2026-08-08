import assert from "node:assert/strict";
import test from "node:test";

import {
  createLiveGameEventSequencer,
  normalizeLiveGameEventBatch,
} from "../lib/live-game-events.mjs";

test("one game stream preserves thinking and move order with capture offsets", () => {
  let now = 1000;
  const stream = createLiveGameEventSequencer(() => now);
  stream.record({ type: "turn.started", turnId: "turn-1", ply: 1, color: "w" });
  now = 1080;
  stream.record({
    type: "thinking.command",
    turnId: "turn-1",
    command: { type: "highlightSquare", square: "e4", intensity: 1, fadeMs: 500 },
  });
  now = 1310;
  stream.record({
    type: "move.played",
    turnId: "turn-1",
    ply: 1,
    san: "e4",
    color: "w",
    from: "e2",
    to: "e4",
    actor: "Model",
    elapsedMs: 310,
  });

  assert.deepEqual(stream.flush(), {
    batchIndex: 1,
    firstSeq: 1,
    lastSeq: 3,
    events: [
      {
        seq: 1,
        offsetMs: 0,
        payload: { type: "turn.started", turnId: "turn-1", ply: 1, color: "w" },
      },
      {
        seq: 2,
        offsetMs: 80,
        payload: {
          type: "thinking.command",
          turnId: "turn-1",
          command: { type: "highlightSquare", square: "e4", intensity: 1, fadeMs: 500 },
        },
      },
      {
        seq: 3,
        offsetMs: 310,
        payload: {
          type: "move.played",
          turnId: "turn-1",
          ply: 1,
          san: "e4",
          color: "w",
          from: "e2",
          to: "e4",
          actor: "Model",
          elapsedMs: 310,
        },
      },
    ],
  });
  assert.equal(stream.flush(), null);
});

test("published batches accept at most 128 thinking commands plus ordered game events", () => {
  const stream = createLiveGameEventSequencer(() => 0);
  stream.record({ type: "turn.started", turnId: "turn-1", ply: 1, color: "w" });
  for (let index = 0; index < 129; index += 1) {
    stream.record({ type: "thinking.command", turnId: "turn-1", command: { type: "clearAll" } });
  }
  const published = stream.flush();
  assert.equal(published.events.length, 129, "the publisher keeps one game event and 128 commands");
  assert.ok(normalizeLiveGameEventBatch(published), "the full ordered batch remains valid");

  const events = Array.from({ length: 129 }, (_, index) => ({
    seq: index + 1,
    offsetMs: index,
    payload: {
      type: "thinking.command",
      turnId: "turn-1",
      command: { type: "clearAll" },
    },
  }));
  assert.equal(normalizeLiveGameEventBatch({
    batchIndex: 1,
    firstSeq: 1,
    lastSeq: 129,
    events,
  }), null);
  assert.ok(normalizeLiveGameEventBatch({
    batchIndex: 1,
    firstSeq: 1,
    lastSeq: 128,
    events: events.slice(0, 128),
  }));
});
