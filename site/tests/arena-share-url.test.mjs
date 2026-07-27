import assert from "node:assert/strict";
import test from "node:test";

import {
  buildArenaShareUrl,
  readSharedModelReferences,
  readSharedPgn,
  withSharedModelReference,
} from "../app/arena/share-url.mjs";

test("a shared arena URL defines both model fields without borrowing device defaults", () => {
  assert.deepEqual(
    readSharedModelReferences(
      "https://example.test/arena?modelA=alice%2Fwhite%40abc123",
    ),
    { a: "alice/white@abc123", b: "" },
  );
  assert.equal(readSharedModelReferences("https://example.test/arena"), null);
});

test("a successfully loaded model can be added to a stable share URL", () => {
  const withA = withSharedModelReference(
    "https://example.test/arena?lesson=1#board",
    "a",
    " alice/white@abc123 ",
  );
  assert.equal(
    withA,
    "https://example.test/arena?lesson=1&modelA=alice%2Fwhite%40abc123#board",
  );

  assert.equal(
    withSharedModelReference(withA, "b", "bob/black@def456"),
    "https://example.test/arena?lesson=1&modelA=alice%2Fwhite%40abc123&modelB=bob%2Fblack%40def456#board",
  );
});

test("a completed game share round-trips PGN and both model references", () => {
  const pgn = `[Event "ChessGPT Arena"]\n[White "Alice"]\n[Black "Bob"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0`;
  const shared = buildArenaShareUrl("https://example.test/arena?lesson=1#board", {
    a: "alice/white@abc123",
    b: "bob/black@def456",
    pgn,
  });

  assert.deepEqual(readSharedModelReferences(shared), {
    a: "alice/white@abc123",
    b: "bob/black@def456",
  });
  assert.equal(readSharedPgn(shared), pgn);
  assert.equal(new URL(shared).hash, "#board");
});

test("a models-only share removes stale PGN and empty model slots", () => {
  const shared = buildArenaShareUrl(
    "https://example.test/arena?modelA=old&modelB=old&pgn=stale",
    { a: "alice/model@abc123", b: "" },
  );

  assert.equal(
    shared,
    "https://example.test/arena?modelA=alice%2Fmodel%40abc123",
  );
  assert.equal(readSharedPgn(shared), null);
});
