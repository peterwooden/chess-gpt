import assert from "node:assert/strict";
import test from "node:test";

import {
  readSharedModelReferences,
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
