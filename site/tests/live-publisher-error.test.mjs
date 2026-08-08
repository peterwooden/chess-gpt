import assert from "node:assert/strict";
import test from "node:test";

import { livePublisherErrorMessage } from "../lib/live-publisher-error.mjs";

test("live-link aborts never expose the browser's raw signal message", () => {
  const fallback = "The live link could not be created. Please try again.";

  assert.equal(
    livePublisherErrorMessage(
      new DOMException("signal is aborted without reason", "AbortError"),
      fallback,
    ),
    fallback,
  );
  assert.equal(livePublisherErrorMessage(new Error("Database unavailable."), fallback), "Database unavailable.");
});
