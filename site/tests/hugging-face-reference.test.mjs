import assert from "node:assert/strict";
import test from "node:test";
import {
  modelPageHref,
  parseHuggingFaceReference,
  resolveHuggingFaceReference,
} from "../app/arena/hugging-face-reference.mjs";

test("parses repository shorthand and Hugging Face URLs", () => {
  assert.deepEqual(parseHuggingFaceReference("owner/model"), {
    repository: "owner/model",
    revision: "main",
  });
  assert.deepEqual(parseHuggingFaceReference("https://huggingface.co/owner/model/tree/v2"), {
    repository: "owner/model",
    revision: "v2",
  });
});

test("resolves a movable revision to a canonical commit reference", async () => {
  const calls = [];
  const resolved = await resolveHuggingFaceReference("owner/model@main", async (url) => {
    calls.push(url);
    return new Response(JSON.stringify({
      id: "owner/model",
      sha: "1234567890abcdef1234567890abcdef12345678",
    }));
  });
  assert.match(calls[0], /revision\/main/);
  assert.equal(resolved.reference, "owner/model@1234567890abcdef1234567890abcdef12345678");
  assert.equal(
    resolved.manifestUrl,
    "https://huggingface.co/owner/model/resolve/1234567890abcdef1234567890abcdef12345678/browser/manifest.json",
  );
});

test("rejects an invalid revision response", async () => {
  await assert.rejects(
    resolveHuggingFaceReference("owner/model", async () => new Response(JSON.stringify({
      id: "owner/model",
      sha: "main",
    }))),
    /invalid model revision/,
  );
});

test("builds a canonical model page link for an immutable reference", () => {
  assert.equal(
    modelPageHref("alice/quiet-model@1234567890abcdef1234567890abcdef12345678"),
    "/models/alice/quiet-model?version=1234567890abcdef1234567890abcdef12345678",
  );
});
