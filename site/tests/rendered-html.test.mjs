import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the roadmap and placement diagnostic", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Chess GPT Learning Lab<\/title>/i);
  assert.match(html, /Predict before/i);
  assert.match(html, /Ten causal questions/i);
  assert.match(html, /Placement diagnostic/i);
  assert.match(html, /What does it mean to learn/i);
  assert.match(html, /How should we spend the budget/i);
  assert.match(html, /Reinforcement learning/i);
  assert.match(html, /How does a win teach earlier moves/i);
  assert.match(html, /Deep Learning/i);
  assert.match(html, /Spinning Up in Deep RL/i);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|react-loading-skeleton/i);
});

test("diagnostic remains gated and device-local", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(page, /CGPT-D0-/);
  assert.match(page, /answered === questions\.length/);
  assert.match(page, /confidenceSet === questions\.length/);
  assert.match(page, /window\.localStorage/);
  assert.match(page, /submitted \? diagnosticResult/);
  assert.match(page, /Chapter 1 stays locked/i);
});
