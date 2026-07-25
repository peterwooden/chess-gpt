import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render(path = "/") {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request(`http://localhost${path}`, { headers: { accept: "text/html" } }),
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
  assert.match(html, /Open the browser arena/i);
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

test("server-renders the client-only browser arena", async () => {
  const response = await render("/arena");
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Browser Arena · Chess GPT Learning Lab<\/title>/i);
  assert.match(html, /Two URLs/i);
  assert.match(html, /Hugging Face model/i);
  assert.match(html, /Human vs A/i);
  assert.match(html, /A vs B/i);
  assert.match(html, /All inference happens on this device/i);
});

test("arena enforces a narrow, revision-aware model contract", async () => {
  const modelLoader = await readFile(new URL("../app/arena/model.ts", import.meta.url), "utf8");

  assert.match(modelLoader, /chess-gpt-browser-v1/);
  assert.match(modelLoader, /huggingface\.co/);
  assert.match(modelLoader, /sha256/i);
  assert.match(modelLoader, /legalMoves/);
  assert.match(modelLoader, /150 MB browser safety limit/);
  assert.doesNotMatch(modelLoader, /\beval\s*\(|new Function\s*\(/);
});
