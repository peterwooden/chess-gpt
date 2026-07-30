import assert from "node:assert/strict";
import test from "node:test";
import { parsePackageManifest } from "../app/arena/package-manifest.mjs";

const digest = "a".repeat(64);

function manifest(overrides = {}) {
  return new TextEncoder().encode(JSON.stringify({
    schema: "chess-gpt-package-v1",
    name: "Exact model",
    entrypoint: { path: "entry.js", bytes: 12, sha256: digest },
    artifacts: { model: { path: "model.onnx", bytes: 34, sha256: digest } },
    config: {},
    ...overrides,
  }));
}

test("the shared package contract accepts a complete compatible manifest", () => {
  const parsed = parsePackageManifest(manifest());
  assert.equal(parsed.manifest.name, "Exact model");
  assert.equal(parsed.manifest.entrypoint.path, "entry.js");
  assert.equal(parsed.packageBytes, manifest().byteLength + 46);
});

test("catalog registration rejects manifests the arena cannot load", () => {
  assert.throws(() => parsePackageManifest(manifest({ artifacts: undefined })), /artifacts and config/);
  assert.throws(() => parsePackageManifest(manifest({
    entrypoint: { path: "../entry.js", bytes: 12, sha256: digest },
  })), /stay beneath browser/);
  assert.throws(() => parsePackageManifest(manifest({
    entrypoint: { path: "entry.js", bytes: 100_000_000, sha256: digest },
  })), /100 MB limit/);
});
