import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve, sep } from "node:path";
import { Chess } from "chess.js";
import * as ort from "onnxruntime-web";

const PACKAGE_LIMIT_BYTES = 100_000_000;
const packageDirectory = resolve(process.argv[2] ?? "");
if (!process.argv[2]) {
  throw new Error("usage: node site/scripts/validate-snapshot-package.mjs PACKAGE_DIRECTORY");
}

function confinedPath(relativePath) {
  if (
    typeof relativePath !== "string" ||
    relativePath.length === 0 ||
    relativePath.startsWith("/") ||
    relativePath.split(/[\\/]/u).includes("..")
  ) {
    throw new Error(`invalid package path: ${relativePath}`);
  }
  const absolutePath = resolve(packageDirectory, relativePath);
  if (!absolutePath.startsWith(`${packageDirectory}${sep}`)) {
    throw new Error(`package path escapes browser directory: ${relativePath}`);
  }
  return absolutePath;
}

async function verifiedPayload(descriptor) {
  const payload = await readFile(confinedPath(descriptor.path));
  const digest = createHash("sha256").update(payload).digest("hex");
  if (payload.byteLength !== descriptor.bytes) {
    throw new Error(`${descriptor.path} has ${payload.byteLength} bytes, expected ${descriptor.bytes}`);
  }
  if (digest !== descriptor.sha256) {
    throw new Error(`${descriptor.path} has SHA-256 ${digest}, expected ${descriptor.sha256}`);
  }
  return new Uint8Array(payload);
}

const manifestBytes = await readFile(resolve(packageDirectory, "manifest.json"));
const manifest = JSON.parse(manifestBytes);
if (manifest.schema !== "chess-gpt-package-v1") {
  throw new Error(`unsupported package schema: ${manifest.schema}`);
}
const descriptors = [manifest.entrypoint, ...Object.values(manifest.artifacts)];
const paths = descriptors.map((descriptor) => descriptor.path);
if (new Set(paths).size !== paths.length) {
  throw new Error("manifest paths must be unique");
}
const payloads = await Promise.all(descriptors.map(verifiedPayload));
const packageBytes = manifestBytes.byteLength + payloads.reduce(
  (total, payload) => total + payload.byteLength,
  0,
);
if (packageBytes > PACKAGE_LIMIT_BYTES) {
  throw new Error(`canonical package has ${packageBytes} bytes, over ${PACKAGE_LIMIT_BYTES}`);
}

const artifacts = new Map();
for (const [name, descriptor] of Object.entries(manifest.artifacts)) {
  artifacts.set(name, await verifiedPayload(descriptor));
}
ort.env.wasm.wasmPaths = new URL("../node_modules/onnxruntime-web/dist/", import.meta.url).href;
const entrypointUrl = `data:text/javascript;base64,${Buffer.from(payloads[0]).toString("base64")}`;
const { loadPackage } = await import(entrypointUrl);
if (typeof loadPackage !== "function") {
  throw new Error("entrypoint does not export loadPackage");
}

const loadedPackage = await loadPackage({ artifacts, config: manifest.config, ort });
let plies = 0;
try {
  const game = await loadedPackage.newGame({ random: () => 0.5 });
  const chess = new Chess();
  const history = [];
  try {
    while (!chess.isGameOver() && plies < 40) {
      const legalMoves = chess.moves();
      const move = await game.chooseMove({ history: [...history], legalMoves });
      if (!legalMoves.includes(move)) {
        throw new Error(`entrypoint returned illegal SAN at ply ${plies + 1}: ${move}`);
      }
      chess.move(move);
      history.push(move);
      plies += 1;
    }
  } finally {
    await game.dispose();
  }
} finally {
  await loadedPackage.dispose();
}

console.log(JSON.stringify({
  architecture: manifest.config?.architecture,
  canonical_package_bytes: packageBytes,
  name: manifest.name,
  self_play_plies: plies,
}, null, 2));
