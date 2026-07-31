import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { Chess } from "chess.js";
import * as ort from "onnxruntime-web";

const dir = resolve(process.argv[2]);
const manifest = JSON.parse(await readFile(resolve(dir, "manifest.json"), "utf8"));
const artifacts = new Map();
for (const [name, descriptor] of Object.entries(manifest.artifacts)) {
  artifacts.set(name, new Uint8Array(await readFile(resolve(dir, descriptor.path))));
}
const entry = await import(pathToFileURL(resolve(dir, manifest.entrypoint.path)).href);
const pkg = await entry.loadPackage({ artifacts, ort });

const probes = [
  { name: "mate-in-1 (Scholar's)", history: ["e4", "e5", "Bc4", "Nc6", "Qh5", "Nf6"], expect: ["Qxf7#"] },
  { name: "win rook w/ check", history: ["e4", "e5", "Qh5", "g6"], expect: ["Qxe5+"] },
  { name: "recapture the queen", history: ["e4", "d5", "exd5", "Qxd5", "Nc3", "Qe5+", "Qe2", "Qxe2+"], expect: ["Bxe2", "Ngxe2", "Kxe2", "Nxe2", "Ncxe2"] },
];

for (const { name, history, expect } of probes) {
  const chess = new Chess();
  for (const san of history) chess.move(san);
  const game = await pkg.newGame();
  const started = Date.now();
  const chosen = await game.chooseMove({
    history,
    legalMoves: chess.moves(),
    moveTimeLimitMs: 4000,
  });
  const verdict = expect.includes(chosen) ? "PASS" : "FAIL";
  console.log(`${verdict}  ${name}: chose ${chosen} (expected ${expect.join("/")}) in ${Date.now() - started}ms`);
  await game.dispose();
}
// Budget check: a tight clock must still return quickly and legally.
const chess = new Chess();
for (const san of ["d4", "d5", "c4"]) chess.move(san);
const game = await pkg.newGame();
const started = Date.now();
const fast = await game.chooseMove({ history: ["d4", "d5", "c4"], legalMoves: chess.moves(), moveTimeLimitMs: 300 });
console.log(`BUDGET  300ms limit: chose ${fast} in ${Date.now() - started}ms (must be well under 375ms hard kill)`);
await game.dispose();
await pkg.dispose();
