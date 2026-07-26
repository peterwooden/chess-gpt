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
  assert.match(html, /Continue Chapter 1/i);
  assert.match(html, /Mission 1/i);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|react-loading-skeleton/i);
});

test("diagnostic remains gated and device-local", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(page, /CGPT-D0-/);
  assert.match(page, /answered === questions\.length/);
  assert.match(page, /confidenceSet === questions\.length/);
  assert.match(page, /window\.localStorage/);
  assert.match(page, /submitted \? diagnosticResult/);
  assert.match(page, /teacher uses this code/i);
});

test("server-renders the first chapter mission", async () => {
  const response = await render("/chapter-1/data-splits");
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Split games, not positions · Chess GPT Learning Lab<\/title>/i);
  assert.match(html, /A score that lies politely/i);
  assert.match(html, /Predict the direction of the error/i);
  assert.match(html, /The exam answers are hiding in the study guide/i);
  assert.match(html, /Watch the leak inflate a score/i);
  assert.match(html, /Shuffle expanded positions/i);
  assert.match(html, /Hash stable game identity/i);
  assert.match(html, /Retrieval checkpoint/i);
  assert.match(html, /The code is still sealed/i);
  assert.match(html, /MLU-Explain/i);
});

test("chapter mission gates a device-local completion code", async () => {
  const lesson = await readFile(
    new URL("../app/chapter-1/data-splits/lesson-client.tsx", import.meta.url),
    "utf8",
  );

  assert.match(lesson, /CGPT-C1M1-GAMEHASH/);
  assert.match(lesson, /window\.localStorage/);
  assert.match(lesson, /forecastCorrect && implementationCorrect && progress\.retrievalChecked && retrievalCorrect/);
  assert.match(lesson, /progress\.implementation === 2/);
  assert.match(lesson, /what identity must remain stable/i);
  assert.match(lesson, /retrievalAttempts/);
  assert.match(lesson, /blob\/[0-9a-f]{40}\/src\/chess_gpt\/baseline\.py/);
});

test("the split simulator is deterministic and server-renderable", async () => {
  const simulator = await readFile(
    new URL("../app/chapter-1/data-splits/split-simulator.tsx", import.meta.url),
    "utf8",
  );

  assert.match(simulator, /mulberry32/);
  assert.match(simulator, /TRUE_SKILL = 0\.3/);
  assert.doesNotMatch(simulator, /Math\.random|Date\.now/);
});

test("server-renders the glossary with stable anchors", async () => {
  const response = await render("/glossary");
  assert.equal(response.status, 200);

  const html = await response.text();
  assert.match(html, /<title>Glossary · Chess GPT Learning Lab<\/title>/i);
  assert.match(html, /id="data-leakage"/);
  assert.match(html, /id="split-unit"/);
  assert.match(html, /id="validation-split"/);
  assert.match(html, /Generalization/);
});

test("server-renders the client-only browser arena", async () => {
  const response = await render("/arena");
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>ChessGPT Arena · Chess GPT Learning Lab<\/title>/i);
  assert.match(html, /ChessGPT arena/i);
  assert.match(html, /Set up game/i);
  assert.match(html, /Player 1/i);
  assert.match(html, /Player 2/i);
  assert.match(html, /Hugging Face model/i);
  assert.match(html, />White</i);
  assert.match(html, />Random</i);
  assert.match(html, />Black</i);
  assert.match(html, /Start game/i);
  assert.doesNotMatch(html, /Two URLs|Human vs A|A vs B/i);
});

test("arena defaults Player 1 to a random side and falls back to a human opponent", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");

  assert.match(arena, /useState<SidePreference>\("random"\)/);
  assert.match(arena, /modelA\.model && modelB\.model \? "models" : "human"/);
  assert.match(arena, /if \(!modelA\.model && !modelB\.model\)/);
  assert.match(arena, /const singleModel = modelA\.model \?\? modelB\.model/);
  assert.match(arena, /setHumanColor\(modelA\.model \? oppositeColor\(resolvedPlayer1Color\) : resolvedPlayer1Color\)/);
  assert.match(arena, /aria-pressed=\{sidePreference === side\.value\}/);
});

test("arena renders filled pieces, player strips, captures, and material advantage", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const font = await readFile(new URL("../public/fonts/chess-merida-unicode.ttf", import.meta.url));

  assert.match(arena, /w:\s*\{ p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" \}/);
  assert.match(arena, /className="promotion-piece"/);
  assert.match(arena, /captured:\s*move\.captured/);
  assert.match(arena, /CAPTURE_VALUES[^}]*p:\s*1[^}]*n:\s*3[^}]*b:\s*3[^}]*r:\s*5[^}]*q:\s*9/s);
  assert.match(arena, /const materialLead = whiteCapturePoints - blackCapturePoints/);
  assert.match(arena, /orientation === "w" \? blackPlayerSummary : whitePlayerSummary/);
  assert.match(arena, /player-strip \$\{color ===/);
  assert.match(arena, /className="captured-pieces"/);
  assert.match(styles, /@font-face\s*\{[^}]*font-family:\s*"Chess Merida"[^}]*chess-merida-unicode\.ttf/s);
  assert.match(styles, /\.piece\s*\{[^}]*font-family:\s*"Chess Merida"/s);
  assert.match(styles, /\.captured-piece\s*\{[^}]*font-family:\s*"Chess Merida"/s);
  assert.ok(font.byteLength > 0);
});

test("arena consolidates match status in the move pane", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");

  assert.doesNotMatch(arena, /className="game-status"/);
  assert.match(arena, /<span>\{mode === "human" \? "Human match" : "Model match"\}<\/span>/);
  assert.match(arena, /<strong aria-live="polite">\{status\}<\/strong>/);
  assert.match(arena, /className="move-header-meta"/);
  assert.match(arena, /\{history\.length\} plies/);
});

test("arena enforces a narrow, revision-aware model contract", async () => {
  const modelLoader = await readFile(new URL("../app/arena/model.ts", import.meta.url), "utf8");
  const modelWorker = await readFile(new URL("../app/arena/model-worker.ts", import.meta.url), "utf8");

  assert.match(modelLoader, /chess-gpt-package-v1/);
  assert.match(modelLoader, /huggingface\.co/);
  assert.match(modelLoader, /sha256/i);
  assert.match(modelLoader, /100_000_000/);
  assert.match(modelLoader, /legalMoves/);
  assert.match(modelLoader, /new Worker/);
  assert.match(modelLoader, /es-module-lexer/);
  assert.match(modelWorker, /loadPackage/);
  assert.match(modelWorker, /newGame/);
  assert.match(modelWorker, /chooseMove/);
  assert.match(modelWorker, /URL\.createObjectURL/);
  assert.doesNotMatch(`${modelLoader}\n${modelWorker}`, /\beval\s*\(|new Function\s*\(/);
  assert.doesNotMatch(modelLoader, /chess-gpt-browser-v1|model\.json\.gz/);
  assert.match(modelWorker, /globalThis\.fetch = \(\) => Promise\.reject/);
});

test("the SAN n-gram adapter implements package v1 without arena-specific imports", async () => {
  const source = await readFile(
    new URL("../../adapters/san-ngram/entry.js", import.meta.url),
    "utf8",
  );
  const moduleUrl = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;
  const adapter = await import(moduleUrl);
  const state = {
    format_version: 1,
    model_type: "san_backoff_ngram",
    order: 1,
    metadata: { experiment_id: "test-ngram" },
    ngrams: { "1": { e4: [["e5", 2]] } },
    side_counts: { "0": [["e4", 3]], "1": [["e5", 2]] },
  };
  const artifacts = new Map([
    ["model", new TextEncoder().encode(JSON.stringify(state))],
  ]);

  const loaded = await adapter.loadPackage({ artifacts, config: {}, ort: {} });
  const game = await loaded.newGame({ random: () => 0.5 });

  assert.equal(await game.chooseMove({ history: [], legalMoves: ["d4", "e4"] }), "e4");
  assert.equal(await game.chooseMove({ history: ["e4"], legalMoves: ["c5", "e5"] }), "e5");
  await game.dispose();
  await loaded.dispose();
  assert.doesNotMatch(source, /from\s+["']|import\s*\(/);
});

test("arena constrains the board to eight equal columns and rows", async () => {
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const boardRule = styles.match(/\.chessboard\s*\{[^}]+\}/)?.[0] ?? "";

  assert.match(boardRule, /grid-template-columns:\s*repeat\(8,\s*minmax\(0,\s*1fr\)\)/);
  assert.match(boardRule, /grid-template-rows:\s*repeat\(8,\s*minmax\(0,\s*1fr\)\)/);
});

test("arena has a single-viewport laptop workspace", async () => {
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const laptopStyles = styles.split("/* Arena v2: board with a contextual setup / move pane */")[1] ?? "";
  const pageRule = laptopStyles.match(/\.arena-page-v2\s*\{[^}]*\}/g)?.at(-1) ?? "";
  const workspaceRule = laptopStyles.match(/\.arena-workspace\s*\{[^}]*\}/g)?.at(-1) ?? "";
  const boardRule = laptopStyles.match(/\.arena-page-v2 \.chessboard\s*\{[^}]*\}/g)?.at(-1) ?? "";
  const playerBoardRule = laptopStyles.match(/\.board-frame\.with-players\s*\{[^}]*\}/g)?.at(-1) ?? "";

  assert.match(laptopStyles, /@media \(min-width:\s*960px\) and \(min-height:\s*640px\)/);
  assert.match(pageRule, /height:\s*100svh/);
  assert.match(pageRule, /overflow:\s*hidden/);
  assert.match(workspaceRule, /grid-template-columns:\s*minmax\(0,\s*1fr\) minmax\(20rem,\s*24rem\)/);
  assert.match(workspaceRule, /min-height:\s*0/);
  assert.match(playerBoardRule, /width:\s*min\(100%,\s*calc\(100svh - 11\.75rem\)\)/);
  assert.match(boardRule, /height:\s*auto/);
});
