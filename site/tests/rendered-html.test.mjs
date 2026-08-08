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

test("arena casual default move time limit is 10 seconds", async () => {
  const model = await readFile(new URL("../app/arena/model.ts", import.meta.url), "utf8");
  assert.match(model, /export const DEFAULT_MOVE_TIME_LIMIT_MS = 10_000;/);
});

test("arena setup exposes a per-player thinking cap defaulting to 10000 ms", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");

  assert.match(arena, /DEFAULT_MOVE_TIME_LIMIT_MS/);
  assert.match(arena, /useState\(DEFAULT_MOVE_TIME_LIMIT_MS\)/);
  assert.match(arena, /Thinking cap \(ms\)/);
  assert.match(arena, /Per-move budget passed to the package/);
  assert.match(arena, /moveTimeLimitMs=\{moveTimeLimitMsA\}/);
  assert.match(arena, /moveTimeLimitMs=\{moveTimeLimitMsB\}/);
  assert.match(arena, /model\.predict\([\s\S]*?activeGame\.history\(\),[\s\S]*?activeGame\.moves\(\),[\s\S]*?moveTimeLimitMs,/);
  assert.doesNotMatch(arena, /chess-gpt:arena-move-time/);
  assert.doesNotMatch(arena, /searchParams\.(get|set)\(["']t(?:ime)?/);
});

test("arena renders filled pieces, player strips, captures, and material advantage", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");
  const playerStrip = await readFile(new URL("../app/arena/player-strip.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const font = await readFile(new URL("../public/fonts/chess-merida-unicode.ttf", import.meta.url));

  assert.match(arena, /w:\s*\{ p: "♟", n: "♞", b: "♝", r: "♜", q: "♛", k: "♚" \}/);
  assert.match(arena, /className="promotion-piece"/);
  assert.match(arena, /captured:\s*move\.captured/);
  assert.match(playerStrip, /CAPTURE_VALUES[^}]*p:\s*1[^}]*n:\s*3[^}]*b:\s*3[^}]*r:\s*5[^}]*q:\s*9/s);
  assert.match(playerStrip, /const lead = Math\.max\(0, ownPoints - opponentPoints\)/);
  assert.match(arena, /orientation === "w" \? blackPlayerSummary : whitePlayerSummary/);
  assert.match(playerStrip, /player-strip \$\{color ===/);
  assert.match(playerStrip, /className="captured-pieces"/);
  assert.match(styles, /@font-face\s*\{[^}]*font-family:\s*"Chess Merida"[^}]*chess-merida-unicode\.ttf/s);
  assert.match(styles, /\.piece\s*\{[^}]*font-family:\s*"Chess Merida"/s);
  assert.match(styles, /\.captured-piece\s*\{[^}]*font-family:\s*"Chess Merida"/s);
  assert.ok(font.byteLength > 0);
});

test("arena consolidates match status in the move pane", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");
  const progressPanel = await readFile(new URL("../app/arena/game-progress-panel.tsx", import.meta.url), "utf8");

  assert.doesNotMatch(arena, /className="game-status"/);
  assert.match(arena, /mode === "human" \? "Human match" : "Model match"/);
  assert.match(arena, /<GameProgressPanel/);
  assert.match(progressPanel, /<strong aria-live="polite">\{status\}<\/strong>/);
  assert.match(progressPanel, /className="move-header-meta"/);
  assert.match(progressPanel, /\{moves\.length\} plies/);
  assert.match(progressPanel, /timeline\.behind/);
});

test("arena uses a compact score sheet and shares completed games as PGN", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");
  const progressPanel = await readFile(new URL("../app/arena/game-progress-panel.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(progressPanel, /className="move-score-heading"/);
  assert.match(progressPanel, /<span>Move<\/span><b>White<\/b><b>Black<\/b>/);
  assert.match(progressPanel, /rows\.map/);
  assert.match(progressPanel, /record\.scrollTop = record\.scrollHeight/);
  assert.match(arena, /className="share-toggle"/);
  assert.match(arena, /Models \+ game/);
  assert.match(arena, /readSharedPgn/);
  assert.match(arena, /restorePgn\(sharedPgn\)/);
  assert.match(arena, /PlayerOneColor/);
  assert.doesNotMatch(arena, /setHeader\("Player1Color"/);
  assert.match(styles, /grid-template-columns:\s*2\.5rem minmax\(0, 1fr\) minmax\(0, 1fr\)/);
  assert.match(arena, /aria-expanded=\{shareOpen\}/);
  assert.match(styles, /\.share-options\s*\{[^}]*position:\s*static/s);
  assert.doesNotMatch(styles, /\.share-menu > div\s*\{[^}]*position:\s*absolute/s);
});

test("arena can review every played position without leaving the live game", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");
  const progressPanel = await readFile(new URL("../app/arena/game-progress-panel.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(progressPanel, /const \[viewedPly, setViewedPly\] = useState<number \| null>\(null\)/);
  assert.match(arena, /const displayedGame = isLiveView \? game : gameAtPly\(moves, displayPly\)/);
  assert.match(progressPanel, /aria-label="Move history navigation"/);
  assert.match(progressPanel, /aria-label="Go to starting position"/);
  assert.match(progressPanel, /timeline\.playing \? "Pause move history" : "Play move history to live"/);
  assert.match(progressPanel, /timeline\.playing \? "Ⅱ" : "▶"/);
  assert.match(progressPanel, /window\.setTimeout/);
  assert.doesNotMatch(arena, /\{gameEnded \? "Final" : "Live"\}/);
  assert.match(progressPanel, /data-ply=\{move\.ply\}/);
  assert.match(progressPanel, /onClick=\{\(\) => timeline\.showPosition\(move\.ply\)\}/);
  assert.match(progressPanel, /record\.querySelector<HTMLElement>/);
  assert.match(styles, /\.history-navigation\s*\{[^}]*border-bottom:\s*1px solid var\(--green\)/s);
  assert.match(styles, /\.move-record\s*\{[^}]*position:\s*relative/s);
  assert.match(styles, /@media \(max-width:\s*959px\)[\s\S]*\.arena-sidebar\s*\{[^}]*flex:\s*1/s);
});

test("completed games receive a compact Stockfish accuracy review", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");
  const review = await readFile(new URL("../app/arena/stockfish-review.mjs", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(arena, /analyzeGameWithStockfish/);
  assert.match(arena, /<b>\{Math\.round\(review\.accuracy\)\}%<\/b> accuracy/);
  assert.match(arena, /\["brilliant", "good"\]/);
  assert.doesNotMatch(arena, /interesting/i);
  assert.match(arena, /\["inaccuracy", "mistake", "blunder"\]/);
  assert.match(arena, /<ReviewGroup label="Good"/);
  assert.match(arena, /<ReviewGroup label="Errors"/);
  assert.match(arena, /players=\{players\}/);
  assert.match(arena, /<b title=\{name\}>\{name\}<\/b>/);
  assert.match(arena, /review\.counts\[judgement\]/);
  assert.match(review, /STOCKFISH_NODES_PER_POSITION = 30_000/);
  assert.match(review, /loss >= 30/);
  assert.match(review, /loss >= 20/);
  assert.match(review, /loss >= 10/);
  assert.match(styles, /\.review-pill\.blunder\.active\s*\{[^}]*background:/s);
  assert.match(styles, /\.review-categories\s*\{[^}]*grid-template-columns:\s*1fr 1fr/s);
  assert.match(styles, /\.review-group\s*\{[^}]*align-content:\s*start/s);
  assert.match(styles, /\.review-stat\s*\{[^}]*grid-template-columns:\s*auto minmax\(0, 1fr\) auto/s);
  assert.match(styles, /\.review-loss\s*\{[^}]*margin-left:\s*auto/s);
  assert.doesNotMatch(styles, /\.score-move \.review-pill\s*\{[^}]*margin-left:\s*auto/s);
  assert.match(styles, /\.review-pill\.inactive\s*\{[^}]*background:/s);
});

test("arena enforces a narrow, revision-aware model contract", async () => {
  const modelLoader = await readFile(new URL("../app/arena/model.ts", import.meta.url), "utf8");
  const packageManifest = await readFile(new URL("../app/arena/package-manifest.mjs", import.meta.url), "utf8");
  const modelWorker = await readFile(new URL("../app/arena/model-worker.ts", import.meta.url), "utf8");

  assert.match(packageManifest, /chess-gpt-package-v1/);
  assert.match(modelLoader, /huggingface\.co/);
  assert.match(modelLoader, /sha256/i);
  assert.match(packageManifest, /100_000_000/);
  assert.match(modelLoader, /legalMoves/);
  assert.match(modelLoader, /new Worker\(new URL\("\.\/model-worker\.ts"/);
  assert.match(modelLoader, /es-module-lexer/);
  assert.match(modelWorker, /loadPackage/);
  assert.match(modelWorker, /newGame/);
  assert.match(modelWorker, /chooseMove/);
  assert.match(modelWorker, /URL\.createObjectURL/);
  assert.doesNotMatch(`${modelLoader}\n${modelWorker}`, /\beval\s*\(|new Function\s*\(/);
  assert.doesNotMatch(modelLoader, /chess-gpt-browser-v1|model\.json\.gz/);
  assert.match(modelWorker, /globalThis\.fetch = \(\) => Promise\.reject/);
});

test("Sites leaves the arena non-isolated so its static model worker can start", async () => {
  const nextConfig = await readFile(new URL("../next.config.ts", import.meta.url), "utf8");
  const viteConfig = await readFile(new URL("../vite.config.ts", import.meta.url), "utf8");
  const staticHeaders = await readFile(new URL("../public/_headers", import.meta.url), "utf8");

  assert.doesNotMatch(`${nextConfig}\n${viteConfig}\n${staticHeaders}`, /Cross-Origin-Embedder-Policy|crossOriginIsolation/);
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

test("pieces scale from their board instead of the viewport", async () => {
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const boardRule = styles.match(/\.chessboard\s*\{[^}]+\}/)?.[0] ?? "";

  assert.match(boardRule, /container-type:\s*inline-size/);
  assert.match(styles, /\.piece\s*\{[^}]*font-size:\s*10cqi/s);
  assert.doesNotMatch(styles, /\.piece\s*\{[^}]*font-size:[^;}]*(?:vw|vh)/s);
  assert.doesNotMatch(styles, /\.(?:arena-page-v2|tournament-live-board)[^{]*\.piece\s*\{[^}]*font-size:/s);
});

test("arena uses standard chess square colours", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");

  // A1 is dark, so an even file-index-plus-rank sum must select the dark class.
  assert.match(arena, /const light = \(FILES\.indexOf\(file\) \+ rank\) % 2 === 0;/);
});

test("signed-in users can create tournaments while anonymous users cannot", async () => {
  const tournaments = await readFile(new URL("../lib/tournaments.ts", import.meta.url), "utf8");
  const page = await readFile(new URL("../app/tournaments/page.tsx", import.meta.url), "utf8");

  assert.match(tournaments, /const creator = requireSignedInUser\(user\);/);
  assert.match(tournaments, /const owner = await ensureHumanPlayer\(creator\);/);
  assert.match(page, /\{user \? <CreateTournamentForm\s*\/\> : null\}/);
  assert.match(page, /Sign in to create one\./);
});

test("tournament directory keeps names and metadata readable on narrow screens", async () => {
  const page = await readFile(new URL("../app/tournaments/page.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(page, /className="row-arrow"/);
  assert.match(
    styles,
    /@media \(max-width: 700px\) \{[\s\S]*?\.history-directory-row\.tournament-row\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\) auto/s,
  );
  assert.match(
    styles,
    /@media \(max-width: 700px\) \{[\s\S]*?\.tournament-row \.tournament-status\s*\{[^}]*grid-column:\s*1/s,
  );
});

test("tournament game directory keeps game details readable on narrow screens", async () => {
  const page = await readFile(new URL("../app/tournaments/[id]/page.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(page, /className="tournament-game-result"/);
  assert.match(page, /className="row-arrow" aria-hidden="true"/);
  assert.match(
    styles,
    /@media \(max-width: 700px\) \{[\s\S]*?\.history-directory-row\.tournament-game-row\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\) auto/s,
  );
  assert.match(
    styles,
    /@media \(max-width: 700px\) \{[\s\S]*?\.tournament-game-row \.tournament-game-result\s*\{[^}]*grid-column:\s*1/s,
  );
});

test("tournament creators can manage their own tournaments", async () => {
  const tournaments = await readFile(new URL("../lib/tournaments.ts", import.meta.url), "utf8");
  const page = await readFile(new URL("../app/tournaments/[id]/page.tsx", import.meta.url), "utf8");

  assert.match(tournaments, /export async function isTournamentManager\(/);
  assert.match(tournaments, /createdByPlayerId/);
  assert.match(tournaments, /await requireTournamentManager\(tournament, user\);/);
  assert.match(page, /isTournamentManager\(tournament, user\)/);
  assert.match(page, /\{manager \? \(/);
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

test("thinking annotations align to the chessboard content inside its border", async () => {
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");

  assert.match(styles, /\.thinking-board-wrap\s*\{[^}]*--thinking-board-inset:\s*0\.45rem/s);
  assert.match(styles, /\.arena-page-v2 \.thinking-board-wrap\s*\{[^}]*--thinking-board-inset:\s*0\.4rem/s);
  assert.match(styles, /\.thinking-overlay\s*\{[^}]*inset:\s*var\(--thinking-board-inset\)[^}]*width:\s*calc\(100% - var\(--thinking-board-border-total\)\)/s);
});

test("mobile setup uses the page-sized workspace as its scroll container", async () => {
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const mobileStyles = styles.match(/@media \(max-width:\s*959px\)\s*\{[\s\S]*?\n\}/)?.[0] ?? "";

  assert.match(arena, /gameStarted \? "game-mode" : "setup-mode"/);
  assert.match(mobileStyles, /\.arena-workspace\.setup-mode\s*\{[^}]*overflow-y:\s*auto/s);
  assert.match(mobileStyles, /\.arena-workspace\.setup-mode \.arena-sidebar\s*\{[^}]*flex:\s*none/s);
  assert.match(mobileStyles, /-webkit-overflow-scrolling:\s*touch/);
  assert.match(mobileStyles, /env\(safe-area-inset-bottom\)/);
});

test("first-class model pages expose exact versions and compact replay references", async () => {
  const catalog = await readFile(new URL("../app/models/page.tsx", import.meta.url), "utf8");
  const modelPage = await readFile(new URL("../app/models/[owner]/[repository]/page.tsx", import.meta.url), "utf8");
  const actions = await readFile(new URL("../app/models/reference-actions.tsx", import.meta.url), "utf8");
  const arena = await readFile(new URL("../app/arena/arena-client.tsx", import.meta.url), "utf8");
  const playerStrip = await readFile(new URL("../app/arena/player-strip.tsx", import.meta.url), "utf8");
  const styles = await readFile(new URL("../app/globals.css", import.meta.url), "utf8");
  const playerPage = await readFile(new URL("../app/players/[id]/page.tsx", import.meta.url), "utf8");

  assert.match(catalog, /Latest · \{model\.repository\}@\{model\.latestCommitSha\}/);
  assert.match(modelPage, /First seen by ChessGPT/);
  assert.match(modelPage, /selectModelVersion\(versions, query\.version\)/);
  assert.match(actions, /navigator\.clipboard\.writeText\(reference\)/);
  assert.match(actions, /modelChallengeHref\(reference\)/);
  assert.doesNotMatch(actions, /import Link from "next\/link"/);
  assert.match(playerStrip, /model-name-with-copy/);
  assert.match(playerStrip, /navigator\.clipboard\.writeText\(reference\)/);
  assert.match(playerStrip, /aria-label=\{`Copy full reference for \$\{name\}`\}/);
  assert.match(playerStrip, /window\.matchMedia\("\(hover: none\)"\)\.matches/);
  assert.match(playerStrip, /event\.preventDefault\(\)/);
  assert.doesNotMatch(arena, /className="replay-model-actions"/);
  assert.match(styles, /\.model-name-with-copy\s*\{[^}]*display:\s*flex[^}]*min-width:\s*0/s);
  assert.match(styles, /\.model-name-copy\s*\{[^}]*flex:\s*none/s);
  assert.match(playerPage, /redirect\(modelPageHref/);
});
