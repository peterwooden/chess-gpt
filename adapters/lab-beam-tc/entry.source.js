import { Chess } from "chess.js";

const FILES = "abcdefgh";
const HISTORY = 8;
const ROOT_BEAM = 6;
const BEAM = 5;
const QUIESCENCE_BEAM = 3;
const QUIESCENCE_MAX_PLIES = 4;
const MAX_MAIN_DEPTH = 6;
const FRONTIER_CAP = 8000;
const CONTEMPT = 0.15;
const WINNING_THRESHOLD = 0.55;
const BUDGET_FRACTION = 0.7; // stay clear of the runner's 1.25x hard kill
const DEFAULT_LIMIT_MS = 3000;
const PIECE_CODES = {
  wp: 1, wn: 2, wb: 3, wr: 4, wq: 5, wk: 6,
  bp: 7, bn: 8, bb: 9, br: 10, bq: 11, bk: 12,
};

function squareIndex(square) {
  return FILES.indexOf(square[0]) + (Number(square[1]) - 1) * 8;
}

function encodePosition(chess) {
  const squares = new Int32Array(64);
  for (let rank = 1; rank <= 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const piece = chess.get(`${FILES[file]}${rank}`);
      squares[(rank - 1) * 8 + file] = piece ? PIECE_CODES[`${piece.color}${piece.type}`] : 0;
    }
  }
  const [, turn, castling, ep, halfmove] = chess.fen().split(" ");
  const state = Int32Array.of(
    Number(turn === "b"),
    Number(castling.includes("K")),
    Number(castling.includes("Q")),
    Number(castling.includes("k")),
    Number(castling.includes("q")),
    ep === "-" ? 64 : squareIndex(ep),
    Math.min(Number(halfmove), 100),
  );
  return { squares, state };
}

function historyArrays(last) {
  const from = new Int32Array(HISTORY).fill(64);
  const to = new Int32Array(HISTORY).fill(64);
  const pad = HISTORY - last.length;
  last.forEach((move, index) => {
    from[pad + index] = move.f;
    to[pad + index] = move.t;
  });
  return { from, to };
}

function positionKey(chess) {
  return chess.fen().split(" ").slice(0, 4).join(" ");
}

function moveIndex(move, promotionMoves) {
  const from = squareIndex(move.from);
  const to = squareIndex(move.to);
  if (!move.promotion) return from * 64 + to;
  const index = promotionMoves.indexOf(`${move.from}${move.to}${move.promotion}`);
  if (index < 0) throw new Error(`Unknown promotion move: ${move.from}${move.to}${move.promotion}`);
  return 4096 + index;
}

function terminalWhiteScore(chess) {
  if (chess.isCheckmate()) return chess.turn() === "w" ? 0.0 : 1.0;
  if (chess.isDraw() || chess.isStalemate()) return 0.5;
  return null;
}

function whiteScore(values, base) {
  const v0 = Number(values[base]);
  const v1 = Number(values[base + 1]);
  const v2 = Number(values[base + 2]);
  const peak = Math.max(v0, v1, v2);
  const e0 = Math.exp(v0 - peak);
  const e1 = Math.exp(v1 - peak);
  const e2 = Math.exp(v2 - peak);
  const total = e0 + e1 + e2;
  return e0 / total + 0.5 * (e1 / total);
}

function appended(last, move) {
  const next = last.concat([{ f: squareIndex(move.from), t: squareIndex(move.to) }]);
  return next.length > HISTORY ? next.slice(-HISTORY) : next;
}

function makeChild(node, move) {
  const chess = new Chess(node.chess.fen());
  chess.move(move.san);
  const child = {
    chess,
    last: appended(node.last, move),
    score: terminalWhiteScore(chess),
    standPat: null,
    noisy: Boolean(move.captured || move.promotion) || chess.inCheck(),
    children: [],
  };
  node.children.push(child);
  return child;
}

function backup(node) {
  if (node.score !== null) return node.score;
  const whiteToMove = node.chess.turn() === "w";
  // Stand-pat: in quiescence, declining every capture is always an option.
  let best = node.standPat;
  for (const child of node.children) {
    const score = backup(child);
    if (best === null || (whiteToMove ? score > best : score < best)) best = score;
  }
  node.score = best === null ? 0.5 : best;
  return node.score;
}

export async function loadPackage({ artifacts, ort }) {
  const modelBytes = artifacts.get("model");
  const vocabularyBytes = artifacts.get("vocabulary");
  if (!(modelBytes instanceof Uint8Array) || !(vocabularyBytes instanceof Uint8Array)) {
    throw new Error("Beam policy requires model and vocabulary artifacts.");
  }
  if (!ort?.InferenceSession || !ort?.Tensor || !ort?.env?.wasm) {
    throw new Error("The runner did not provide ONNX Runtime Web 1.27.0.");
  }
  const vocabulary = JSON.parse(new TextDecoder().decode(vocabularyBytes));
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  let msPerRow = 0.5; // adaptive cost model, refined by every batch

  async function run(nodes) {
    const n = nodes.length;
    const squares = new Int32Array(n * 64);
    const state = new Int32Array(n * 7);
    const from = new Int32Array(n * HISTORY);
    const to = new Int32Array(n * HISTORY);
    nodes.forEach((node, index) => {
      const encoded = encodePosition(node.chess);
      squares.set(encoded.squares, index * 64);
      state.set(encoded.state, index * 7);
      const history = historyArrays(node.last);
      from.set(history.from, index * HISTORY);
      to.set(history.to, index * HISTORY);
    });
    const began = Date.now();
    const result = await session.run({
      squares: new ort.Tensor("int32", squares, [n, 64]),
      state: new ort.Tensor("int32", state, [n, 7]),
      history_from: new ort.Tensor("int32", from, [n, HISTORY]),
      history_to: new ort.Tensor("int32", to, [n, HISTORY]),
    });
    msPerRow = 0.7 * msPerRow + 0.3 * Math.max(0.05, (Date.now() - began) / n);
    return { policy: result.policy.data, value: result.value.data };
  }

  function rankedMoves(node, policy, offset, filterNoisy) {
    let legal = node.chess.moves({ verbose: true });
    if (filterNoisy && !node.chess.inCheck()) {
      legal = legal.filter((move) => move.captured || move.promotion);
    }
    const scored = legal.map((move) => ({
      move,
      logit: Number(policy[offset + moveIndex(move, vocabulary.promotion_uci_moves)]),
    }));
    scored.sort((a, b) => b.logit - a.logit);
    return scored;
  }

  return {
    async newGame() {
      return {
        async chooseMove({ history, legalMoves, moveTimeLimitMs }) {
          if (!Array.isArray(legalMoves) || legalMoves.length === 0) {
            throw new Error("chooseMove requires at least one legal SAN move.");
          }
          const started = Date.now();
          const limit = Number.isFinite(moveTimeLimitMs) && moveTimeLimitMs > 0
            ? moveTimeLimitMs
            : DEFAULT_LIMIT_MS;
          const deadline = started + limit * BUDGET_FRACTION;
          const timeFor = (rows) => Date.now() + rows * msPerRow < deadline;

          const game = new Chess();
          const seen = new Map([[positionKey(game), 1]]);
          let last = [];
          for (const san of history) {
            const move = game.move(san);
            if (!move) throw new Error(`Could not replay SAN move: ${san}`);
            last = appended(last, move);
            const key = positionKey(game);
            seen.set(key, (seen.get(key) ?? 0) + 1);
          }
          if (legalMoves.length === 1) return legalMoves[0];
          const moverIsWhite = game.turn() === "w";
          const verboseBySan = new Map(game.moves({ verbose: true }).map((m) => [m.san, m]));

          // Root: value-screen every legal move.
          const roots = [];
          for (const san of legalMoves) {
            const move = verboseBySan.get(san);
            if (!move) throw new Error(`Runner/chess.js SAN mismatch: ${san}`);
            const shell = { chess: game, last, children: [] };
            const node = makeChild(shell, move);
            node.san = san;
            node.repetition = (seen.get(positionKey(node.chess)) ?? 0) >= 1;
            roots.push(node);
          }
          const open = roots.filter((node) => node.score === null);
          if (open.length > 0) {
            const screen = await run(open);
            open.forEach((node, index) => {
              node.score = whiteScore(screen.value, index * 3);
            });
          }

          // Iterative deepening within the clock: expand the screen's favourites.
          const ordered = open
            .slice()
            .sort((a, b) => (moverIsWhite ? b.score - a.score : a.score - b.score));
          let frontier = ordered.slice(0, ROOT_BEAM);
          let deepened = false;
          for (let level = 1; level < MAX_MAIN_DEPTH; level += 1) {
            if (frontier.length === 0 || frontier.length > FRONTIER_CAP) break;
            if (!timeFor(frontier.length * (1 + BEAM))) break;
            if (!deepened) {
              for (const node of frontier) node.score = null; // subtree will judge them
              deepened = true;
            }
            const results = await run(frontier);
            const next = [];
            frontier.forEach((node, index) => {
              // No stand-pat on main-line nodes: in minimax you must move.
              for (const { move } of rankedMoves(node, results.policy, index * 4272, false).slice(0, BEAM)) {
                const child = makeChild(node, move);
                if (child.score === null) next.push(child);
              }
            });
            frontier = next;
          }

          // Quiescence: extend noisy leaves (captures/promotions/checks) with
          // noisy-only continuations so the value head only judges quiet positions.
          let qFrontier = frontier;
          for (let extension = 0; extension < QUIESCENCE_MAX_PLIES; extension += 1) {
            const noisy = qFrontier.filter((node) => node.noisy);
            const quiet = qFrontier.filter((node) => !node.noisy);
            if (quiet.length > 0) {
              const values = await run(quiet);
              quiet.forEach((node, index) => {
                node.score = whiteScore(values.value, index * 3);
              });
            }
            if (noisy.length === 0 || !timeFor(noisy.length * (1 + QUIESCENCE_BEAM))) {
              qFrontier = noisy;
              break;
            }
            const results = await run(noisy);
            const next = [];
            noisy.forEach((node, index) => {
              node.standPat = whiteScore(results.value, index * 3);
              const continuations = rankedMoves(node, results.policy, index * 4272, true);
              for (const { move } of continuations.slice(0, QUIESCENCE_BEAM)) {
                const child = makeChild(node, move);
                if (child.score === null) next.push(child);
              }
            });
            qFrontier = next;
          }
          if (qFrontier.length > 0) {
            const values = await run(qFrontier);
            qFrontier.forEach((node, index) => {
              node.score = whiteScore(values.value, index * 3);
            });
          }

          let bestSan = legalMoves[0];
          let bestScore = -Infinity;
          for (const node of roots) {
            const white = backup(node);
            let mine = moverIsWhite ? white : 1 - white;
            if (node.repetition && mine > WINNING_THRESHOLD) mine -= CONTEMPT;
            if (mine > bestScore) {
              bestScore = mine;
              bestSan = node.san;
            }
          }
          return bestSan;
        },
        async dispose() {},
      };
    },
    async dispose() {
      await session.release();
    },
  };
}
