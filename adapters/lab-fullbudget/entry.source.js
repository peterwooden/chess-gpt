import { Chess } from "chess.js";

const FILES = "abcdefgh";
const HISTORY = 8;
// Search shape budgeted for the single-thread WASM runner (2026-08-08 rollback):
// this model measures ~31.5 ms/eval on one thread, so a 10s clock affords ~300
// evals. Full tree at these widths is ~250-270 evals (root screen ~30 + main
// 4+12+36 + quiescence), inside the budget with margin; the runtime msPerRow
// guard still adapts to the actual machine.
const ROOT_BEAM = 4;
const BEAM = 3;
const QUIESCENCE_BEAM = 2;
const QUIESCENCE_MAX_PLIES = 4;
const MAX_MAIN_DEPTH = 4;
const FRONTIER_CAP = 8000;
const CONTEMPT = 0.15;
const WINNING_THRESHOLD = 0.55;
const BUDGET_FRACTION = 0.85;
const DEFAULT_LIMIT_MS = 3000;
const POLICY_OPENING_PLIES = 6; // no book: trust the policy head early
const PIECE_CODES = {
  wp: 1, wn: 2, wb: 3, wr: 4, wq: 5, wk: 6,
  bp: 7, bn: 8, bb: 9, br: 10, bq: 11, bk: 12,
};

function squareIndex(square) {
  return FILES.indexOf(square[0]) + (Number(square[1]) - 1) * 8;
}

function encodePosition(chess) {
  // Perspective canonicalization: black-to-move positions are mirrored so the
  // mover always looks "up" the board and plays the white pieces.
  const flipped = chess.turn() === "b";
  const squares = new Int32Array(64);
  for (let rank = 1; rank <= 8; rank += 1) {
    for (let file = 0; file < 8; file += 1) {
      const src = flipped ? `${FILES[file]}${9 - rank}` : `${FILES[file]}${rank}`;
      const piece = chess.get(src);
      let code = piece ? PIECE_CODES[`${piece.color}${piece.type}`] : 0;
      if (flipped && code) code = code <= 6 ? code + 6 : code - 6;
      squares[(rank - 1) * 8 + file] = code;
    }
  }
  const [, , castling, ep, halfmove] = chess.fen().split(" ");
  const rights = flipped
    ? [castling.includes("k"), castling.includes("q"), castling.includes("K"), castling.includes("Q")]
    : [castling.includes("K"), castling.includes("Q"), castling.includes("k"), castling.includes("q")];
  let epSquare = 64;
  if (ep !== "-") {
    epSquare = squareIndex(ep);
    if (flipped) epSquare ^= 56;
  }
  const state = Int32Array.of(
    0, Number(rights[0]), Number(rights[1]), Number(rights[2]), Number(rights[3]),
    epSquare, Math.min(Number(halfmove), 100),
  );
  return { squares, state, flipped };
}

function historyArrays(last, flipped) {
  const from = new Int32Array(HISTORY).fill(64);
  const to = new Int32Array(HISTORY).fill(64);
  const pad = HISTORY - last.length;
  last.forEach((move, index) => {
    from[pad + index] = flipped ? move.f ^ 56 : move.f;
    to[pad + index] = flipped ? move.t ^ 56 : move.t;
  });
  return { from, to };
}

function positionKey(chess) {
  return chess.fen().split(" ").slice(0, 4).join(" ");
}

function terminalWhiteScore(chess) {
  if (chess.isCheckmate()) return chess.turn() === "w" ? 0.0 : 1.0;
  if (chess.isDraw() || chess.isStalemate()) return 0.5;
  return null;
}

function moverScore(values, base) {
  const v0 = Number(values[base]);
  const v1 = Number(values[base + 1]);
  const v2 = Number(values[base + 2]);
  const peak = Math.max(v0, v1, v2);
  const e0 = Math.exp(v0 - peak);
  const e1 = Math.exp(v1 - peak);
  const e2 = Math.exp(v2 - peak);
  const total = e0 + e1 + e2;
  return e0 / total + 0.5 * (e1 / total); // P(side to move wins) + half-draw
}

function appended(last, move) {
  const next = last.concat([{ f: squareIndex(move.from), t: squareIndex(move.to) }]);
  return next.length > HISTORY ? next.slice(-HISTORY) : next;
}

export async function loadPackage({ artifacts, ort }) {
  const modelBytes = artifacts.get("model");
  const vocabularyBytes = artifacts.get("vocabulary");
  if (!(modelBytes instanceof Uint8Array) || !(vocabularyBytes instanceof Uint8Array)) {
    throw new Error("Capstone policy requires model and vocabulary artifacts.");
  }
  if (!ort?.InferenceSession || !ort?.Tensor || !ort?.env?.wasm) {
    throw new Error("The runner did not provide ONNX Runtime Web 1.27.0.");
  }
  const vocabulary = JSON.parse(new TextDecoder().decode(vocabularyBytes));
  const promoFlip = new Map();
  vocabulary.promotion_uci_moves.forEach((uci, index) => {
    const flip = (s) => s[0] + String(9 - Number(s[1]));
    const mirrored = flip(uci.slice(0, 2)) + flip(uci.slice(2, 4)) + uci.slice(4);
    promoFlip.set(uci, 4096 + vocabulary.promotion_uci_moves.indexOf(mirrored));
  });
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });
  let msPerRow = 30; // prior: measured single-thread WASM cost of this model

  function moveIndexFor(move, flipped) {
    if (move.promotion) {
      const uci = `${move.from}${move.to}${move.promotion}`;
      if (!flipped) return 4096 + vocabulary.promotion_uci_moves.indexOf(uci);
      const mapped = promoFlip.get(uci);
      if (mapped === undefined) throw new Error(`Unknown promotion move: ${uci}`);
      return mapped;
    }
    const from = squareIndex(move.from);
    const to = squareIndex(move.to);
    return flipped ? (from ^ 56) * 64 + (to ^ 56) : from * 64 + to;
  }

  async function run(nodes) {
    const n = nodes.length;
    const squares = new Int32Array(n * 64);
    const state = new Int32Array(n * 7);
    const from = new Int32Array(n * HISTORY);
    const to = new Int32Array(n * HISTORY);
    nodes.forEach((node, index) => {
      const encoded = encodePosition(node.chess);
      node.flipped = encoded.flipped;
      squares.set(encoded.squares, index * 64);
      state.set(encoded.state, index * 7);
      const history = historyArrays(node.last, encoded.flipped);
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

  function whiteScoreAt(values, index, moverIsWhite) {
    const mover = moverScore(values, index * 3);
    return moverIsWhite ? mover : 1 - mover;
  }

  function rankedMoves(node, policy, offset, filterNoisy) {
    let legal = node.chess.moves({ verbose: true });
    if (filterNoisy && !node.chess.inCheck()) {
      legal = legal.filter((move) => move.captured || move.promotion);
    }
    const scored = legal.map((move) => ({
      move,
      logit: Number(policy[offset + moveIndexFor(move, node.flipped)]),
    }));
    scored.sort((a, b) => b.logit - a.logit);
    return scored;
  }

  function backup(node) {
    if (node.score !== null) return node.score;
    let best = node.standPat;
    const whiteToMove = node.chess.turn() === "w";
    for (const child of node.children) {
      const score = backup(child);
      if (best === null || (whiteToMove ? score > best : score < best)) best = score;
    }
    node.score = best === null ? 0.5 : best;
    return node.score;
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

          // Never decline an immediate mate, even in the opening window (rules-only scan).
          for (const san of legalMoves) {
            game.move(san);
            const mate = game.isCheckmate();
            game.undo();
            if (mate) return san;
          }
          // Bookless openings: for the first plies the policy head alone decides.
          if (history.length < POLICY_OPENING_PLIES) {
            const rootNode = { chess: game, last, flipped: false };
            const out = await run([rootNode]);
            let bestSan = legalMoves[0];
            let bestLogit = -Infinity;
            for (const san of legalMoves) {
              const move = verboseBySan.get(san);
              if (!move) throw new Error(`Runner/chess.js SAN mismatch: ${san}`);
              const logit = Number(out.policy[moveIndexFor(move, rootNode.flipped)]);
              if (logit > bestLogit) {
                bestLogit = logit;
                bestSan = san;
              }
            }
            return bestSan;
          }

          const roots = [];
          for (const san of legalMoves) {
            const move = verboseBySan.get(san);
            if (!move) throw new Error(`Runner/chess.js SAN mismatch: ${san}`);
            const chess = new Chess(game.fen());
            chess.move(san);
            roots.push({
              san,
              chess,
              last: appended(last, move),
              flipped: false,
              score: terminalWhiteScore(chess),
              standPat: null,
              repetition: (seen.get(positionKey(chess)) ?? 0) >= 1,
              children: [],
            });
          }
          const open = roots.filter((node) => node.score === null);
          if (open.length > 0) {
            const screen = await run(open);
            open.forEach((node, index) => {
              node.score = whiteScoreAt(screen.value, index, node.chess.turn() === "w");
            });
          }

          const ordered = open
            .slice()
            .sort((a, b) => (moverIsWhite ? b.score - a.score : a.score - b.score));
          let frontier = ordered.slice(0, ROOT_BEAM);
          let deepened = false;
          for (let level = 1; level < MAX_MAIN_DEPTH; level += 1) {
            if (frontier.length === 0 || frontier.length > FRONTIER_CAP) break;
            if (!timeFor(frontier.length * (1 + BEAM))) break;
            if (!deepened) {
              for (const node of frontier) node.score = null;
              deepened = true;
            }
            const results = await run(frontier);
            const next = [];
            frontier.forEach((node, index) => {
              for (const { move } of rankedMoves(node, results.policy, index * 4272, false).slice(0, BEAM)) {
                const chess = new Chess(node.chess.fen());
                chess.move(move.san);
                const child = {
                  chess,
                  last: appended(node.last, move),
                  flipped: false,
                  score: terminalWhiteScore(chess),
                  standPat: null,
                  noisy: Boolean(move.captured || move.promotion) || chess.inCheck(),
                  children: [],
                };
                node.children.push(child);
                if (child.score === null) next.push(child);
              }
            });
            frontier = next;
          }

          let qFrontier = frontier;
          for (let extension = 0; extension < QUIESCENCE_MAX_PLIES; extension += 1) {
            const noisy = qFrontier.filter((node) => node.noisy);
            const quiet = qFrontier.filter((node) => !node.noisy);
            if (quiet.length > 0) {
              const values = await run(quiet);
              quiet.forEach((node, index) => {
                node.score = whiteScoreAt(values.value, index, node.chess.turn() === "w");
              });
            }
            if (noisy.length === 0 || !timeFor(noisy.length * (1 + QUIESCENCE_BEAM))) {
              qFrontier = noisy;
              break;
            }
            const results = await run(noisy);
            const next = [];
            noisy.forEach((node, index) => {
              node.standPat = whiteScoreAt(results.value, index, node.chess.turn() === "w");
              for (const { move } of rankedMoves(node, results.policy, index * 4272, true).slice(0, QUIESCENCE_BEAM)) {
                const chess = new Chess(node.chess.fen());
                chess.move(move.san);
                const child = {
                  chess,
                  last: appended(node.last, move),
                  flipped: false,
                  score: terminalWhiteScore(chess),
                  standPat: null,
                  noisy: Boolean(move.captured || move.promotion) || chess.inCheck(),
                  children: [],
                };
                node.children.push(child);
                if (child.score === null) next.push(child);
              }
            });
            qFrontier = next;
          }
          if (qFrontier.length > 0) {
            const values = await run(qFrontier);
            qFrontier.forEach((node, index) => {
              node.score = whiteScoreAt(values.value, index, node.chess.turn() === "w");
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
