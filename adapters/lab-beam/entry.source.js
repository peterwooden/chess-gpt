import { Chess } from "chess.js";

const FILES = "abcdefgh";
const HISTORY = 8;
const DEPTH = 4; // plies of lookahead
const BEAM = 5; // policy-chosen branches per internal node
const ROOT_BEAM = 6; // value-screened root moves that get deepened
const CONTEMPT = 0.15;
const WINNING_THRESHOLD = 0.55;
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
    const result = await session.run({
      squares: new ort.Tensor("int32", squares, [n, 64]),
      state: new ort.Tensor("int32", state, [n, 7]),
      history_from: new ort.Tensor("int32", from, [n, HISTORY]),
      history_to: new ort.Tensor("int32", to, [n, HISTORY]),
    });
    return { policy: result.policy.data, value: result.value.data };
  }

  function backup(node) {
    if (node.score !== null) return node.score;
    let best = null;
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
        async chooseMove({ history, legalMoves }) {
          if (!Array.isArray(legalMoves) || legalMoves.length === 0) {
            throw new Error("chooseMove requires at least one legal SAN move.");
          }
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

          // Root: every legal move gets a 1-ply value screen.
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
              score: terminalWhiteScore(chess),
              repetition: (seen.get(positionKey(chess)) ?? 0) >= 1,
              children: [],
            });
          }
          const open = roots.filter((node) => node.score === null);
          if (open.length > 0) {
            const screen = await run(open);
            open.forEach((node, index) => {
              node.score = whiteScore(screen.value, index * 3);
            });
          }

          // Deepen only the screen's favourites; their verdicts come from the subtree.
          const ordered = open
            .slice()
            .sort((a, b) => (moverIsWhite ? b.score - a.score : a.score - b.score));
          let frontier = ordered.slice(0, ROOT_BEAM);
          for (const node of frontier) node.score = null;
          for (let level = 1; level < DEPTH && frontier.length > 0; level += 1) {
            const results = await run(frontier);
            const next = [];
            frontier.forEach((node, index) => {
              const legal = node.chess.moves({ verbose: true });
              const scored = legal.map((move) => ({
                move,
                logit: Number(
                  results.policy[index * 4272 + moveIndex(move, vocabulary.promotion_uci_moves)]
                ),
              }));
              scored.sort((a, b) => b.logit - a.logit);
              for (const { move } of scored.slice(0, BEAM)) {
                const chess = new Chess(node.chess.fen());
                chess.move(move.san);
                const child = {
                  chess,
                  last: appended(node.last, move),
                  score: terminalWhiteScore(chess),
                  children: [],
                };
                node.children.push(child);
                if (child.score === null) next.push(child);
              }
            });
            frontier = next;
          }
          if (frontier.length > 0) {
            const leaves = await run(frontier);
            frontier.forEach((node, index) => {
              node.score = whiteScore(leaves.value, index * 3);
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
