import { Chess } from "chess.js";

const FILES = "abcdefgh";
const HISTORY = 8;
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

function historyArrays(chess) {
  const moves = chess.history({ verbose: true }).slice(-HISTORY);
  const from = new Int32Array(HISTORY).fill(64);
  const to = new Int32Array(HISTORY).fill(64);
  const pad = HISTORY - moves.length;
  moves.forEach((move, index) => {
    from[pad + index] = squareIndex(move.from);
    to[pad + index] = squareIndex(move.to);
  });
  return { from, to };
}

function positionKey(chess) {
  return chess.fen().split(" ").slice(0, 4).join(" ");
}

function moverScore(v0, v1, v2, moverIsWhite) {
  const peak = Math.max(v0, v1, v2);
  const e0 = Math.exp(v0 - peak);
  const e1 = Math.exp(v1 - peak);
  const e2 = Math.exp(v2 - peak);
  const total = e0 + e1 + e2;
  const white = e0 / total + 0.5 * (e1 / total);
  return moverIsWhite ? white : 1 - white;
}

export async function loadPackage({ artifacts, ort }) {
  const modelBytes = artifacts.get("model");
  if (!(modelBytes instanceof Uint8Array)) {
    throw new Error("Value-search policy requires the model artifact.");
  }
  if (!ort?.InferenceSession || !ort?.Tensor || !ort?.env?.wasm) {
    throw new Error("The runner did not provide ONNX Runtime Web 1.27.0.");
  }
  ort.env.wasm.numThreads = 1;
  const session = await ort.InferenceSession.create(modelBytes, { executionProviders: ["wasm"] });

  return {
    async newGame() {
      return {
        async chooseMove({ history, legalMoves }) {
          if (!Array.isArray(legalMoves) || legalMoves.length === 0) {
            throw new Error("chooseMove requires at least one legal SAN move.");
          }
          const chess = new Chess();
          const seen = new Map([[positionKey(chess), 1]]);
          for (const san of history) {
            if (!chess.move(san)) throw new Error(`Could not replay SAN move: ${san}`);
            const key = positionKey(chess);
            seen.set(key, (seen.get(key) ?? 0) + 1);
          }
          const moverIsWhite = chess.turn() === "w";

          // One ply of consequence: score every legal move by the value head's
          // verdict on the position it creates.
          const rows = [];
          const batch = { squares: [], state: [], from: [], to: [] };
          for (const san of legalMoves) {
            if (!chess.move(san)) throw new Error(`Runner/chess.js SAN mismatch: ${san}`);
            const row = { san };
            if (chess.isCheckmate()) {
              row.terminal = 1.0;
            } else if (chess.isDraw() || chess.isStalemate()) {
              row.terminal = 0.5;
            } else {
              row.repetition = (seen.get(positionKey(chess)) ?? 0) >= 1;
              const { squares, state } = encodePosition(chess);
              const { from, to } = historyArrays(chess);
              batch.squares.push(squares);
              batch.state.push(state);
              batch.from.push(from);
              batch.to.push(to);
              row.batchIndex = batch.squares.length - 1;
            }
            rows.push(row);
            chess.undo();
          }

          let values = null;
          if (batch.squares.length > 0) {
            const n = batch.squares.length;
            const flatten = (list, width) => {
              const out = new Int32Array(n * width);
              list.forEach((entry, index) => out.set(entry, index * width));
              return out;
            };
            const result = await session.run({
              squares: new ort.Tensor("int32", flatten(batch.squares, 64), [n, 64]),
              state: new ort.Tensor("int32", flatten(batch.state, 7), [n, 7]),
              history_from: new ort.Tensor("int32", flatten(batch.from, HISTORY), [n, HISTORY]),
              history_to: new ort.Tensor("int32", flatten(batch.to, HISTORY), [n, HISTORY]),
            });
            values = result.value.data;
          }

          let bestSan = legalMoves[0];
          let bestScore = -Infinity;
          for (const row of rows) {
            let score;
            if (row.terminal !== undefined) {
              score = row.terminal;
            } else {
              const base = row.batchIndex * 3;
              score = moverScore(
                Number(values[base]), Number(values[base + 1]), Number(values[base + 2]),
                moverIsWhite,
              );
              if (row.repetition && score > WINNING_THRESHOLD) score -= CONTEMPT;
            }
            if (score > bestScore) {
              bestScore = score;
              bestSan = row.san;
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
