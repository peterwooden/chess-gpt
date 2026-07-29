import { Chess } from "chess.js";

const FILES = "abcdefgh";
const PIECE_CODES = {
  wp: 1,
  wn: 2,
  wb: 3,
  wr: 4,
  wq: 5,
  wk: 6,
  bp: 7,
  bn: 8,
  bb: 9,
  br: 10,
  bq: 11,
  bk: 12,
};

function classifyPhase(chess, historyLength) {
  const values = { n: 3, b: 3, r: 5, q: 9 };
  let material = 0;
  let queens = 0;
  for (const rank of chess.board()) {
    for (const piece of rank) {
      if (!piece) continue;
      material += values[piece.type] ?? 0;
      queens += Number(piece.type === "q");
    }
  }
  if (historyLength < 20 && material >= 40) return 0;
  if (material <= 18 || (queens === 0 && material <= 24)) return 2;
  return 1;
}

function encodePosition(chess, historyLength) {
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
    ep === "-" ? 64 : FILES.indexOf(ep[0]) + (Number(ep[1]) - 1) * 8,
    Math.min(Number(halfmove), 100),
  );
  return { squares, state, phase: classifyPhase(chess, historyLength) };
}

function moveIndex(move, promotionMoves) {
  const from = FILES.indexOf(move.from[0]) + (Number(move.from[1]) - 1) * 8;
  const to = FILES.indexOf(move.to[0]) + (Number(move.to[1]) - 1) * 8;
  if (!move.promotion) return from * 64 + to;
  const index = promotionMoves.indexOf(`${move.from}${move.to}${move.promotion}`);
  if (index < 0) throw new Error(`Unknown promotion move: ${move.from}${move.to}${move.promotion}`);
  return 4096 + index;
}

export async function loadPackage({ artifacts, ort }) {
  const modelBytes = artifacts.get("model");
  const vocabularyBytes = artifacts.get("vocabulary");
  if (!(modelBytes instanceof Uint8Array) || !(vocabularyBytes instanceof Uint8Array)) {
    throw new Error("Board policy requires model and vocabulary artifacts.");
  }
  if (!ort?.InferenceSession || !ort?.Tensor || !ort?.env?.wasm) {
    throw new Error("The runner did not provide ONNX Runtime Web 1.27.0.");
  }
  const vocabulary = JSON.parse(new TextDecoder().decode(vocabularyBytes));
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
          for (const san of history) {
            if (!chess.move(san)) throw new Error(`Could not replay SAN move: ${san}`);
          }
          const { squares, state, phase } = encodePosition(chess, history.length);
          const result = await session.run({
            squares: new ort.Tensor("int32", squares, [1, 64]),
            state: new ort.Tensor("int32", state, [1, 7]),
            phase: new ort.Tensor("int32", Int32Array.of(phase), [1]),
          });
          const logits = result.logits.data;
          const verboseBySan = new Map(chess.moves({ verbose: true }).map((move) => [move.san, move]));
          let bestSan = legalMoves[0];
          let bestScore = -Infinity;
          for (const san of legalMoves) {
            const move = verboseBySan.get(san);
            if (!move) throw new Error(`Runner/chess.js SAN mismatch: ${san}`);
            const score = Number(logits[moveIndex(move, vocabulary.promotion_uci_moves)]);
            if (score > bestScore) {
              bestScore = score;
              bestSan = san;
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
