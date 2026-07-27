import { Chess } from "chess.js";

export const STOCKFISH_NODES_PER_POSITION = 30_000;
export const STOCKFISH_ENGINE_NAME = "Stockfish 18 lite";

const ENGINE_URL = "/stockfish/stockfish-18-lite-single.js";
const ENGINE_READY_TIMEOUT_MS = 45_000;
const ENGINE_SEARCH_TIMEOUT_MS = 45_000;

export async function analyzeGameWithStockfish(sanHistory, onProgress = () => {}, signal) {
  const positions = positionsFromSan(sanHistory);
  const engine = new UciEngine(ENGINE_URL, signal);
  const evaluations = [];

  try {
    await engine.initialize();
    for (let index = 0; index < positions.length; index += 1) {
      throwIfAborted(signal);
      const position = positions[index];
      const terminalScore = terminalWhiteScore(position.game);
      const evaluation = terminalScore === null
        ? await engine.analyze(position.fen, STOCKFISH_NODES_PER_POSITION)
        : { whiteScore: terminalScore, bestMoveUci: null };

      evaluations.push({
        ...evaluation,
        bestMoveSan: evaluation.bestMoveUci
          ? uciToSan(position.fen, evaluation.bestMoveUci)
          : null,
        bestMoveGap: bestMoveGapForSideToMove(
          evaluation.whiteScore,
          evaluation.secondBestWhiteScore,
          position.game.turn(),
        ),
      });
      onProgress({ completed: index + 1, total: positions.length });
    }
  } finally {
    engine.dispose();
  }

  return buildGameReview(evaluations, sanHistory);
}

export function buildGameReview(evaluations, sanHistory = []) {
  if (evaluations.length < 2) throw new Error("A game review needs at least one move.");
  const winPercents = evaluations.map(({ whiteScore }) => winPercentFromCentiPawns(whiteScore));
  const moves = [];

  for (let index = 0; index < evaluations.length - 1; index += 1) {
    const color = index % 2 === 0 ? "w" : "b";
    const before = color === "w" ? winPercents[index] : 100 - winPercents[index];
    const after = color === "w" ? winPercents[index + 1] : 100 - winPercents[index + 1];
    const loss = Math.max(0, before - after);
    const bestMoveSan = evaluations[index].bestMoveSan ?? null;
    const isBestMove = bestMoveSan !== null && sanHistory[index] === bestMoveSan;
    moves.push({
      ply: index + 1,
      color,
      accuracy: moveAccuracy(before, after),
      winningChanceLoss: loss,
      judgement: classifyPgnJudgement(loss, isBestMove, evaluations[index].bestMoveGap ?? 0),
      bestMoveSan,
    });
  }

  return {
    engine: STOCKFISH_ENGINE_NAME,
    nodesPerPosition: STOCKFISH_NODES_PER_POSITION,
    moves,
    players: {
      w: summarizePlayer("w", moves, winPercents),
      b: summarizePlayer("b", moves, winPercents),
    },
  };
}

export function winPercentFromCentiPawns(centiPawns) {
  const cp = Math.max(-1000, Math.min(1000, centiPawns));
  const winningChances = 2 / (1 + Math.exp(-0.00368208 * cp)) - 1;
  return Math.max(0, Math.min(100, 50 + 50 * winningChances));
}

export function moveAccuracy(before, after) {
  if (after >= before) return 100;
  const difference = before - after;
  const raw = 103.1668100711649 * Math.exp(-0.04354415386753951 * difference)
    - 3.166924740191411
    + 1;
  return Math.max(0, Math.min(100, raw));
}

export function classifyWinningChanceLoss(loss) {
  if (loss >= 30) return "blunder";
  if (loss >= 20) return "mistake";
  if (loss >= 10) return "inaccuracy";
  return null;
}

export function classifyPgnJudgement(loss, isBestMove, bestMoveGap = 0) {
  const error = classifyWinningChanceLoss(loss);
  if (error) return error;
  if (isBestMove && bestMoveGap >= 10) return "brilliant";
  if (isBestMove) return "good";
  return "interesting";
}

function summarizePlayer(color, moves, winPercents) {
  const playerMoves = moves.filter((move) => move.color === color);
  const counts = {
    brilliant: 0,
    good: 0,
    interesting: 0,
    inaccuracy: 0,
    mistake: 0,
    blunder: 0,
  };
  for (const move of playerMoves) {
    if (move.judgement) counts[move.judgement] += 1;
  }
  return {
    accuracy: gameAccuracy(color, moves, winPercents),
    counts,
  };
}

function gameAccuracy(color, moves, winPercents) {
  const moveCount = moves.length;
  if (moveCount === 0) return 100;
  const windowSize = Math.max(2, Math.min(8, Math.floor(moveCount / 10)));
  const windows = [];
  const repeatedOpeningWindows = Math.max(0, Math.min(windowSize, winPercents.length) - 2);
  for (let index = 0; index < repeatedOpeningWindows; index += 1) {
    windows.push(winPercents.slice(0, windowSize));
  }
  for (let index = 0; index <= winPercents.length - windowSize; index += 1) {
    windows.push(winPercents.slice(index, index + windowSize));
  }

  const weightedMoves = moves
    .map((move, index) => ({ move, weight: clamp(standardDeviation(windows[index] ?? winPercents), 0.5, 12) }))
    .filter(({ move }) => move.color === color);
  if (weightedMoves.length === 0) return 100;

  const weightTotal = weightedMoves.reduce((total, item) => total + item.weight, 0);
  const weightedMean = weightedMoves.reduce(
    (total, item) => total + item.move.accuracy * item.weight,
    0,
  ) / weightTotal;
  const hasZero = weightedMoves.some(({ move }) => move.accuracy === 0);
  const harmonicMean = hasZero
    ? 0
    : weightedMoves.length / weightedMoves.reduce((total, item) => total + 1 / item.move.accuracy, 0);
  return clamp((weightedMean + harmonicMean) / 2, 0, 100);
}

function positionsFromSan(sanHistory) {
  const game = new Chess();
  const positions = [{ fen: game.fen(), game: new Chess(game.fen()) }];
  for (const san of sanHistory) {
    game.move(san);
    positions.push({ fen: game.fen(), game: new Chess(game.fen()) });
  }
  return positions;
}

function terminalWhiteScore(game) {
  if (game.isCheckmate()) return game.turn() === "w" ? -100_000 : 100_000;
  if (game.isDraw() || game.isStalemate() || game.isThreefoldRepetition() || game.isInsufficientMaterial()) {
    return 0;
  }
  return null;
}

function uciToSan(fen, uci) {
  if (!/^[a-h][1-8][a-h][1-8][qrbn]?$/.test(uci)) return null;
  try {
    const game = new Chess(fen);
    return game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] }).san;
  } catch {
    return null;
  }
}

function parseSearchInfo(line, sideToMove) {
  if (!line.startsWith("info ") || line.includes(" lowerbound") || line.includes(" upperbound")) return null;
  const score = line.match(/\bscore (cp|mate) (-?\d+)/);
  if (!score) return null;
  const rawScore = score[1] === "mate"
    ? Math.sign(Number(score[2])) * 100_000
    : Number(score[2]);
  return {
    multipv: Number(line.match(/\bmultipv (\d+)/)?.[1] ?? 1),
    whiteScore: sideToMove === "w" ? rawScore : -rawScore,
  };
}

function bestMoveGapForSideToMove(bestWhiteScore, secondBestWhiteScore, sideToMove) {
  if (secondBestWhiteScore === null || secondBestWhiteScore === undefined) return 0;
  const bestWhiteChance = winPercentFromCentiPawns(bestWhiteScore);
  const secondWhiteChance = winPercentFromCentiPawns(secondBestWhiteScore);
  return sideToMove === "w"
    ? Math.max(0, bestWhiteChance - secondWhiteChance)
    : Math.max(0, secondWhiteChance - bestWhiteChance);
}

function sideToMoveFromFen(fen) {
  return fen.split(" ")[1] === "b" ? "b" : "w";
}

class UciEngine {
  constructor(url, signal) {
    this.worker = new Worker(url);
    this.waiter = null;
    this.latestWhiteScores = new Map();
    this.sideToMove = "w";
    this.disposed = false;
    this.worker.onmessage = (event) => this.onLine(String(event.data));
    this.worker.onerror = () => this.fail(new Error("Stockfish could not start in this browser."));
    this.abortHandler = () => this.fail(abortError());
    signal?.addEventListener("abort", this.abortHandler, { once: true });
    this.signal = signal;
  }

  async initialize() {
    await this.commandAndWait("uci", (line) => line === "uciok", ENGINE_READY_TIMEOUT_MS);
    this.worker.postMessage("setoption name Hash value 16");
    this.worker.postMessage("setoption name MultiPV value 2");
    await this.commandAndWait("isready", (line) => line === "readyok", ENGINE_READY_TIMEOUT_MS);
  }

  async analyze(fen, nodes) {
    this.latestWhiteScores = new Map();
    this.sideToMove = sideToMoveFromFen(fen);
    this.worker.postMessage(`position fen ${fen}`);
    const bestMoveLine = await this.commandAndWait(
      `go nodes ${nodes}`,
      (line) => line.startsWith("bestmove "),
      ENGINE_SEARCH_TIMEOUT_MS,
    );
    const whiteScore = this.latestWhiteScores.get(1);
    if (whiteScore === undefined) throw new Error("Stockfish returned no evaluation.");
    const bestMoveUci = bestMoveLine.split(/\s+/)[1];
    return {
      whiteScore,
      secondBestWhiteScore: this.latestWhiteScores.get(2) ?? null,
      bestMoveUci: bestMoveUci && bestMoveUci !== "(none)" ? bestMoveUci : null,
    };
  }

  commandAndWait(command, matches, timeoutMs) {
    if (this.disposed) return Promise.reject(new Error("Stockfish is no longer available."));
    if (this.waiter) return Promise.reject(new Error("Stockfish received overlapping commands."));
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        this.waiter = null;
        reject(new Error("Stockfish analysis timed out."));
      }, timeoutMs);
      this.waiter = { matches, resolve, reject, timeout };
      this.worker.postMessage(command);
    });
  }

  onLine(line) {
    const score = parseSearchInfo(line, this.sideToMove);
    if (score !== null) this.latestWhiteScores.set(score.multipv, score.whiteScore);
    if (!this.waiter || !this.waiter.matches(line)) return;
    const waiter = this.waiter;
    this.waiter = null;
    clearTimeout(waiter.timeout);
    waiter.resolve(line);
  }

  fail(error) {
    if (this.waiter) {
      clearTimeout(this.waiter.timeout);
      this.waiter.reject(error);
      this.waiter = null;
    }
    this.dispose();
  }

  dispose() {
    if (this.disposed) return;
    this.disposed = true;
    this.signal?.removeEventListener("abort", this.abortHandler);
    this.worker.terminate();
  }
}

function standardDeviation(values) {
  if (values.length === 0) return 0;
  const mean = values.reduce((total, value) => total + value, 0) / values.length;
  const variance = values.reduce((total, value) => total + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

function clamp(value, minimum, maximum) {
  return Math.max(minimum, Math.min(maximum, value));
}

function throwIfAborted(signal) {
  if (signal?.aborted) throw abortError();
}

function abortError() {
  return new DOMException("Stockfish analysis was cancelled.", "AbortError");
}
