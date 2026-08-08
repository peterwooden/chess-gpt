import { Chess } from "chess.js";

/** Offset applied to the black player's seed so both sides differ. */
const BLACK_SEED_OFFSET = 0x9e3779b9;

/**
 * Play one complete game between two packages.
 *
 * This is the whole of the match protocol in one place: it owns the per-move
 * time budget, forfeit attribution, and the PGN that gets recorded.
 * It deliberately knows nothing about React or the DOM so that the component
 * deciding tournament outcomes can be tested directly.
 *
 * @param {import("./play-game.js").GamePlayer} white
 * @param {import("./play-game.js").GamePlayer} black
 * @param {import("./play-game.js").PlayGameOptions} options
 * @returns {Promise<import("./play-game.js").GameOutcome>}
 */
export async function playGame(white, black, options) {
  const {
    moveTimeLimitMs,
    seed = 0,
    now = () => Date.now(),
    onMove,
    signal,
    openingMoves = [],
    openingName,
    onTurn,
    onThinking,
  } = options;

  if (!Number.isFinite(moveTimeLimitMs) || moveTimeLimitMs <= 0) {
    throw new Error("playGame requires a positive moveTimeLimitMs.");
  }
  const game = new Chess();
  /** @type {import("./play-game.js").PlayedMove[]} */
  const moves = [];
  const playerFor = (color) => (color === "w" ? white : black);

  for (const color of ["w", "b"]) {
    const player = playerFor(color);
    if (!player.newGame) continue;
    try {
      await player.newGame((color === "w" ? seed : seed ^ BLACK_SEED_OFFSET) >>> 0);
    } catch (error) {
      return finish(game, moves, color, `failed to start: ${describe(error)}`, openingName);
    }
  }

  // Book moves are the runner's, not either package's: they cost no thinking
  // time and a bad book line is a harness bug, so it throws instead of
  // forfeiting a competitor.
  for (const san of openingMoves) {
    let move;
    try {
      move = game.move(san);
    } catch {
      move = null;
    }
    if (!move) throw new Error(`The opening move “${san}” is not legal from its position.`);
    const played = {
      ply: moves.length + 1,
      san: move.san,
      color: move.color,
      from: move.from,
      to: move.to,
      captured: move.captured ?? null,
      elapsedMs: 0,
      actor: "book",
      turnId: null,
    };
    moves.push(played);
    await onMove?.(played, game);
  }

  while (true) {
    if (game.isGameOver()) break;
    if (signal?.aborted) throw new DOMException("The game was aborted.", "AbortError");

    const color = game.turn();
    const player = playerFor(color);
    const turn = { turnId: `${moves.length + 1}-${color}`, ply: moves.length + 1, color };
    await onTurn?.(turn);
    const startedAt = now();
    let san;
    try {
      const prediction = await player.predict(
        game.history(),
        game.moves(),
        moveTimeLimitMs,
        (sample) => onThinking?.(sample, turn),
      );
      san = prediction?.san;
    } catch (error) {
      return finish(game, moves, color, describe(error), openingName);
    }
    if (typeof san !== "string") {
      return finish(game, moves, color, "returned a non-string move", openingName);
    }

    let move;
    try {
      move = game.move(san);
    } catch {
      move = null;
    }
    if (!move) return finish(game, moves, color, `returned illegal SAN move “${san}”`, openingName);

    const played = {
      ply: moves.length + 1,
      san: move.san,
      color: move.color,
      from: move.from,
      to: move.to,
      captured: move.captured ?? null,
      elapsedMs: Math.max(0, now() - startedAt),
      actor: player.name,
      turnId: turn.turnId,
    };
    moves.push(played);
    await onMove?.(played, game);
  }

  const termination = terminationForPosition(game);
  const result = game.isCheckmate()
    ? (game.turn() === "w" ? "0-1" : "1-0")
    : "1/2-1/2";
  return { result, termination, moves, pgn: buildPgn(game, result, termination, openingName), failure: null };
}

/**
 * The side to move at the moment of failure loses, which matches how the arena
 * has always attributed a forfeit.
 */
function finish(game, moves, color, reason, openingName) {
  const result = color === "w" ? "0-1" : "1-0";
  const termination = `forfeit: ${reason}`;
  return {
    result,
    termination,
    moves,
    pgn: buildPgn(game, result, termination, openingName),
    failure: { color, reason },
  };
}

function terminationForPosition(game) {
  if (game.isCheckmate()) return "checkmate";
  if (game.isStalemate()) return "stalemate";
  if (game.isThreefoldRepetition()) return "threefold repetition";
  if (game.isInsufficientMaterial()) return "insufficient material";
  if (game.isDrawByFiftyMoves()) return "fifty-move rule";
  return "draw";
}

function buildPgn(game, result, termination, openingName) {
  const canonical = new Chess();
  for (const san of game.history()) canonical.move(san);
  canonical.setHeader("Result", result);
  canonical.setHeader("Termination", termination);
  if (openingName) canonical.setHeader("Opening", openingName);
  return canonical.pgn({ maxWidth: 0 });
}

function describe(error) {
  if (error instanceof Error && error.message) return error.message;
  return "failed to return a legal move";
}
