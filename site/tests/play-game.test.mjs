import assert from "node:assert/strict";
import test from "node:test";
import { Chess } from "chess.js";

import { MAX_PLIES_TERMINATION, playGame } from "../app/arena/play-game.mjs";

/** A player that replays a fixed script of SAN moves. */
function scripted(name, sanMoves) {
  let index = 0;
  return {
    name,
    async predict() {
      const san = sanMoves[index];
      index += 1;
      return { san };
    },
  };
}

/** A player that always plays the first legal move it is offered. */
function firstLegal(name) {
  return {
    name,
    async predict(_history, legalMoves) {
      return { san: legalMoves[0] };
    },
  };
}

const OPTIONS = { moveTimeLimitMs: 1_000, maxPlies: 200 };

test("a checkmate is recorded as a win for the mating side", async () => {
  const white = scripted("white", ["f3", "g4"]);
  const black = scripted("black", ["e5", "Qh4#"]);

  const outcome = await playGame(white, black, OPTIONS);

  assert.equal(outcome.result, "0-1");
  assert.equal(outcome.termination, "checkmate");
  assert.equal(outcome.failure, null);
  assert.equal(outcome.moves.length, 4);
  assert.match(outcome.pgn, /\[Result "0-1"\]/);
  assert.match(outcome.pgn, /\[Termination "checkmate"\]/);
});

test("a game reaching the ply cap is a draw the history layer will accept", async () => {
  const outcome = await playGame(firstLegal("a"), firstLegal("b"), {
    ...OPTIONS,
    maxPlies: 6,
  });

  assert.equal(outcome.result, "1/2-1/2");
  assert.equal(outcome.termination, MAX_PLIES_TERMINATION);
  assert.equal(outcome.moves.length, 6);
  assert.match(outcome.pgn, /\[Termination "max_plies"\]/);

  // The recorded position is deliberately not terminal, which is exactly the
  // case lib/history.ts had to grow an escape hatch for.
  const replay = new Chess();
  replay.loadPgn(outcome.pgn);
  assert.equal(replay.isGameOver(), false);
});

test("the ply cap does not pre-empt a checkmate delivered on the final ply", async () => {
  const white = scripted("white", ["f3", "g4"]);
  const black = scripted("black", ["e5", "Qh4#"]);

  const outcome = await playGame(white, black, { ...OPTIONS, maxPlies: 4 });

  assert.equal(outcome.termination, "checkmate");
  assert.equal(outcome.result, "0-1");
});

test("a package that throws forfeits the game to its opponent", async () => {
  const black = {
    name: "black",
    async predict() {
      throw new Error("the session crashed");
    },
  };

  const outcome = await playGame(firstLegal("white"), black, OPTIONS);

  assert.equal(outcome.result, "1-0");
  assert.equal(outcome.termination, "forfeit: the session crashed");
  assert.deepEqual(outcome.failure, { color: "b", reason: "the session crashed" });
  assert.match(outcome.pgn, /\[Result "1-0"\]/);
});

test("an illegal move forfeits and names the move", async () => {
  const outcome = await playGame(scripted("white", ["e9"]), firstLegal("black"), OPTIONS);

  assert.equal(outcome.result, "0-1");
  assert.equal(outcome.failure.color, "w");
  assert.match(outcome.termination, /^forfeit: returned illegal SAN move/);
});

test("a non-string move forfeits rather than corrupting the game", async () => {
  const white = { name: "white", async predict() { return { san: undefined }; } };

  const outcome = await playGame(white, firstLegal("black"), OPTIONS);

  assert.equal(outcome.result, "0-1");
  assert.equal(outcome.termination, "forfeit: returned a non-string move");
});

test("a package that cannot start loses before move one", async () => {
  const white = {
    name: "white",
    async newGame() {
      throw new Error("out of memory");
    },
    async predict() {
      throw new Error("should never be called");
    },
  };

  const outcome = await playGame(white, firstLegal("black"), OPTIONS);

  assert.equal(outcome.result, "0-1");
  assert.equal(outcome.termination, "forfeit: failed to start: out of memory");
  assert.equal(outcome.moves.length, 0);
});

test("each side receives a distinct seed", async () => {
  const seeds = [];
  const player = (name) => ({
    name,
    async newGame(seed) { seeds.push(seed); },
    async predict(_history, legalMoves) { return { san: legalMoves[0] }; },
  });

  await playGame(player("white"), player("black"), { ...OPTIONS, maxPlies: 2, seed: 7 });

  assert.equal(seeds.length, 2);
  assert.notEqual(seeds[0], seeds[1]);
});

test("the move time limit is passed through to the package", async () => {
  const seen = [];
  const player = (name) => ({
    name,
    async predict(_history, legalMoves, moveTimeLimitMs) {
      seen.push(moveTimeLimitMs);
      return { san: legalMoves[0] };
    },
  });

  await playGame(player("white"), player("black"), {
    moveTimeLimitMs: 250,
    maxPlies: 4,
  });

  assert.deepEqual(seen, [250, 250, 250, 250]);
});

test("per-move elapsed time is measured from the injected clock", async () => {
  let clock = 0;
  const player = (name) => ({
    name,
    async predict(_history, legalMoves) {
      clock += 40;
      return { san: legalMoves[0] };
    },
  });

  const outcome = await playGame(player("white"), player("black"), {
    ...OPTIONS,
    maxPlies: 2,
    now: () => clock,
  });

  assert.deepEqual(outcome.moves.map((move) => move.elapsedMs), [40, 40]);
});

test("a stalemate is a draw", async () => {
  // 1. e3 a5 2. Qh5 Ra6 3. Qxa5 h5 4. Qxc7 Rah6 5. h4 f6 6. Qxd7+ Kf7
  // 7. Qxb7 Qd3 8. Qxb8 Qh7 9. Qxc8 Kg6 10. Qe6 — black is stalemated.
  const white = scripted("white", [
    "e3", "Qh5", "Qxa5", "Qxc7", "h4", "Qxd7+", "Qxb7", "Qxb8", "Qxc8", "Qe6",
  ]);
  const black = scripted("black", [
    "a5", "Ra6", "h5", "Rah6", "f6", "Kf7", "Qd3", "Qh7", "Kg6",
  ]);

  const outcome = await playGame(white, black, OPTIONS);

  assert.equal(outcome.result, "1/2-1/2");
  assert.equal(outcome.termination, "stalemate");
});

test("an aborted game rejects rather than recording a result", async () => {
  const controller = new AbortController();
  const player = (name) => ({
    name,
    async predict(_history, legalMoves) {
      controller.abort();
      return { san: legalMoves[0] };
    },
  });

  await assert.rejects(
    playGame(player("white"), player("black"), { ...OPTIONS, signal: controller.signal }),
    (error) => error.name === "AbortError",
  );
});

test("playGame rejects a missing or nonsensical budget", async () => {
  await assert.rejects(
    playGame(firstLegal("a"), firstLegal("b"), { moveTimeLimitMs: 0, maxPlies: 10 }),
    /positive moveTimeLimitMs/,
  );
  await assert.rejects(
    playGame(firstLegal("a"), firstLegal("b"), { moveTimeLimitMs: 100, maxPlies: 0 }),
    /positive integer maxPlies/,
  );
});
