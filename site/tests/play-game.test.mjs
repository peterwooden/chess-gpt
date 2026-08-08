import assert from "node:assert/strict";
import test from "node:test";
import { playGame } from "../app/arena/play-game.mjs";

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

const OPTIONS = { moveTimeLimitMs: 1_000 };

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
    async predict(history, legalMoves) {
      if (history.length >= 2) throw new Error("test complete");
      return { san: legalMoves[0] };
    },
  });

  await playGame(player("white"), player("black"), { ...OPTIONS, seed: 7 });

  assert.equal(seeds.length, 2);
  assert.notEqual(seeds[0], seeds[1]);
});

test("the move time limit is passed through to the package", async () => {
  const seen = [];
  const player = (name) => ({
    name,
    async predict(history, legalMoves, moveTimeLimitMs) {
      if (history.length >= 4) throw new Error("test complete");
      seen.push(moveTimeLimitMs);
      return { san: legalMoves[0] };
    },
  });

  await playGame(player("white"), player("black"), {
    moveTimeLimitMs: 250,
  });

  assert.deepEqual(seen, [250, 250, 250, 250]);
});

test("thinking samples are associated with their turn before its move", async () => {
  const events = [];
  const player = (name) => ({
    name,
    async predict(history, legalMoves, _moveTimeLimitMs, onThinking) {
      if (history.length >= 1) throw new Error("test complete");
      onThinking?.({
        elapsedMs: 12,
        command: { type: "highlightSquare", square: "e4", intensity: 1, fadeMs: 500 },
      });
      return { san: legalMoves[0] };
    },
  });

  await playGame(player("white"), player("black"), {
    ...OPTIONS,
    onTurn(turn) {
      events.push(["turn", turn.turnId, turn.ply, turn.color]);
    },
    onThinking(sample, turn) {
      events.push(["thinking", turn.turnId, sample.elapsedMs]);
    },
    onMove(move) {
      events.push(["move", move.turnId, move.ply]);
    },
  });

  assert.deepEqual(events.slice(0, 3), [
    ["turn", "1-w", 1, "w"],
    ["thinking", "1-w", 12],
    ["move", "1-w", 1],
  ]);
});

test("per-move elapsed time is measured from the injected clock", async () => {
  let clock = 0;
  const player = (name) => ({
    name,
    async predict(history, legalMoves) {
      if (history.length >= 2) throw new Error("test complete");
      clock += 40;
      return { san: legalMoves[0] };
    },
  });

  const outcome = await playGame(player("white"), player("black"), {
    ...OPTIONS,
    now: () => clock,
  });

  assert.deepEqual(outcome.moves.map((move) => move.elapsedMs), [40, 40]);
});

test("an asynchronous move observer finishes before the next prediction starts", async () => {
  const events = [];
  const player = (name) => ({
    name,
    async predict(history, legalMoves) {
      events.push(`predict-${history.length + 1}`);
      if (history.length >= 2) throw new Error("test complete");
      return { san: legalMoves[0] };
    },
  });

  await playGame(player("white"), player("black"), {
    ...OPTIONS,
    async onMove(move) {
      await Promise.resolve();
      events.push(`publish-${move.ply}`);
    },
  });

  assert.deepEqual(events.slice(0, 4), ["predict-1", "publish-1", "predict-2", "publish-2"]);
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

test("opening moves are played by the book before either package moves", async () => {
  const historiesSeen = [];
  const player = (name) => ({
    name,
    async predict(history, legalMoves) {
      historiesSeen.push([...history]);
      if (history.length >= 8) throw new Error("test complete");
      return { san: legalMoves[0] };
    },
  });

  const opening = ["e4", "c5", "Nf3", "d6", "d4", "cxd4"];
  const outcome = await playGame(player("white"), player("black"), {
    ...OPTIONS,
    openingMoves: opening,
    openingName: "Sicilian, Open",
  });

  assert.deepEqual(historiesSeen[0], opening, "the first prediction sees the full book line");
  assert.deepEqual(
    outcome.moves.slice(0, 6).map((move) => move.san),
    opening,
  );
  for (const move of outcome.moves.slice(0, 6)) {
    assert.equal(move.actor, "book");
    assert.equal(move.elapsedMs, 0);
  }
  assert.notEqual(outcome.moves[6]?.actor, "book");
  assert.match(outcome.pgn, /\[Opening "Sicilian, Open"\]/);
  assert.match(outcome.pgn, /1\. e4 c5 2\. Nf3 d6 3\. d4 cxd4/);
});

test("an illegal opening move throws instead of forfeiting a package", async () => {
  await assert.rejects(
    playGame(firstLegal("white"), firstLegal("black"), {
      ...OPTIONS,
      openingMoves: ["e4", "e4"],
    }),
    /opening move/,
  );
});

test("playGame rejects a nonsensical move budget", async () => {
  await assert.rejects(
    playGame(firstLegal("a"), firstLegal("b"), { moveTimeLimitMs: 0 }),
    /positive moveTimeLimitMs/,
  );
});
