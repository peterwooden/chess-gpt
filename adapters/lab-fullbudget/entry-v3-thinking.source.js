import { Chess } from "chess.js";

// v3-thinking: entry-v3.source.js plus the thinking display added to the rules
// on 2026-08-08. The search is untouched -- every constant, the simulation
// loop, and the argmax-visits choice are identical to v3, and the display is a
// pure read of the root's visit counts. See entry-v3.source.js for the search
// commentary; only the display is documented here.
//
// Visit share drives intensity. Unlike the beam adapter (entry.source.js),
// whose value-head scores across a beam span a few hundredths and had to fall
// back on rank, MCTS visit counts carry real dynamic range and evolve through
// the move, so they can be shown honestly. Redraws are throttled to a fixed
// count over the clock rather than a fixed interval: at 250-ish simulations per
// 10s move, emitting per simulation would be ~4x the runner's 64-per-500ms
// limiter and past the live broadcast's 64-per-batch ceiling, and a fixed
// interval would blow the same caps on a longer clock instead.

const FILES = "abcdefgh";
const HISTORY = 8;
const CPUCT = 1.5;
const CONTEMPT = 0.15;
const DRAW_VALUE = 0.5 - CONTEMPT / 2; // 0.425 to the root player, always
const SOFT_FRACTION = 0.88;
const DEFAULT_LIMIT_MS = 3000;
const POLICY_OPENING_PLIES = 6; // no book: trust the policy head early
const SATURATION_SIMS = 3000; // stop when terminals dominate and evals stall
const COMMAND_BUDGET = 56;
const CANDIDATE_ARROWS = 4;
const MAX_REDRAWS = 6; // bounds commands on any clock; ~1.5s apart at 10s
const MIN_REDRAW_MS = 600;
const VISIT_FLOOR = 0.35; // the 4th candidate stays visible, the best dominates
const FINAL_FADE_MS = 1500;
const PIECE_CODES = {
  wp: 1, wn: 2, wb: 3, wr: 4, wq: 5, wk: 6,
  bp: 7, bn: 8, bb: 9, br: 10, bq: 11, bk: 12,
};

function squareIndex(square) {
  return FILES.indexOf(square[0]) + (Number(square[1]) - 1) * 8;
}

/**
 * Wraps the runner's optional thinking display. The runner is required to supply
 * a synchronous non-throwing emitter, but the packaging probe harness and every
 * arena build published before 2026-08-08 omit `thinking` entirely, so absence
 * is the normal case and must degrade to a no-op. Errors are swallowed for the
 * same reason the rules give: telemetry never invalidates a valid move.
 */
function createDisplay(thinking) {
  let spent = 0;
  const emit = typeof thinking?.emit === "function" ? thinking.emit.bind(thinking) : null;
  function push(command) {
    try {
      emit(command);
    } catch {
      // Cosmetic only.
    }
  }
  return {
    send(command) {
      // Two commands stay reserved: if a pathological position ever exhausted
      // the budget mid-search, the move actually played is the one thing that
      // must still reach the board.
      if (!emit || spent >= COMMAND_BUDGET - 2) return;
      spent += 1;
      push(command);
    },
    sendFinal(command) {
      if (!emit) return;
      spent += 1;
      push(command);
    },
  };
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

function appendedSquares(last, f, t) {
  const next = last.concat([{ f, t }]);
  return next.length > HISTORY ? next.slice(-HISTORY) : next;
}

function appended(last, move) {
  return appendedSquares(last, squareIndex(move.from), squareIndex(move.to));
}

function probs3(values, base) {
  const v0 = Number(values[base]);
  const v1 = Number(values[base + 1]);
  const v2 = Number(values[base + 2]);
  const peak = Math.max(v0, v1, v2);
  const e0 = Math.exp(v0 - peak);
  const e1 = Math.exp(v1 - peak);
  const e2 = Math.exp(v2 - peak);
  const total = e0 + e1 + e2;
  return [e0 / total, e1 / total, e2 / total];
}

function freshNode(prior) {
  return { prior, n: 0, w: 0, children: new Map(), terminal: null, expanded: false, f: 64, t: 64 };
}

// Mover-perspective terminal value, or null. drawWeight is the value of a draw
// from the perspective of the side to move at this node (root contempt).
function terminalMoverValue(chess, seenCount, drawWeight) {
  if (chess.isCheckmate()) return 0.0; // side to move is mated
  if (chess.isStalemate() || chess.isInsufficientMaterial()) return drawWeight;
  if (Number(chess.fen().split(" ")[4]) >= 100) return drawWeight; // 50-move
  if (seenCount >= 3) return drawWeight; // threefold via the seen map
  return null;
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
  let msPerEval = 35; // prior: measured single-thread WASM cost of this model
  let evalCount = 0;

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

  // Single-position evaluation (batch 1); msPerEval is a running estimate kept
  // for safety margins only -- the deadline check itself uses raw elapsed time.
  async function evaluate(chess, last) {
    const encoded = encodePosition(chess);
    const history = historyArrays(last, encoded.flipped);
    const began = Date.now();
    const result = await session.run({
      squares: new ort.Tensor("int32", encoded.squares, [1, 64]),
      state: new ort.Tensor("int32", encoded.state, [1, 7]),
      history_from: new ort.Tensor("int32", history.from, [1, HISTORY]),
      history_to: new ort.Tensor("int32", history.to, [1, HISTORY]),
    });
    msPerEval = 0.7 * msPerEval + 0.3 * Math.max(1, Date.now() - began);
    evalCount += 1;
    return { policy: result.policy.data, value: result.value.data, flipped: encoded.flipped };
  }

  // Expand a leaf: children get softmax-of-logits priors over legal moves;
  // returns the leaf value from the mover's perspective at this node.
  async function expand(node, chess, last, drawWeight) {
    const out = await evaluate(chess, last);
    const legal = chess.moves({ verbose: true });
    const logits = legal.map((move) => Number(out.policy[moveIndexFor(move, out.flipped)]));
    const peak = Math.max(...logits);
    const exps = logits.map((logit) => Math.exp(logit - peak));
    const total = exps.reduce((sum, e) => sum + e, 0);
    legal.forEach((move, index) => {
      const child = freshNode(exps[index] / total);
      child.f = squareIndex(move.from);
      child.t = squareIndex(move.to);
      node.children.set(move.san, child);
    });
    node.expanded = true;
    const p = probs3(out.value, 0);
    return p[0] + drawWeight * p[1];
  }

  function selectChild(node) {
    const sqrtN = Math.sqrt(node.n + 1);
    let bestSan = null;
    let bestChild = null;
    let bestScore = -Infinity;
    for (const [san, child] of node.children) {
      const q = child.n > 0 ? 1 - child.w / child.n : 0.5;
      const u = (CPUCT * child.prior * sqrtN) / (1 + child.n);
      if (q + u > bestScore) {
        bestScore = q + u;
        bestSan = san;
        bestChild = child;
      }
    }
    return [bestSan, bestChild];
  }

  return {
    async newGame() {
      // Per-game logical state: replayed position, seen-position counts for
      // repetition, last-8-move history for the NN, and the reusable tree.
      const game = new Chess();
      const seen = new Map([[positionKey(game), 1]]);
      let last = [];
      let seenPlies = 0;
      let root = null;

      // One simulation from the root: descend by PUCT, expand or hit a
      // terminal, back up the value with alternating perspective. Path
      // increments to the seen map are undone afterwards.
      async function simulate(rootFen) {
        const scratch = new Chess(rootFen);
        let pathLast = last;
        const pushedKeys = [];
        const path = [root];
        let node = root;
        let depth = 0;
        let value;
        for (;;) {
          const drawWeight = depth % 2 === 0 ? DRAW_VALUE : 1 - DRAW_VALUE;
          if (node.terminal === null && !node.expanded) {
            const over = terminalMoverValue(
              scratch, seen.get(positionKey(scratch)) ?? 0, drawWeight,
            );
            if (over !== null) node.terminal = over;
          }
          if (node.terminal !== null) {
            value = node.terminal;
            break;
          }
          if (!node.expanded) {
            value = await expand(node, scratch, pathLast, drawWeight);
            break;
          }
          const [san, child] = selectChild(node);
          scratch.move(san);
          const key = positionKey(scratch);
          seen.set(key, (seen.get(key) ?? 0) + 1);
          pushedKeys.push(key);
          pathLast = appendedSquares(pathLast, child.f, child.t);
          node = child;
          depth += 1;
          path.push(node);
        }
        for (let i = path.length - 1; i >= 0; i -= 1) {
          path[i].n += 1;
          path[i].w += value;
          value = 1 - value;
        }
        for (const key of pushedKeys) {
          const count = seen.get(key);
          if (count <= 1) seen.delete(key);
          else seen.set(key, count - 1);
        }
      }

      return {
        async chooseMove({ history, legalMoves, moveTimeLimitMs, thinking }) {
          if (!Array.isArray(legalMoves) || legalMoves.length === 0) {
            throw new Error("chooseMove requires at least one legal SAN move.");
          }
          const display = createDisplay(thinking);
          display.send({ type: "clearAll" });
          const started = Date.now();
          const limit = Number.isFinite(moveTimeLimitMs) && moveTimeLimitMs > 0
            ? moveTimeLimitMs
            : DEFAULT_LIMIT_MS;
          const softDeadline = started + limit * SOFT_FRACTION;

          if (history.length < seenPlies) {
            // Runner rewound the game: rebuild logical state from scratch.
            game.reset();
            seen.clear();
            seen.set(positionKey(game), 1);
            last = [];
            seenPlies = 0;
            root = null;
          }
          // Advance the replayed position AND the search tree through every
          // move played since our last turn (tree reuse within the game).
          for (const san of history.slice(seenPlies)) {
            const move = game.move(san);
            if (!move) throw new Error(`Could not replay SAN move: ${san}`);
            last = appended(last, move);
            const key = positionKey(game);
            seen.set(key, (seen.get(key) ?? 0) + 1);
            if (root !== null) root = root.children.get(san) ?? null;
          }
          seenPlies = history.length;
          if (root === null) root = freshNode(1.0);

          const verboseBySan = new Map(game.moves({ verbose: true }).map((m) => [m.san, m]));

          // Announces the move about to be returned. Every exit from chooseMove
          // goes through here so the board is never left showing a stale candidate.
          const commit = (san) => {
            const move = verboseBySan.get(san);
            if (move) {
              display.sendFinal({ type: "clearAll" });
              display.sendFinal({
                type: "drawArrow",
                from: move.from,
                to: move.to,
                intensity: 1,
                fadeMs: FINAL_FADE_MS,
              });
            }
            return san;
          };

          if (legalMoves.length === 1) return commit(legalMoves[0]);

          // Never decline an immediate mate, even in the opening window
          // (rules-only scan, costs no evals).
          for (const san of legalMoves) {
            game.move(san);
            const mate = game.isCheckmate();
            game.undo();
            if (mate) return commit(san);
          }

          // Bookless openings: for the first plies the policy head alone decides.
          if (history.length < POLICY_OPENING_PLIES) {
            const out = await evaluate(game, last);
            const scored = [];
            for (const san of legalMoves) {
              const move = verboseBySan.get(san);
              if (!move) throw new Error(`Runner/chess.js SAN mismatch: ${san}`);
              scored.push({ san, move, logit: Number(out.policy[moveIndexFor(move, out.flipped)]) });
            }
            // No search to display for the first plies, so show the policy head's
            // preference instead, ranked the same way the candidates are.
            const liked = scored.slice().sort((a, b) => b.logit - a.logit).slice(0, CANDIDATE_ARROWS);
            liked.forEach(({ move }, rank) => {
              display.send({
                type: "drawArrow",
                from: move.from,
                to: move.to,
                intensity: liked.length < 2 ? 1 : 1 - (1 - VISIT_FLOOR) * (rank / (liked.length - 1)),
                fadeMs: Math.max(1000, softDeadline - Date.now()),
              });
            });
            let bestSan = legalMoves[0];
            let bestLogit = -Infinity;
            for (const { san, logit } of scored) {
              if (logit > bestLogit) {
                bestLogit = logit;
                bestSan = san;
              }
            }
            return commit(bestSan);
          }

          // Root candidates by visit share, refreshed a bounded number of times
          // across the clock. Departed candidates are cleared individually so a
          // dropped move does not linger, and fadeMs overlaps the next redraw so
          // the surviving arrows never blink.
          const redrawIntervalMs = Math.max(MIN_REDRAW_MS, (softDeadline - started) / MAX_REDRAWS);
          const shown = new Map();
          let lastRedrawAt = 0;
          const redrawCandidates = (now) => {
            if (now - lastRedrawAt < redrawIntervalMs) return;
            const ranked = [...root.children]
              .filter(([, child]) => child.n > 0)
              .sort((a, b) => b[1].n - a[1].n)
              .slice(0, CANDIDATE_ARROWS);
            if (ranked.length === 0) return;
            lastRedrawAt = now;
            const bestVisits = ranked[0][1].n;
            const surviving = new Set(ranked.map(([san]) => san));
            for (const [san, arrow] of shown) {
              if (surviving.has(san)) continue;
              display.send({ type: "clearArrow", from: arrow.from, to: arrow.to });
              shown.delete(san);
            }
            for (const [san, child] of ranked) {
              const move = verboseBySan.get(san);
              if (!move) continue;
              shown.set(san, { from: move.from, to: move.to });
              display.send({
                type: "drawArrow",
                from: move.from,
                to: move.to,
                intensity: VISIT_FLOOR + (1 - VISIT_FLOOR) * (child.n / bestVisits),
                fadeMs: redrawIntervalMs * 2,
              });
            }
          };
          // The tree is reused within a game, so on most moves the root already
          // carries the opponent-turn visits: show them before simulating rather
          // than leaving the board blank for the first interval.
          redrawCandidates(started);

          // PUCT: spend the whole budget. The soft deadline (0.88 * limit) is
          // checked before every evaluation -- each simulation performs at
          // most one, so overshoot is bounded by roughly one eval. The first
          // simulation always runs so the root has visited children. The
          // saturation guard stops eval-less spinning when the reachable tree
          // is terminal-dominated (same guard as the validated Python search).
          const rootFen = game.fen();
          let sims = 0;
          let simsAtLastEval = 0;
          for (;;) {
            const before = evalCount;
            await simulate(rootFen);
            sims += 1;
            if (evalCount > before) simsAtLastEval = sims;
            if (root.terminal !== null) break; // claimable draw at the root
            const now = Date.now();
            redrawCandidates(now);
            if (now >= softDeadline) break;
            if (sims - simsAtLastEval > SATURATION_SIMS) break;
          }

          // Argmax visits; ties break toward the lower opponent-perspective
          // average (matches the Python (n, -w/n) key).
          let bestSan = legalMoves[0];
          let bestVisits = -1;
          let bestAvg = Infinity;
          for (const [san, child] of root.children) {
            const avg = child.n > 0 ? child.w / child.n : 0.0;
            if (child.n > bestVisits || (child.n === bestVisits && avg < bestAvg)) {
              bestVisits = child.n;
              bestAvg = avg;
              bestSan = san;
            }
          }
          return commit(bestSan);
        },
        async dispose() {},
      };
    },
    async dispose() {
      await session.release();
    },
  };
}
