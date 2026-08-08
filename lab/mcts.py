"""PUCT (AlphaZero-style MCTS) prototype vs the arena-shaped beam player.

Quick experiment, not production: one game, same fullbudget-68 weights on both
sides, ~250 NN evals/move each. The beam side is a Python port of the arena
adapter's retuned search (root 4 / beam 3 / depth 4 / quiescence 2, contempt
0.15) because the harness _beam_player has no quiescence.

Usage:
  uv run python -m lab.mcts sanity --checkpoint <path>
  uv run python -m lab.mcts game --checkpoint <path>
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import chess
import chess.pgn
import torch

from lab.match import _inputs, load_model, move_index_for

CPUCT = 1.5
BUDGET = 250
BUDGET_FRACTION = 0.85  # arena adapter's wall-clock safety margin
CONTEMPT = 0.15
DRAW_VALUE = 0.5 - CONTEMPT / 2  # mover-perspective draw score, both players
WINNING_THRESHOLD = 0.55

# arena adapter shape (adapters/lab-fullbudget/entry.source.js, 2026-08-08 retune)
ROOT_BEAM = 4
BEAM = 3
MAX_MAIN_DEPTH = 4
QUIESCENCE_BEAM = 2
QUIESCENCE_MAX_PLIES = 4


class Net:
    """Batched forward passes with an eval (row) counter."""

    def __init__(self, model):
        self.model = model
        self.evals = 0

    def run(self, boards: list[chess.Board]) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (policy logits [N,4272], mover-perspective [win,draw,loss] probs [N,3])."""
        batched = [_inputs(self.model, b) for b in boards]
        stacked = {k: torch.cat([b[k] for b in batched]) for k in batched[0]}
        with torch.no_grad():
            out = self.model(**stacked)
        self.evals += len(boards)
        return out["policy"], torch.softmax(out["value"], dim=-1)


def _terminal_mover_value(board: chess.Board) -> float | None:
    """Value from the perspective of the side to move, or None if not over."""
    if board.is_checkmate():
        return 0.0  # side to move is mated
    if board.is_game_over(claim_draw=True):
        return DRAW_VALUE
    return None


# ---------------------------------------------------------------- PUCT player


class PNode:
    __slots__ = ("prior", "n", "w", "children", "terminal", "expanded")

    def __init__(self, prior: float):
        self.prior = prior
        self.n = 0
        self.w = 0.0  # value sums from THIS node's side-to-move perspective
        self.children: dict[chess.Move, PNode] = {}
        self.terminal: float | None = None  # cached once computed
        self.expanded = False


class PuctPlayer:
    """v1: flat mover-perspective 0.425 draws, always burns the full budget.

    v2 flags: root_contempt applies the draw penalty from the root player's
    perspective only (0.425 at root parity, 0.575 at opponent parity, so a draw
    is always worth 0.425 to us at the root); early_stop plays instantly on a
    rules-scan mate-in-1 and stops the search once the visit leader cannot be
    overtaken in the remaining wall time.
    """

    def __init__(self, net: Net, cpuct: float = CPUCT, budget: int | None = BUDGET,
                 time_limit: float | None = None, root_contempt: bool = False,
                 early_stop: bool = False):
        self.net = net
        self.cpuct = cpuct
        self.budget = budget  # eval cap; ignored when time_limit is set
        self.time_limit = time_limit  # seconds of wall time per move
        self.root_contempt = root_contempt
        self.early_stop = early_stop
        self.root: PNode | None = None
        self.seen_plies = 0  # how much of the game stack our root reflects
        self.root_parity = 0  # ply parity of the current root (set per move)

    def _advance(self, board: chess.Board) -> None:
        """Reuse the tree: walk the root down moves played since our last search."""
        new_moves = board.move_stack[self.seen_plies :]
        for move in new_moves:
            if self.root is not None:
                self.root = self.root.children.get(move)
        self.seen_plies = len(board.move_stack)
        if self.root is None:
            self.root = PNode(prior=1.0)

    def _draw_weight(self, board: chess.Board) -> float:
        """Mover-perspective value of a draw at this node."""
        if self.root_contempt:
            same_side = (board.ply() & 1) == self.root_parity
            return DRAW_VALUE if same_side else 1.0 - DRAW_VALUE
        return DRAW_VALUE

    def _terminal_value(self, board: chess.Board) -> float | None:
        if board.is_checkmate():
            return 0.0  # side to move is mated
        if board.is_game_over(claim_draw=True):
            return self._draw_weight(board)
        return None

    def _expand(self, node: PNode, board: chess.Board) -> float:
        policy, probs = self.net.run([board])
        legal = list(board.legal_moves)
        idx = torch.tensor([move_index_for(self.net.model, board, m) for m in legal])
        priors = torch.softmax(policy[0][idx], dim=0)
        for move, prior in zip(legal, priors):
            node.children[move] = PNode(float(prior))
        node.expanded = True
        p = probs[0]
        return float(p[0]) + self._draw_weight(board) * float(p[1])

    def _select(self, node: PNode) -> tuple[chess.Move, PNode]:
        sqrt_n = math.sqrt(node.n + 1)
        best, best_score = None, -math.inf
        for move, child in node.children.items():
            q = 1.0 - child.w / child.n if child.n > 0 else 0.5
            u = self.cpuct * child.prior * sqrt_n / (1 + child.n)
            if q + u > best_score:
                best_score, best = q + u, (move, child)
        return best

    def _simulate(self, board: chess.Board) -> None:
        scratch = board.copy(stack=True)
        node = self.root
        path = [node]
        while True:
            if node.terminal is None and not node.expanded:
                over = self._terminal_value(scratch)
                if over is not None:
                    node.terminal = over
            if node.terminal is not None:
                value = node.terminal
                break
            if not node.expanded:
                value = self._expand(node, scratch)
                break
            move, child = self._select(node)
            scratch.push(move)
            node = child
            path.append(node)
        for ancestor in reversed(path):
            ancestor.n += 1
            ancestor.w += value
            value = 1.0 - value

    def choose(self, board: chess.Board) -> chess.Move:
        started = time.perf_counter()
        self._advance(board)
        self.root_parity = board.ply() & 1
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]
        if self.early_stop:
            for move in legal:  # rules-only mate scan, free and instant
                board.push(move)
                mate = board.is_checkmate()
                board.pop()
                if mate:
                    return move
        start = self.net.evals
        sims = 0
        sims_at_last_eval = 0
        while True:
            if self.time_limit is not None:
                if time.perf_counter() - started >= self.time_limit:
                    break
            elif self.net.evals - start >= self.budget:
                break
            if sims - sims_at_last_eval > 3000:
                break  # search saturated (terminal-dominated); stop burning clock
            before = self.net.evals
            self._simulate(board)
            sims += 1
            if self.net.evals > before:
                sims_at_last_eval = sims
            if (self.early_stop and self.time_limit is not None
                    and sims % 32 == 0 and sims >= 64):
                elapsed = time.perf_counter() - started
                remaining_sims = (self.time_limit - elapsed) * sims / max(elapsed, 1e-3)
                visits = sorted((c.n for c in self.root.children.values()), reverse=True)
                if len(visits) > 1 and visits[0] - visits[1] > remaining_sims:
                    break  # leader can no longer be overtaken
        best = max(
            self.root.children.items(),
            key=lambda kv: (kv[1].n, -kv[1].w / kv[1].n if kv[1].n else 0.0),
        )[0]
        return best

    def root_q(self) -> float:
        if self.root and self.root.n:
            return self.root.w / self.root.n
        return float("nan")

    def __call__(self, board: chess.Board) -> chess.Move:
        return self.choose(board)


# ------------------------------------------------- arena-shaped beam player


class BNode:
    __slots__ = ("board", "score", "stand_pat", "noisy", "repetition", "children")

    def __init__(self, board: chess.Board, score: float | None,
                 noisy: bool = False, repetition: bool = False):
        self.board = board
        self.score = score  # white-perspective
        self.stand_pat: float | None = None
        self.noisy = noisy
        self.repetition = repetition
        self.children: list[BNode] = []


class ArenaBeamPlayer:
    """Python port of adapters/lab-fullbudget/entry.source.js search, eval-capped."""

    def __init__(self, net: Net, budget: int | None = BUDGET,
                 time_limit: float | None = None):
        self.net = net
        self.budget = budget  # eval cap; ignored when time_limit is set
        self.time_limit = time_limit
        self.sec_per_row = 0.02  # EMA, adapts like the adapter's msPerRow
        self._used_this_move = 0
        self._deadline = 0.0

    def _run(self, nodes: list[BNode]) -> tuple[torch.Tensor, torch.Tensor]:
        began = time.perf_counter()
        policy, probs = self.net.run([n.board for n in nodes])
        self.sec_per_row = 0.7 * self.sec_per_row + 0.3 * max(
            5e-5, (time.perf_counter() - began) / len(nodes))
        self._used_this_move += len(nodes)
        mover = probs[:, 0] + DRAW_VALUE * probs[:, 1]
        white = torch.tensor(
            [float(m) if n.board.turn == chess.WHITE else 1.0 - float(m)
             for n, m in zip(nodes, mover)]
        )
        return policy, white

    def _budget_for(self, rows: int) -> bool:
        if self.time_limit is not None:
            return time.perf_counter() + rows * self.sec_per_row < self._deadline
        return self._used_this_move + rows <= self.budget

    def _ranked(self, node: BNode, logits: torch.Tensor, noisy_only: bool) -> list[chess.Move]:
        legal = list(node.board.legal_moves)
        if noisy_only and not node.board.is_check():
            legal = [m for m in legal if node.board.is_capture(m) or m.promotion]
        if not legal:
            return []
        idx = torch.tensor([move_index_for(self.net.model, node.board, m) for m in legal])
        order = logits[idx].argsort(descending=True)
        return [legal[int(i)] for i in order]

    def _child(self, node: BNode, move: chess.Move) -> BNode:
        successor = node.board.copy(stack=True)
        noisy = node.board.is_capture(move) or bool(move.promotion)
        successor.push(move)
        noisy = noisy or successor.is_check()
        white_terminal = None
        mover_terminal = _terminal_mover_value(successor)
        if mover_terminal is not None:
            white_terminal = (mover_terminal if successor.turn == chess.WHITE
                              else 1.0 - mover_terminal)
        child = BNode(successor, white_terminal, noisy=noisy)
        node.children.append(child)
        return child

    def _backup(self, node: BNode) -> float:
        if node.score is not None:
            return node.score
        best = node.stand_pat
        white_to_move = node.board.turn == chess.WHITE
        for child in node.children:
            score = self._backup(child)
            if best is None or (score > best if white_to_move else score < best):
                best = score
        node.score = 0.5 if best is None else best
        return node.score

    def __call__(self, board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]
        self._used_this_move = 0
        if self.time_limit is not None:
            self._deadline = time.perf_counter() + self.time_limit * BUDGET_FRACTION
        mover_is_white = board.turn == chess.WHITE

        # never decline an immediate mate (rules-only scan, free)
        for move in legal:
            board.push(move)
            mate = board.is_checkmate()
            board.pop()
            if mate:
                return move

        roots: list[BNode] = []
        for move in legal:
            successor = board.copy(stack=True)
            successor.push(move)
            mover_terminal = _terminal_mover_value(successor)
            white_terminal = None
            if mover_terminal is not None:
                white_terminal = (mover_terminal if successor.turn == chess.WHITE
                                  else 1.0 - mover_terminal)
            roots.append(BNode(successor, white_terminal,
                               repetition=successor.is_repetition(2)))
        open_roots = [n for n in roots if n.score is None]
        if open_roots:
            _, screen = self._run(open_roots)
            for node, value in zip(open_roots, screen):
                node.score = float(value)

        ordered = sorted(open_roots, key=lambda n: n.score,
                         reverse=mover_is_white)
        frontier = ordered[:ROOT_BEAM]
        deepened = False
        for _ in range(1, MAX_MAIN_DEPTH):
            if not frontier or not self._budget_for(len(frontier) * (1 + BEAM)):
                break
            if not deepened:
                for node in frontier:
                    node.score = None
                deepened = True
            policies, _ = self._run(frontier)
            next_frontier: list[BNode] = []
            for node, logits in zip(frontier, policies):
                for move in self._ranked(node, logits, noisy_only=False)[:BEAM]:
                    child = self._child(node, move)
                    if child.score is None:
                        next_frontier.append(child)
            frontier = next_frontier

        q_frontier = frontier
        for _ in range(QUIESCENCE_MAX_PLIES):
            noisy = [n for n in q_frontier if n.noisy]
            quiet = [n for n in q_frontier if not n.noisy]
            if quiet:
                _, values = self._run(quiet)
                for node, value in zip(quiet, values):
                    node.score = float(value)
            if not noisy or not self._budget_for(len(noisy) * (1 + QUIESCENCE_BEAM)):
                q_frontier = noisy
                break
            policies, values = self._run(noisy)
            next_frontier = []
            for node, logits, value in zip(noisy, policies, values):
                node.stand_pat = float(value)
                for move in self._ranked(node, logits, noisy_only=True)[:QUIESCENCE_BEAM]:
                    child = self._child(node, move)
                    if child.score is None:
                        next_frontier.append(child)
            q_frontier = next_frontier
        if q_frontier:
            _, values = self._run(q_frontier)
            for node, value in zip(q_frontier, values):
                node.score = float(value)

        best_move, best_score = legal[0], -math.inf
        for move, node in zip(legal, roots):
            white = self._backup(node)
            mine = white if mover_is_white else 1.0 - white
            if node.repetition and mine > WINNING_THRESHOLD:
                mine -= CONTEMPT
            if mine > best_score:
                best_move, best_score = move, mine
        return best_move


# ------------------------------------------------------------------ harness

MATERIAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
            chess.ROOK: 5, chess.QUEEN: 9}


def material_balance(board: chess.Board) -> int:
    total = 0
    for piece_type, value in MATERIAL.items():
        total += value * (len(board.pieces(piece_type, chess.WHITE))
                          - len(board.pieces(piece_type, chess.BLACK)))
    return total


def sanity(checkpoint: Path) -> None:
    model = load_model(checkpoint)
    probes = [
        ("mate-in-1 (white)", "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1", ["a1a8"]),
        ("mate-in-1 (black, flip path)", "r5k1/5ppp/8/8/8/8/5PPP/6K1 b - - 0 1", ["a8a1"]),
        ("hung queen recapture", "rnb1kbnr/pppp1ppp/8/4p3/3q4/4P3/PPPP1PPP/RNBQKBNR w KQkq - 0 1",
         ["e3d4"]),
        ("recapture the queen (Kxf7)", "r1bqkbnr/pppp1Qpp/2n5/4p3/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 0 3",
         ["e8f7"]),
    ]
    for spec in ("puct1", "puct2"):
        for name, fen, expected in probes:
            net, player = build_player(spec, model, time_limit=None)
            board = chess.Board(fen)
            move = player.choose(board)
            visits = sorted(((m.uci(), c.n) for m, c in player.root.children.items()),
                            key=lambda kv: -kv[1])[:4] if player.root else []
            ok = move.uci() in expected
            print(f"{'PASS' if ok else 'FAIL'}  [{spec}] {name}: chose {move.uci()} "
                  f"(expected {expected}), evals={net.evals}, top visits={visits}")


PLAYER_SPECS = ("beam", "puct1", "puct2")


def build_player(spec: str, model, time_limit: float | None):
    net = Net(model)
    budget = None if time_limit is not None else BUDGET
    if spec == "beam":
        return net, ArenaBeamPlayer(net, budget=budget, time_limit=time_limit)
    if spec == "puct1":
        return net, PuctPlayer(net, budget=budget, time_limit=time_limit)
    if spec == "puct2":
        return net, PuctPlayer(net, budget=budget, time_limit=time_limit,
                               root_contempt=True, early_stop=True)
    raise ValueError(f"unknown player spec: {spec}")


def game(checkpoint: Path, max_plies: int = 300, time_limit: float | None = None,
         white_spec: str = "puct1", black_spec: str = "beam",
         opening_seed: int = 0) -> None:
    import random as random_module

    from lab.match import random_opening

    model = load_model(checkpoint)
    white_net, white_player = build_player(white_spec, model, time_limit)
    black_net, black_player = build_player(black_spec, model, time_limit)

    board = chess.Board()
    opening: list[chess.Move] = []
    if opening_seed:
        opening = random_opening(random_module.Random(opening_seed), 6)
        for move in opening:
            board.push(move)
        print(f"opening (seed {opening_seed}): "
              f"{' '.join(m.uci() for m in opening)}", flush=True)

    stats = {"white": {"evals": [], "times": []}, "black": {"evals": [], "times": []}}
    material_log = []
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        white_to_move = board.turn == chess.WHITE
        net = white_net if white_to_move else black_net
        player = white_player if white_to_move else black_player
        spec = white_spec if white_to_move else black_spec
        side = "white" if white_to_move else "black"
        before = net.evals
        started = time.perf_counter()
        move = player(board)
        elapsed = time.perf_counter() - started
        used = net.evals - before
        stats[side]["evals"].append(used)
        stats[side]["times"].append(elapsed)
        san = board.san(move)
        board.push(move)
        material_log.append(material_balance(board))
        extra = (f" rootQ={player.root_q():.3f}"
                 if isinstance(player, PuctPlayer) else "")
        print(f"ply {board.ply():3d} {side:5s} [{spec}] "
              f"{san:8s} evals={used:4d} t={elapsed:5.2f}s "
              f"mat={material_log[-1]:+d}{extra}", flush=True)

    if board.is_game_over(claim_draw=True):
        result = board.result(claim_draw=True)
        outcome = board.outcome(claim_draw=True)
        termination = outcome.termination.name if outcome else "?"
    else:
        result = "1/2-1/2"
        termination = f"ADJUDICATED_DRAW_{max_plies}_PLIES"

    budget_tag = f"{time_limit}s/move" if time_limit is not None else "250 evals/move"
    tags = {
        "beam": f"Arena beam r4/b3/d4/q2 c0.15 {budget_tag}",
        "puct1": f"PUCT-v1 cpuct=1.5 {budget_tag}",
        "puct2": f"PUCT-v2 cpuct=1.5 root-contempt early-stop {budget_tag}",
    }
    pgn_game = chess.pgn.Game.from_board(board)
    pgn_game.headers["Event"] = (f"{white_spec} vs {black_spec} "
                                 f"(fullbudget-68, {budget_tag})")
    pgn_game.headers["White"] = tags[white_spec]
    pgn_game.headers["Black"] = tags[black_spec]
    pgn_game.headers["Result"] = result
    print("\n=== PGN ===")
    print(pgn_game)
    print(f"\nResult: {result}  Termination: {termination}  Plies: {board.ply()}")
    for side in ("white", "black"):
        evals, times = stats[side]["evals"], stats[side]["times"]
        if evals:
            print(f"{side}: avg evals/move={sum(evals)/len(evals):.1f} "
                  f"(min {min(evals)}, max {max(evals)}), "
                  f"avg time/move={sum(times)/len(times):.2f}s, moves={len(evals)}")
    swings = [(i + 1, a, b) for i, (a, b) in enumerate(zip(material_log, material_log[1:]))
              if abs(b - a) >= 3]
    print("material swings (>=3):", swings if swings else "none")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sanity", "game"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--max-plies", type=int, default=300)
    parser.add_argument("--time-per-move", type=float, default=None,
                        help="wall-clock seconds per move (replaces the 250-eval cap)")
    parser.add_argument("--white", choices=PLAYER_SPECS, default="puct1")
    parser.add_argument("--black", choices=PLAYER_SPECS, default="beam")
    parser.add_argument("--opening-seed", type=int, default=0,
                        help="seeded 6-ply random opening; 0 = standard start")
    parser.add_argument("--threads", type=int, default=0,
                        help="torch intra-op threads; 1 for parallel game processes")
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if args.command == "sanity":
        sanity(args.checkpoint)
    else:
        game(args.checkpoint, args.max_plies, args.time_per_move,
             args.white, args.black, args.opening_seed)


if __name__ == "__main__":
    main()
