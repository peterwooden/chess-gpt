"""Play lab checkpoints: against random, each other, or engine rungs, with decode options."""

from __future__ import annotations

import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path

import chess
import torch

from chess_gpt.snapshot_model import PROMOTION_UCI_MOVES, encode_board, move_index
from lab.model import TinyPolicy

Player = Callable[[chess.Board], chess.Move]

_PROMO_INDEX = {u: i for i, u in enumerate(PROMOTION_UCI_MOVES)}


def _mirror_move(move: chess.Move) -> chess.Move:
    return chess.Move(move.from_square ^ 56, move.to_square ^ 56, promotion=move.promotion)


def move_index_for(model, board: chess.Board, move: chess.Move) -> int:
    """Model-frame move index: mirrored when a flip model sees a black-to-move position."""
    if getattr(model, "flip", False) and board.turn == chess.BLACK:
        return move_index(_mirror_move(move))
    return move_index(move)


def white_value(model, mover_is_white: bool, scores: torch.Tensor) -> torch.Tensor:
    """Convert value-head win score to white's perspective.

    Flip models output P(side-to-move wins); legacy models output white-perspective.
    """
    if getattr(model, "flip", False):
        return scores if mover_is_white else 1.0 - scores
    return scores


def load_model(checkpoint: Path) -> TinyPolicy:
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    if isinstance(saved, dict) and "sweep_recipe" in saved:
        from lab.cloud_sweep import DEFAULTS, TinyPolicy as SweepPolicy

        model = SweepPolicy({**DEFAULTS, **saved["sweep_recipe"]})
        model.load_state_dict(saved["model"])
    elif isinstance(saved, dict) and "config" in saved:
        model = TinyPolicy(**saved["config"])
        model.load_state_dict(saved["model"])
    else:  # legacy bench-S checkpoint
        model = TinyPolicy()
        model.load_state_dict(saved, strict=False)
    model.eval()
    return model


def _inputs(model: TinyPolicy, board: chess.Board) -> dict[str, torch.Tensor]:
    flipped = getattr(model, "flip", False) and board.turn == chess.BLACK
    snapshot = encode_board(board.mirror() if flipped else board)
    inputs: dict[str, torch.Tensor] = {
        "squares": torch.tensor([snapshot.squares], dtype=torch.long),
        "state": torch.tensor([snapshot.state], dtype=torch.long),
    }
    if model.history:
        moves = board.move_stack[-model.history :]
        if flipped:
            moves = [_mirror_move(m) for m in moves]
        pad = model.history - len(moves)
        inputs["history_from"] = torch.tensor(
            [[64] * pad + [m.from_square for m in moves]], dtype=torch.long
        )
        inputs["history_to"] = torch.tensor(
            [[64] * pad + [m.to_square for m in moves]], dtype=torch.long
        )
    if model.use_repetition:
        count = 2 if board.is_repetition(3) else 1 if board.is_repetition(2) else 0
        inputs["repetition"] = torch.tensor([count], dtype=torch.long)
    return inputs


def _white_score(value_logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(value_logits, dim=-1)
    return p[..., 0] + 0.5 * p[..., 1]


def make_player(
    checkpoint: Path,
    temperature: float = 0.0,
    top_k: int = 0,
    search: str = "none",
    contempt: float = 0.0,
    seed: int = 0,
    depth: int = 4,
    beam: int = 6,
    root_beam: int = 8,
    device: str = "auto",
) -> Player:
    model = load_model(checkpoint)
    if search == "beam":
        # concurrent matches thrash a single GPU; --device cpu lets them run truly in parallel
        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        device = torch.device(device)
        model.to(device)
        return _beam_player(model, device, depth, beam, root_beam, contempt)
    rng = random.Random(seed)

    def policy_choice(board: chess.Board) -> chess.Move:
        with torch.no_grad():
            logits = model(**_inputs(model, board))["policy"][0]
        legal = list(board.legal_moves)
        scores = logits[torch.tensor([move_index_for(model, board, m) for m in legal])]
        if temperature > 0:
            keep = scores.topk(min(top_k, len(legal))).indices if top_k else torch.arange(len(legal))
            probabilities = torch.softmax(scores[keep] / temperature, dim=0)
            pick = keep[torch.multinomial(probabilities, 1, generator=None)].item()
            return legal[pick]
        return legal[int(scores.argmax())]

    def value_search_choice(board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        my_turn_is_white = board.turn == chess.WHITE
        terminal: dict[int, float] = {}
        batched: list[dict[str, torch.Tensor]] = []
        batch_slots: list[int] = []
        repetition_flags: list[bool] = []
        for slot, move in enumerate(legal):
            board.push(move)
            if board.is_checkmate():
                terminal[slot] = 1.0
            elif board.is_game_over(claim_draw=True):
                terminal[slot] = 0.5
            else:
                batched.append(_inputs(model, board))
                batch_slots.append(slot)
                repetition_flags.append(board.is_repetition(2))
            board.pop()
        scores = [0.0] * len(legal)
        for slot, value in terminal.items():
            scores[slot] = value
        if batched:
            stacked = {
                key: torch.cat([b[key] for b in batched]) for key in batched[0]
            }
            with torch.no_grad():
                raw = _white_score(model(**stacked)["value"])
                white = white_value(model, not my_turn_is_white, raw)
            for i, slot in enumerate(batch_slots):
                mine = float(white[i]) if my_turn_is_white else 1.0 - float(white[i])
                if contempt > 0 and repetition_flags[i] and mine > 0.55:
                    mine -= contempt
                scores[slot] = mine
        best = max(range(len(legal)), key=scores.__getitem__)
        return legal[best]

    return value_search_choice if search == "value1" else policy_choice


def _terminal_white_score(board: chess.Board) -> float | None:
    """White-perspective score if the position ends the game, else None."""
    if board.is_checkmate():
        return 0.0 if board.turn == chess.WHITE else 1.0
    if board.is_game_over(claim_draw=True):
        return 0.5
    return None


class _Node:
    __slots__ = ("board", "score", "children", "repetition")

    def __init__(self, board: chess.Board, score: float | None, repetition: bool = False):
        self.board = board
        self.score = score  # white-perspective; None until evaluated or backed up
        self.children: list[_Node] = []
        self.repetition = repetition


def _beam_player(
    model: TinyPolicy,
    device: torch.device,
    depth: int,
    beam: int,
    root_beam: int,
    contempt: float,
) -> Player:
    """Policy-pruned minimax: value screen at the root, policy beam below, value leaves."""

    def run(boards: list[chess.Board]) -> tuple[torch.Tensor, torch.Tensor]:
        batched = [_inputs(model, board) for board in boards]
        stacked = {
            key: torch.cat([b[key] for b in batched]).to(device) for key in batched[0]
        }
        with torch.no_grad():
            output = model(**stacked)
        raw = _white_score(output["value"]).cpu()
        if getattr(model, "flip", False):
            movers_white = torch.tensor([b.turn == chess.WHITE for b in boards])
            raw = torch.where(movers_white, raw, 1.0 - raw)
        return output["policy"].cpu(), raw

    def expand(node: _Node, policy_logits: torch.Tensor, width: int) -> list[_Node]:
        legal = list(node.board.legal_moves)
        scores = policy_logits[torch.tensor([move_index_for(model, node.board, m) for m in legal])]
        for rank in scores.argsort(descending=True)[:width]:
            successor = node.board.copy()
            successor.push(legal[int(rank)])
            node.children.append(_Node(successor, _terminal_white_score(successor)))
        return [child for child in node.children if child.score is None]

    def backup(node: _Node) -> float:
        if node.score is not None:
            return node.score
        child_scores = [backup(child) for child in node.children]
        if not child_scores:  # unexpanded frontier leaf that never got valued
            node.score = 0.5
            return node.score
        node.score = (
            max(child_scores) if node.board.turn == chess.WHITE else min(child_scores)
        )
        return node.score

    def choose(board: chess.Board) -> chess.Move:
        legal = list(board.legal_moves)
        if len(legal) == 1:
            return legal[0]
        mover_is_white = board.turn == chess.WHITE

        roots: list[_Node] = []
        for move in legal:
            successor = board.copy()
            successor.push(move)
            roots.append(
                _Node(successor, _terminal_white_score(successor), successor.is_repetition(2))
            )
        open_roots = [n for n in roots if n.score is None]
        if open_roots:
            _, screen = run([n.board for n in open_roots])
            for node, value in zip(open_roots, screen):
                node.score = float(value)

        # Deepen only the root moves the 1-ply screen likes best.
        ordered = sorted(
            open_roots,
            key=lambda n: n.score if mover_is_white else -n.score,
            reverse=True,
        )
        frontier = ordered[:root_beam]
        for node in frontier:
            node.score = None  # their verdicts now come from the subtree
        for _ in range(depth - 1):
            if not frontier:
                break
            policies, values = run([n.board for n in frontier])
            next_frontier: list[_Node] = []
            for node, policy_logits in zip(frontier, policies):
                next_frontier.extend(expand(node, policy_logits, beam))
            frontier = next_frontier
        if frontier:
            _, leaf_values = run([n.board for n in frontier])
            for node, value in zip(frontier, leaf_values):
                node.score = float(value)

        best_move, best_score = legal[0], -1.0
        for move, node in zip(legal, roots):
            white = backup(node)
            mine = white if mover_is_white else 1.0 - white
            if contempt > 0 and node.repetition and mine > 0.55:
                mine -= contempt
            if mine > best_score:
                best_move, best_score = move, mine
        return best_move

    return choose


def load_policy(checkpoint: Path) -> Player:
    """Greedy argmax player — the ladder's standard decode."""
    return make_player(checkpoint)


def random_player(rng: random.Random) -> Player:
    return lambda board: rng.choice(list(board.legal_moves))


def random_opening(rng: random.Random, plies: int) -> list[chess.Move]:
    """A seeded random opening so deterministic players still produce distinct games."""
    while True:
        board = chess.Board()
        moves: list[chess.Move] = []
        for _ in range(plies):
            moves.append(rng.choice(list(board.legal_moves)))
            board.push(moves[-1])
        if not board.is_game_over():
            return moves


def play(white: Player, black: Player, opening: list[chess.Move], max_plies: int) -> str:
    board = chess.Board()
    for move in opening:
        board.push(move)
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mover = white if board.turn == chess.WHITE else black
        board.push(mover(board))
    return board.result(claim_draw=True)


def play_series(
    candidate: Player,
    opponent: Player,
    games: int,
    rng: random.Random,
    opening_plies: int = 6,
    max_plies: int = 200,
) -> dict[str, int]:
    """Color-reversed opening pairs; returns the candidate's win/draw/loss tally."""
    tally = {"win": 0, "draw": 0, "loss": 0}
    for _ in range((games + 1) // 2):
        opening = random_opening(rng, opening_plies)
        for candidate_is_white in (True, False):
            white, black = (candidate, opponent) if candidate_is_white else (opponent, candidate)
            result = play(white, black, opening, max_plies)
            if result == "1-0":
                tally["win" if candidate_is_white else "loss"] += 1
            elif result == "0-1":
                tally["loss" if candidate_is_white else "win"] += 1
            else:
                tally["draw"] += 1
    return tally


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, help="Opponent checkpoint; random-legal if omitted")
    parser.add_argument("--opponent-search", choices=("none", "value1", "beam"), default="none")
    parser.add_argument("--stockfish-elo", type=int, help="Play calibrated Stockfish (UCI_LimitStrength) instead of a checkpoint")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--search", choices=("none", "value1", "beam"), default="none")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--beam", type=int, default=6)
    parser.add_argument("--root-beam", type=int, default=8)
    parser.add_argument("--contempt", type=float, default=0.0)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--max-plies", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto",
                        help="cpu avoids GPU thrashing when several matches run concurrently")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    candidate = make_player(
        args.checkpoint, args.temperature, args.top_k, args.search, args.contempt, args.seed,
        depth=args.depth, beam=args.beam, root_beam=args.root_beam, device=args.device,
    )
    engine = None
    if args.stockfish_elo is not None:
        import chess.engine

        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": args.stockfish_elo})
        opponent = lambda board: engine.play(  # noqa: E731
            board, chess.engine.Limit(time=0.05)
        ).move
    elif args.checkpoint_b:
        opponent = make_player(
            args.checkpoint_b,
            search=args.opponent_search,
            contempt=0.15 if args.opponent_search != "none" else 0.0,
            device=args.device,
        )
    else:
        opponent = random_player(rng)
    try:
        tally = play_series(candidate, opponent, args.games, rng, args.opening_plies, args.max_plies)
    finally:
        if engine is not None:
            engine.quit()
    score = tally["win"] + 0.5 * tally["draw"]
    print(json.dumps({**tally, "score": score, "games": sum(tally.values())}))


if __name__ == "__main__":
    main()
