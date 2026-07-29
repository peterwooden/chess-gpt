"""Board-snapshot policy models and their stable tournament encodings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import chess
import torch
from torch import nn

BASE_MOVE_CLASSES = 64 * 64
PROMOTION_PIECES = "bnqr"


def _promotion_uci_moves() -> tuple[str, ...]:
    moves: list[str] = []
    for source_rank, target_rank in (("7", "8"), ("2", "1")):
        for source_file in "abcdefgh":
            source_index = ord(source_file) - ord("a")
            for target_index in range(max(0, source_index - 1), min(7, source_index + 1) + 1):
                target_file = chr(ord("a") + target_index)
                for promotion in PROMOTION_PIECES:
                    moves.append(
                        f"{source_file}{source_rank}{target_file}{target_rank}{promotion}"
                    )
    return tuple(sorted(moves))


PROMOTION_UCI_MOVES = _promotion_uci_moves()
PROMOTION_INDEX = {
    uci: BASE_MOVE_CLASSES + offset for offset, uci in enumerate(PROMOTION_UCI_MOVES)
}
MOVE_VOCAB_SIZE = BASE_MOVE_CLASSES + len(PROMOTION_UCI_MOVES)


@dataclass(frozen=True)
class BoardSnapshot:
    """The complete rule-relevant position consumed by a policy network."""

    squares: tuple[int, ...]
    state: tuple[int, ...]
    phase: int


Architecture = Literal["snapshot", "phase_moe"]


@dataclass(frozen=True)
class ModelConfig:
    architecture: Architecture = "snapshot"
    d_model: int = 256
    layers: int = 6
    heads: int = 8
    ff_multiplier: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.architecture not in {"snapshot", "phase_moe"}:
            raise ValueError(f"unsupported architecture: {self.architecture}")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if min(self.d_model, self.layers, self.heads, self.ff_multiplier) < 1:
            raise ValueError("model dimensions must be positive")


class _Expert(nn.Module):
    def __init__(self, width: int, multiplier: int, dropout: float) -> None:
        super().__init__()
        hidden = width * multiplier
        self.network = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.norm(inputs + self.network(inputs))


class SnapshotPolicy(nn.Module):
    """Transformer policy over a position, optionally routed through phase experts."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        width = config.d_model
        self.piece_embedding = nn.Embedding(13, width)
        self.square_embedding = nn.Embedding(64, width)
        self.turn_embedding = nn.Embedding(2, width)
        self.castling_embeddings = nn.ModuleList(nn.Embedding(2, width) for _ in range(4))
        self.ep_embedding = nn.Embedding(65, width)
        self.halfmove_embedding = nn.Embedding(101, width)
        self.phase_embedding = nn.Embedding(3, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.heads,
            dim_feedforward=width * config.ff_multiplier,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.layers)
        expert_count = 3 if config.architecture == "phase_moe" else 1
        self.experts = nn.ModuleList(
            _Expert(width, config.ff_multiplier, config.dropout) for _ in range(expert_count)
        )
        self.policy = nn.Linear(width, MOVE_VOCAB_SIZE)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls, std=0.02)

    def forward(
        self,
        squares: torch.Tensor,
        state: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        positions = torch.arange(64, device=squares.device)
        board_tokens = self.piece_embedding(squares) + self.square_embedding(positions)[None]
        state_token = self.cls.expand(squares.shape[0], -1, -1).squeeze(1)
        state_token = state_token + self.turn_embedding(state[:, 0])
        for offset, embedding in enumerate(self.castling_embeddings, start=1):
            state_token = state_token + embedding(state[:, offset])
        state_token = state_token + self.ep_embedding(state[:, 5])
        state_token = state_token + self.halfmove_embedding(state[:, 6])
        if self.config.architecture == "phase_moe":
            state_token = state_token + self.phase_embedding(phase)
        encoded = self.encoder(torch.cat((state_token[:, None], board_tokens), dim=1))[:, 0]

        if self.config.architecture == "phase_moe":
            expert_outputs = torch.stack([expert(encoded) for expert in self.experts], dim=1)
            selector = torch.nn.functional.one_hot(phase, num_classes=3).to(encoded.dtype)
            encoded = (expert_outputs * selector[:, :, None]).sum(dim=1)
        else:
            encoded = self.experts[0](encoded)
        return self.policy(encoded)


def classify_phase(board: chess.Board) -> int:
    """Return 0/1/2 for opening/middlegame/endgame using visible board state."""
    material = sum(
        len(board.pieces(piece_type, color)) * value
        for piece_type, value in (
            (chess.KNIGHT, 3),
            (chess.BISHOP, 3),
            (chess.ROOK, 5),
            (chess.QUEEN, 9),
        )
        for color in chess.COLORS
    )
    queens = len(board.pieces(chess.QUEEN, chess.WHITE)) + len(
        board.pieces(chess.QUEEN, chess.BLACK)
    )
    if board.ply() < 20 and material >= 40:
        return 0
    if material <= 18 or (queens == 0 and material <= 24):
        return 2
    return 1


def encode_board(board: chess.Board) -> BoardSnapshot:
    """Encode a position without retaining the SAN history that produced it."""
    squares: list[int] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        squares.append(0 if piece is None else piece.piece_type + (0 if piece.color else 6))
    ep_square = (
        board.ep_square
        if board.has_legal_en_passant() and board.ep_square is not None
        else 64
    )
    state = (
        int(board.turn == chess.BLACK),
        int(board.has_kingside_castling_rights(chess.WHITE)),
        int(board.has_queenside_castling_rights(chess.WHITE)),
        int(board.has_kingside_castling_rights(chess.BLACK)),
        int(board.has_queenside_castling_rights(chess.BLACK)),
        ep_square,
        min(board.halfmove_clock, 100),
    )
    return BoardSnapshot(tuple(squares), state, classify_phase(board))


def move_index(move: chess.Move) -> int:
    """Map a legal move to the stable 4,272-class policy vocabulary."""
    if move.promotion is not None:
        try:
            return PROMOTION_INDEX[move.uci()]
        except KeyError as error:
            raise ValueError(f"unsupported promotion move: {move.uci()}") from error
    return move.from_square * 64 + move.to_square
