"""The lab's configurable board-snapshot policy family.

Every overnight architecture/representation/objective variant is a constructor
flag here, defaulting to the ratified control (d=128 L=6 transformer, summary
readout, policy+value heads). Checkpoints save their config, so any variant
rebuilds from its .pt alone.
"""

from __future__ import annotations

import torch
from torch import nn

PIECE_CODES = 13  # empty + 6 white + 6 black
BASE_MOVES = 64 * 64
PROMOTION_MOVES = 176
MOVE_CLASSES = BASE_MOVES + PROMOTION_MOVES
HALFMOVE_CAP = 100
# codes 1..6 white PNBRQK, 7..12 black; kings carry no material value
PIECE_VALUES = [0.0, 1.0, 3.0, 3.0, 5.0, 9.0, 0.0, -1.0, -3.0, -3.0, -5.0, -9.0, 0.0]


def _geometry_bias() -> torch.Tensor:
    """65x65 additive attention bias: +1 between squares sharing rank/file/diagonal."""
    bias = torch.zeros(65, 65)
    for a in range(64):
        for b in range(64):
            same_rank = a // 8 == b // 8
            same_file = a % 8 == b % 8
            same_diag = abs(a // 8 - b // 8) == abs(a % 8 - b % 8)
            if a != b and (same_rank or same_file or same_diag):
                bias[a + 1, b + 1] = 1.0
    return bias


class TinyPolicy(nn.Module):
    def __init__(
        self,
        d_model: int = 64,
        layers: int = 2,
        heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.0,
        arch: str = "transformer",  # or "mlp"
        mlp_hidden: int = 512,
        per_square_readout: bool = False,
        moe: bool = False,
        geo_bias: bool = False,
        piece_value_init: bool = False,
        state_token: bool = True,
        material_feature: bool = False,
        repetition_feature: bool = False,
        history: int = 0,  # number of prior moves appended as tokens
        aux_material: bool = False,
        aux_plies: bool = False,
    ) -> None:
        super().__init__()
        self.arch = arch
        self.use_state_token = state_token
        self.use_material = material_feature
        self.use_repetition = repetition_feature
        self.history = history
        self.per_square_readout = per_square_readout
        self.use_moe = moe
        self.use_geo_bias = geo_bias
        assert not (arch == "mlp" and (moe or geo_bias))

        self.piece = nn.Embedding(PIECE_CODES, d_model)
        if piece_value_init:
            with torch.no_grad():
                self.piece.weight[:, 0] = torch.tensor(PIECE_VALUES) * 0.3
        self.square = nn.Parameter(torch.zeros(64, d_model))
        self.turn = nn.Embedding(2, d_model)
        self.castling = nn.ModuleList(nn.Embedding(2, d_model) for _ in range(4))
        self.en_passant = nn.Embedding(65, d_model)
        self.halfmove = nn.Embedding(HALFMOVE_CAP + 1, d_model)
        if material_feature:
            self.material = nn.Embedding(41, d_model)  # balance clamped to ±20
            self.register_buffer("piece_values", torch.tensor(PIECE_VALUES))
        if repetition_feature:
            self.repetition = nn.Embedding(4, d_model)
        if history:
            self.history_from = nn.Embedding(65, d_model)  # 64 = no-move padding
            self.history_to = nn.Embedding(65, d_model)
            self.history_position = nn.Parameter(torch.zeros(history, d_model))

        if arch == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model, heads, dim_feedforward=ff_mult * d_model,
                batch_first=True, norm_first=True, dropout=dropout,
            )
            self.encoder = nn.TransformerEncoder(layer, layers)
            if geo_bias:
                self.register_buffer("geo", _geometry_bias())
        else:
            # Token-in, token-out MLP trunk: same readout heads as the transformer.
            width = (65 + history) * d_model
            blocks: list[nn.Module] = []
            for _ in range(layers):
                blocks += [nn.Linear(width, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
                width = mlp_hidden
            blocks.append(nn.Linear(width, 65 * d_model))
            self.mlp = nn.Sequential(*blocks)

        if moe:
            self.experts = nn.ModuleList(
                nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU()) for _ in range(3)
            )
            self.register_buffer("moe_values", torch.tensor(PIECE_VALUES).abs())

        if per_square_readout:
            self.to_square = nn.Linear(d_model, 64)  # each square scores its destinations
            self.promotions = nn.Linear(d_model, PROMOTION_MOVES)
        else:
            self.policy = nn.Linear(d_model, MOVE_CLASSES)
        self.value = nn.Linear(d_model, 3)  # white wins / draw / black wins
        self.aux_material_head = nn.Linear(d_model, 41) if aux_material else None
        self.aux_plies_head = nn.Linear(d_model, 8) if aux_plies else None

    def _state_vector(self, squares: torch.Tensor, state: torch.Tensor,
                      repetition: torch.Tensor | None) -> torch.Tensor:
        vector = self.turn(state[:, 0])
        for index, embedding in enumerate(self.castling):
            vector = vector + embedding(state[:, 1 + index])
        vector = vector + self.en_passant(state[:, 5])
        vector = vector + self.halfmove(state[:, 6].clamp(max=HALFMOVE_CAP))
        if self.use_material:
            balance = self.piece_values[squares].sum(dim=1).round().long().clamp(-20, 20)
            vector = vector + self.material(balance + 20)
        if self.use_repetition:
            counts = repetition if repetition is not None else torch.zeros(
                len(squares), dtype=torch.long, device=squares.device
            )
            vector = vector + self.repetition(counts.clamp(0, 3))
        return vector

    def forward(
        self,
        squares: torch.Tensor,
        state: torch.Tensor,
        history_from: torch.Tensor | None = None,
        history_to: torch.Tensor | None = None,
        repetition: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        board = self.piece(squares) + self.square
        state_vector = self._state_vector(squares, state, repetition)

        if self.arch == "mlp":
            parts = [state_vector[:, None], board]
            if self.history:
                moves = self.history_from(history_from) + self.history_to(history_to)
                parts.append(moves + self.history_position)
            tokens = self.mlp(torch.cat(parts, dim=1).flatten(1)).view(-1, 65, board.shape[-1])
            summary = tokens[:, 0]
        else:
            parts = [state_vector[:, None] if self.use_state_token else
                     torch.zeros_like(state_vector)[:, None], board]
            if self.history:
                moves = self.history_from(history_from) + self.history_to(history_to)
                parts.append(moves + self.history_position)
            sequence = torch.cat(parts, dim=1)
            mask = None
            if self.use_geo_bias:
                n = sequence.shape[1]
                mask = torch.zeros(n, n, device=sequence.device)
                mask[:65, :65] = self.geo
            tokens = self.encoder(sequence, mask=mask)
            summary = tokens[:, 0]

        if self.use_moe:
            material = self.moe_values[squares].sum(dim=1)
            gate = torch.bucketize(material, torch.tensor([30.0, 60.0], device=material.device))
            expert_out = torch.stack([expert(summary) for expert in self.experts], dim=1)
            summary = summary + expert_out.gather(
                1, (2 - gate).view(-1, 1, 1).expand(-1, 1, summary.shape[-1])
            ).squeeze(1)

        if self.per_square_readout:
            base = self.to_square(tokens[:, 1:65]).flatten(1)  # (B, 64*64) as from*64+to
            policy = torch.cat((base, self.promotions(summary)), dim=1)
        else:
            policy = self.policy(summary)

        output = {"policy": policy, "value": self.value(summary)}
        if self.aux_material_head is not None:
            output["aux_material"] = self.aux_material_head(summary)
        if self.aux_plies_head is not None:
            output["aux_plies"] = self.aux_plies_head(summary)
        return output
