# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "pyarrow", "huggingface_hub", "schedulefree"]
# ///
"""Fleet trainer v3: recipe via RECIPE env JSON, metrics pushed to the shard repo."""

import json
import math
import os
import shutil
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn
from torch.nn import functional as F

DATASET = "peterwooden/chess-gpt-lab-shards"
SHARDS = ["shards/enriched-240k.parquet", "shards/enriched-fresh240k.parquet"]
ELITE_SHARD = "shards/winner2000-enriched.parquet"
BIG_SHARD = "shards/enriched-960k-b.parquet"
TEACHER_REPO = "peterwooden/chess-gpt-cloud-57"
HALFMOVE_CAP, PROMOTION_MOVES = 100, 176

DEFAULTS = {
    "id": "control", "optimizer": "adamw", "lr": 1.2e-3, "wd": 0.01, "betas": [0.9, 0.999],
    "eps": 1e-8, "clip": 0.0, "schedule": "cosine", "warmup": 0.02, "cosine_floor": 0.0,
    "cycles": 1, "embed_lr_scale": 1.0, "batch": 1024, "epochs": 2, "ema": 0.0, "swa": False,
    "grad_noise": 0.0, "arch": "mlp", "heads": 4, "ffn_ratio": 4, "attn_bias": False,
    "mega_corpus": False, "activation": "relu", "residual": False, "block_norm": False,
    "gated": False, "two_tower": False, "input_norm": False, "input_residual": False,
    "token_rank": 0, "untied_readout": False, "layers": 2, "hidden": 1152, "d_model": 128,
    "dropout": 0.1, "label_smoothing": 0.0, "value_weight": 1.0, "value_mode": "ce",
    "value_ply_weight": False, "value_decisive_only": False, "focal": False,
    "entropy_bonus": 0.0, "aux_plies": False, "aux_material": False, "aux_next_move": False,
    "aux_masked": False, "elite_mix": 0.0, "endgame_oversample": False, "curriculum": "none",
    "distill": False, "compile": True, "seed": 20260730, "save_ckpt": False,
    "history_k": 8, "shuffle_history": False, "zero_halfmove": False, "zero_castling": False,
    "rankfile_squares": False, "input_square_dropout": 0.0, "shuffle_squares": False,
    "ply_min": 0, "ply_max": 999, "subsample": 1.0, "label_noise": 0.0, "big_shard": False,
    "flip": False, "shard_set": "legacy", "init_from": "",
    "cnn_blocks": 8, "cnn_filters": 128, "cnn_rays": False, "cnn_fullrays": False, "cnn_knightmask": False,
    "cnn_modern": False, "bilinear_head": False,
    "qkv_tie": "", "max_steps": 0, "time_budget_s": 0.0, "val_shard": "", "compile_mode": "",
    "ckpt_every_frac": 0.0, "resume_from": "",
}

# Which projection each of q, k, v reads from. "" keeps the legacy fused qkv weight so
# older checkpoints still load; "none" is the split-projection control for tie experiments.
TIE_ROLES = {
    "none": ("q", "k", "v"), "qk": ("qk", "qk", "v"), "kv": ("q", "kv", "kv"),
    "qv": ("qv", "k", "qv"), "qkv": ("qkv", "qkv", "qkv"),
}

SHARD_SETS = {
    "elo1600": [f"shards/elo1600-{c}.parquet" for c in "abcd"],
    "elite2600": [f"shards/elite2600-{c}.parquet" for c in "ab"],
    "fullbudget-nobullet": [
        f"shards/fullbudget-nobullet-{month}.parquet" for month in ("jan", "feb", "mar")
    ],
    **{
        f"slice67-{name}": [f"shards/slice67-{name}.parquet"]
        for name in (
            "control", "unfiltered", "elo1800", "elo2000", "elo2200",
            "nobullet", "decisive", "noforfeit", "dedup64",
        )
    },
}


def _promotion_moves():
    moves = []
    for source_rank, target_rank in (("7", "8"), ("2", "1")):
        for f in "abcdefgh":
            i = ord(f) - ord("a")
            for j in range(max(0, i - 1), min(7, i + 1) + 1):
                for piece in "bnqr":
                    moves.append(f"{f}{source_rank}{chr(ord('a') + j)}{target_rank}{piece}")
    return sorted(moves)


def _flip_uci(uci):
    def sq(s):
        return s[0] + str(9 - int(s[1]))
    return sq(uci[:2]) + sq(uci[2:4]) + uci[4:]


PROMOS = _promotion_moves()
PROMO_INDEX = {u: i for i, u in enumerate(PROMOS)}
PROMO_FLIP = np.array([PROMO_INDEX[_flip_uci(u)] for u in PROMOS], dtype=np.int64)


def flip_in_place(part):
    """Canonicalize to side-to-move perspective: mirror ranks, swap colors, remap targets."""
    black = part["state"][:, 0] == 1
    if not black.any():
        return
    sq_perm = np.arange(64) ^ 56
    squares = part["squares"][black][:, sq_perm]
    part["squares"][black] = np.where(
        squares == 0, 0, np.where(squares <= 6, squares + 6, squares - 6)
    )
    state = part["state"][black]
    state[:, 0] = 0
    state[:, [1, 2, 3, 4]] = state[:, [3, 4, 1, 2]]
    state[:, 5] = np.where(state[:, 5] < 64, state[:, 5] ^ 56, 64)
    part["state"][black] = state
    for name in ("history_from", "history_to"):
        h = part[name][black]
        part[name][black] = np.where(h < 64, h ^ 56, 64)
    t = part["target"][black]
    base = t < 4096
    flipped = np.where(base, ((t // 64) ^ 56) * 64 + ((t % 64) ^ 56),
                       4096 + PROMO_FLIP[np.clip(t - 4096, 0, 175)])
    part["target"][black] = flipped
    part["result"][black] = 2 - part["result"][black]
    part["future_material"][black] = -part["future_material"][black]


class BiasedBlock(nn.Module):
    """Pre-norm transformer block with a learned per-head additive attention bias."""

    def __init__(self, d, heads, ffn_ratio, tokens=73, tie=""):
        super().__init__()
        self.heads, self.dh = heads, d // heads
        self.norm1 = nn.LayerNorm(d)
        self.tie = tie
        if tie:
            self.roles = TIE_ROLES[tie]
            self.projs = nn.ModuleDict(
                {name: nn.Linear(d, d, bias=False) for name in dict.fromkeys(self.roles)}
            )
        else:
            self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d)
        self.bias = nn.Parameter(torch.zeros(heads, tokens, tokens))
        self.norm2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_ratio * d), nn.GELU(), nn.Linear(ffn_ratio * d, d)
        )

    def forward(self, tokens):
        b, n, d = tokens.shape
        x = self.norm1(tokens)
        if self.tie:
            # each distinct projection is computed once, so a tie saves compute as well as weights
            cache = {name: proj(x) for name, proj in self.projs.items()}
            q, k, v = (cache[name] for name in self.roles)
        else:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
        shape = (b, n, self.heads, self.dh)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        mixed = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=self.bias[:, :n, :n]
        )
        tokens = tokens + self.out(mixed.transpose(1, 2).reshape(b, n, d))
        return tokens + self.ffn(self.norm2(tokens))


class ConvBlock(nn.Module):
    """Residual 3x3 block with KataGo-style global pooling bias."""

    def __init__(self, filters):
        super().__init__()
        self.conv1 = nn.Conv2d(filters, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters, 3, padding=1)
        self.norm1 = nn.BatchNorm2d(filters)
        self.norm2 = nn.BatchNorm2d(filters)
        self.pool_bias = nn.Linear(filters, filters)

    def forward(self, x):
        h = torch.relu(self.norm1(self.conv1(x)))
        pooled = self.pool_bias(h.mean(dim=(2, 3)))
        h = h + pooled[:, :, None, None]
        return torch.relu(x + self.norm2(self.conv2(h)))


class RayBranch(nn.Module):
    """Full-rank and full-file kernels: every square sees its whole row and column."""

    def __init__(self, filters):
        super().__init__()
        self.rank = nn.Conv2d(filters, filters // 2, (1, 15), padding=(0, 7))
        self.file = nn.Conv2d(filters, filters // 2, (15, 1), padding=(7, 0))
        self.mix = nn.Conv2d(filters * 2, filters, 1)

    def forward(self, x):
        rays = torch.cat([torch.relu(self.rank(x)), torch.relu(self.file(x)), x], dim=1)
        return torch.relu(x + self.mix(rays))


class FullRayBranch(nn.Module):
    """All sliding directions plus the knight leap: rank, file, both diagonals, 5x5.

    Diagonals via shear: shift row r sideways by r so diagonals become columns,
    apply a vertical 15-kernel, shear back. Linear kernels see the whole ray;
    blocking logic is left to the nonlinear layers above.
    """

    def __init__(self, filters):
        super().__init__()
        g = filters // 4
        self.rank = nn.Conv2d(filters, g, (1, 15), padding=(0, 7))
        self.file = nn.Conv2d(filters, g, (15, 1), padding=(7, 0))
        self.diag = nn.Conv2d(filters, g, (15, 1), padding=(7, 0))
        self.anti = nn.Conv2d(filters, g, (15, 1), padding=(7, 0))
        self.leap = nn.Conv2d(filters, g, 5, padding=2)
        knight = torch.zeros(1, 1, 5, 5)
        for dr, dc in ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -2 + 1)):
            knight[0, 0, 2 + dr, 2 + dc] = 1.0
        knight[0, 0, 2, 2] = 1.0  # centre tap: condition on the piece itself
        self.register_buffer("knight_mask", knight)
        self.mask_leap = False
        self.mix = nn.Conv2d(filters + 5 * g, filters, 1)
        rows = torch.arange(8)[:, None]
        cols = torch.arange(15)[None, :]
        self.register_buffer("shear_main", (cols + rows).expand(8, 15).clone())
        self.register_buffer("shear_anti", (cols + (7 - rows)).expand(8, 15).clone())
        out_cols = torch.arange(8)[None, :]
        self.register_buffer("unshear_main", (out_cols - rows + 7).expand(8, 8).clone())
        self.register_buffer("unshear_anti", (out_cols - (7 - rows) + 7).expand(8, 8).clone())

    def _ray(self, x, conv, shear, unshear):
        b, c = x.shape[:2]
        padded = torch.nn.functional.pad(x, (7, 7))  # (B, C, 8, 22)
        sheared = padded.gather(3, shear[None, None].expand(b, c, 8, 15))
        out = torch.relu(conv(sheared))  # vertical kernel now reads a full diagonal
        return out.gather(3, unshear[None, None].expand(b, out.shape[1], 8, 8))

    def forward(self, x):
        parts = [
            x,
            torch.relu(self.rank(x)),
            torch.relu(self.file(x)),
            self._ray(x, self.diag, self.shear_main, self.unshear_main),
            self._ray(x, self.anti, self.shear_anti, self.unshear_anti),
            torch.relu(
                torch.nn.functional.conv2d(
                    x, self.leap.weight * self.knight_mask, self.leap.bias, padding=2
                ) if self.mask_leap else self.leap(x)
            ),
        ]
        return torch.relu(x + self.mix(torch.cat(parts, dim=1)))


class ModernBlock(nn.Module):
    """ConvNeXt-style chess block: 7x7 depthwise, inverted-bottleneck FFN, GELU,
    squeeze-excite gate, mean+max global pooling bias, depthwise compass rays."""

    def __init__(self, f):
        super().__init__()
        self.dw = nn.Conv2d(f, f, 7, padding=3, groups=f)
        self.norm = nn.GroupNorm(1, f)
        self.expand = nn.Conv2d(f, 4 * f, 1)
        self.project = nn.Conv2d(4 * f, f, 1)
        self.se = nn.Sequential(nn.Linear(f, f // 8), nn.GELU(), nn.Linear(f // 8, f))
        self.pool_bias = nn.Linear(2 * f, f)
        self.gamma = nn.Parameter(torch.full((f, 1, 1), 1e-6))  # LayerScale: branch starts silent
        self.ray_rank = nn.Conv2d(f, f, (1, 15), padding=(0, 7), groups=f)
        self.ray_file = nn.Conv2d(f, f, (15, 1), padding=(7, 0), groups=f)
        self.ray_diag = nn.Conv2d(f, f, (15, 1), padding=(7, 0), groups=f)
        self.ray_anti = nn.Conv2d(f, f, (15, 1), padding=(7, 0), groups=f)
        self.ray_leap = nn.Conv2d(f, f, 5, padding=2, groups=f)
        self.ray_gates = nn.Parameter(torch.zeros(5))
        rows = torch.arange(8)[:, None]
        cols15 = torch.arange(15)[None, :]
        out_cols = torch.arange(8)[None, :]
        self.register_buffer("sh_m", (cols15 + rows).expand(8, 15).clone())
        self.register_buffer("sh_a", (cols15 + (7 - rows)).expand(8, 15).clone())
        self.register_buffer("un_m", (out_cols - rows + 7).expand(8, 8).clone())
        self.register_buffer("un_a", (out_cols - (7 - rows) + 7).expand(8, 8).clone())

    @torch.compiler.disable
    def _sheared(self, x, conv, shear, unshear):
        b, c = x.shape[:2]
        padded = torch.nn.functional.pad(x, (7, 7))
        sheared = padded.gather(3, shear[None, None].expand(b, c, 8, 15))
        out = conv(sheared)
        return out.gather(3, unshear[None, None].expand(b, c, 8, 8))

    def forward(self, x):
        base = self.norm(x)  # pre-norm feeds every branch input
        h = self.dw(base)
        g = torch.tanh(self.ray_gates)
        h = h + g[0] * self.ray_rank(base) + g[1] * self.ray_file(base)
        h = h + g[2] * self._sheared(base, self.ray_diag, self.sh_m, self.un_m)
        h = h + g[3] * self._sheared(base, self.ray_anti, self.sh_a, self.un_a)
        h = h + g[4] * self.ray_leap(base)
        h = self.project(torch.nn.functional.gelu(self.expand(h)))
        h = h * torch.sigmoid(self.se(h.mean(dim=(2, 3))))[:, :, None, None]
        h = h + self.pool_bias(torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1))[:, :, None, None]
        return x + self.gamma * h




class BEB(nn.Module):
    """Board-embedding broadcast: spatially-weighted global summary added everywhere."""

    def __init__(self, c):
        super().__init__()
        self.summary = nn.Conv2d(c, c // 2, 8)  # valid conv -> (B, C/2, 1, 1)
        self.up = nn.Conv2d(c // 2, c, 1)

    def forward(self, x):
        return x + self.up(torch.relu(self.summary(x)))


class CNN2Block(nn.Module):
    """Pre-activation residual block; second conv zero-initialized (SkipInit)."""

    def __init__(self, c, ray=None):
        super().__init__()
        self.norm = nn.BatchNorm2d(c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        if ray == "rank":
            self.conv2 = nn.Conv2d(c, c, (1, 15), padding=(0, 7))
        elif ray == "file":
            self.conv2 = nn.Conv2d(c, c, (15, 1), padding=(7, 0))
        else:
            self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x):
        h = torch.relu(self.conv1(torch.relu(self.norm(x))))
        return x + self.conv2(h)


class ChessCNN2(nn.Module):
    """Fresh-design dense CNN: few fat kernels, bilinear policy head."""

    def __init__(self, r):
        super().__init__()
        c, n = r["cnn_filters"], r["cnn_blocks"]
        self.history = 8
        self.use_repetition = False
        self.flip = r["flip"]
        self.stem = nn.Conv2d(36, c, 5, padding=2)
        self.stem_norm = nn.BatchNorm2d(c)
        rays = {n * 3 // 8: "rank", n * 6 // 8: "file"}
        self.blocks = nn.ModuleList(CNN2Block(c, rays.get(i)) for i in range(n))
        self.bebs = nn.ModuleDict({str(i): BEB(c) for i in (n // 4, n // 2, 3 * n // 4)})
        self.f_proj = nn.Conv2d(c, 128, 1)
        self.t_proj = nn.Conv2d(c, 128, 1)
        self.promotions = nn.Linear(128, PROMOTION_MOVES)
        self.value_reduce = nn.Conv2d(c, 32, 1)
        self.value_fc1 = nn.Linear(32 * 64, 128)
        self.value_fc2 = nn.Linear(128, 3)

    def forward(self, squares, state, history_from, history_to, masked_positions=None):
        b = squares.shape[0]
        device = squares.device
        # Built by concatenation (no index_put) so ONNX export works.
        piece = torch.zeros(b, 13, 64, device=device)
        piece.scatter_(1, squares.unsqueeze(1), 1.0)                  # 0-12 pieces
        castling = state[:, 1:5].float()[:, :, None].expand(b, 4, 64)
        ep = state[:, 5]
        has_ep = ep < 64
        file_index = (ep % 8).clamp(0, 7)
        cols = torch.arange(64, device=device) % 8
        ep_plane = has_ep[:, None].float() * (cols[None, :] == file_index[:, None]).float()
        half_plane = (state[:, 6].float() / 100.0)[:, None].expand(b, 64)
        hist_f = F.one_hot(history_from.clamp(max=64).long(), 65)[..., :64].float()  # sentinel 64 -> zeros
        hist_t = F.one_hot(history_to.clamp(max=64).long(), 65)[..., :64].float()
        edge = torch.ones(b, 1, 64, device=device)                    # edge-detector constant
        planes = torch.cat(
            (piece, castling, ep_plane[:, None], half_plane[:, None], hist_f, hist_t, edge), dim=1
        )
        x = torch.relu(self.stem_norm(self.stem(planes.view(b, 36, 8, 8))))
        for i, block in enumerate(self.blocks):
            x = block(x)
            if str(i) in self.bebs:
                x = self.bebs[str(i)](x)
        f = self.f_proj(x).flatten(2).transpose(1, 2)                 # (B, 64, 128)
        t = self.t_proj(x).flatten(2).transpose(1, 2)
        base = (f @ t.transpose(1, 2) / 128 ** 0.5).flatten(1)        # from x to logits
        policy = torch.cat((base, self.promotions(f.mean(dim=1))), dim=1)
        v = torch.relu(self.value_fc1(self.value_reduce(x).flatten(1)))
        return {"policy": policy, "value": self.value_fc2(v)}

class ChessCNN(nn.Module):
    """AlphaZero-family trunk over 8x8 planes, sharing the lab's heads and inputs."""

    def __init__(self, r, d):
        super().__init__()
        channels = 13 + 7 + 16  # piece one-hots, broadcast state, history from/to planes
        f = r["cnn_filters"]
        self.stem = nn.Conv2d(channels, f, 5, padding=2)  # knight/king/pawn geometry in one hop
        block_type = ModernBlock if r["cnn_modern"] else ConvBlock
        self.blocks = nn.ModuleList(block_type(f) for _ in range(r["cnn_blocks"]))
        self.ray = FullRayBranch(f) if r["cnn_fullrays"] else RayBranch(f) if r["cnn_rays"] else None
        if self.ray is not None and isinstance(self.ray, FullRayBranch):
            self.ray.mask_leap = r["cnn_knightmask"]
        self.to_tokens = nn.Conv2d(f, d, 1)
        self.summary = nn.Linear(f, d)

    def forward(self, squares, state, history_from, history_to):
        b = squares.shape[0]
        planes = torch.zeros(b, 36, 64, device=squares.device)
        planes.scatter_(1, squares.unsqueeze(1), 1.0)  # channels 0-12 by piece code
        planes[:, 13:20] = (state.float() / torch.tensor(
            [1, 1, 1, 1, 1, 64, 100], device=squares.device
        ))[:, :, None]
        for slot in range(8):
            fr, to = history_from[:, slot], history_to[:, slot]
            on_board = fr < 64
            planes[torch.arange(b, device=squares.device)[on_board], 20 + slot, fr[on_board]] = 1.0
            on_board = to < 64
            planes[torch.arange(b, device=squares.device)[on_board], 28 + slot, to[on_board]] = 1.0
        x = torch.relu(self.stem(planes.view(b, 36, 8, 8)))
        for index, block in enumerate(self.blocks):
            x = block(x)
            if self.ray is not None and index == len(self.blocks) // 2:
                x = self.ray(x)
        square_tokens = self.to_tokens(x).flatten(2).transpose(1, 2)  # (B, 64, d)
        summary = self.summary(x.mean(dim=(2, 3)))
        return torch.cat([summary[:, None], square_tokens], dim=1)  # 65 tokens


def trunk(width_in, hidden, layers, r):
    inproj = nn.Linear(width_in, hidden * (2 if r["gated"] else 1))
    blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers - 1))
    norms = nn.ModuleList(
        nn.LayerNorm(hidden) if r["block_norm"] else nn.Identity() for _ in range(layers - 1)
    )
    return inproj, blocks, norms


class TinyPolicy(nn.Module):
    def __init__(self, r):
        super().__init__()
        self.r = r
        d, hidden, history = r["d_model"], r["hidden"], 8
        self.history = history
        self.use_repetition = False  # match-harness compatibility
        self.flip = r["flip"]  # side-to-move canonicalized inputs/outputs
        self.piece = nn.Embedding(14 if r["aux_masked"] else 13, d)
        if r["rankfile_squares"]:
            self.rank_embed = nn.Embedding(8, d)
            self.file_embed = nn.Embedding(8, d)
        else:
            self.square = nn.Parameter(torch.zeros(64, d))
        if r["shuffle_squares"]:
            gen = torch.Generator().manual_seed(7)
            self.register_buffer("square_perm", torch.randperm(64, generator=gen))
        self.turn = nn.Embedding(2, d)
        self.castling = nn.ModuleList(nn.Embedding(2, d) for _ in range(4))
        self.en_passant = nn.Embedding(65, d)
        self.halfmove = nn.Embedding(HALFMOVE_CAP + 1, d)
        self.history_from = nn.Embedding(65, d)
        self.history_to = nn.Embedding(65, d)
        self.history_position = nn.Parameter(torch.zeros(history, d))
        act = {"relu": nn.ReLU, "gelu": nn.GELU}.get(r["activation"], nn.ReLU)
        self.act = act()
        self.relu2 = r["activation"] == "relu2"
        self.drop = nn.Dropout(r["dropout"])
        width = (65 + history) * d
        self.input_norm = nn.LayerNorm(width) if r["input_norm"] else nn.Identity()
        if r["arch"] == "cnn2":
            self.encoder = ChessCNN2(r)
        elif r["arch"] == "cnn":
            self.encoder = ChessCNN(r, d)
        elif r["arch"] == "transformer":
            if r["attn_bias"]:
                self.encoder = nn.Sequential(
                    *(BiasedBlock(d, r["heads"], r["ffn_ratio"], tie=r["qkv_tie"])
                      for _ in range(r["layers"]))
                )
            else:
                layer = nn.TransformerEncoderLayer(
                    d, r["heads"], dim_feedforward=r["ffn_ratio"] * d,
                    batch_first=True, norm_first=True, dropout=r["dropout"],
                )
                self.encoder = nn.TransformerEncoder(layer, r["layers"])
        elif r["arch"] == "mlp":
            self.inproj, self.blocks, self.norms = trunk(width, hidden, r["layers"], r)
        if r["arch"] == "mlp":
            if r["token_rank"] > 0:
                self.outproj = nn.Sequential(
                    nn.Linear(hidden, r["token_rank"]), nn.Linear(r["token_rank"], 65 * d)
                )
            else:
                self.outproj = nn.Linear(hidden, 65 * d)
        if r["two_tower"]:
            self.v_inproj, self.v_blocks, self.v_norms = trunk(width, hidden, r["layers"], r)
            self.v_out = nn.Linear(hidden, d)
        if r["input_residual"]:
            self.skip_scale = nn.Parameter(torch.tensor(0.1))
        if r["bilinear_head"]:
            self.f_proj = nn.Linear(d, 128)
            self.t_proj = nn.Linear(d, 128)
        elif r["untied_readout"]:
            self.to_square_bank = nn.Parameter(torch.zeros(64, d, 64))
        else:
            self.to_square = nn.Linear(d, 64)
        self.promotions = nn.Linear(d, PROMOTION_MOVES)
        self.value = nn.Linear(d, 3)
        self.aux_plies_head = nn.Linear(d, 8) if r["aux_plies"] else None
        self.aux_material_head = nn.Linear(d, 41) if r["aux_material"] else None
        self.aux_next_head = nn.Linear(d, 4096 + PROMOTION_MOVES) if r["aux_next_move"] else None
        self.aux_masked_head = nn.Linear(d, 13) if r["aux_masked"] else None

    def _act(self, x):
        return torch.relu(x) ** 2 if self.relu2 else self.act(x)

    def _run_trunk(self, flat, inproj, blocks, norms):
        h = inproj(flat)
        if self.r["gated"]:
            a, b = h.chunk(2, dim=-1)
            h = a * torch.sigmoid(b)
        else:
            h = self._act(h)
        h = self.drop(h)
        for block, norm in zip(blocks, norms):
            update = self.drop(self._act(block(norm(h))))
            h = h + update if self.r["residual"] else update
        return h

    def forward(self, squares, state, history_from, history_to, masked_positions=None):
        d = self.r["d_model"]
        if self.r["arch"] == "cnn2":
            return self.encoder(squares, state, history_from, history_to)
        if self.r["arch"] == "cnn":
            tokens = self.encoder(squares, state, history_from, history_to)
            summary = tokens[:, 0]
            base = self.to_square(tokens[:, 1:65]).flatten(1)
            policy = torch.cat((base, self.promotions(summary)), dim=1)
            return {"policy": policy, "value": self.value(summary)}
        if self.r["rankfile_squares"]:
            ranks = torch.arange(64, device=squares.device) // 8
            files = torch.arange(64, device=squares.device) % 8
            square_vec = self.rank_embed(ranks) + self.file_embed(files)
        else:
            square_vec = self.square
        if self.r["shuffle_squares"]:
            square_vec = square_vec[self.square_perm]
        board = self.piece(squares) + square_vec
        vector = self.turn(state[:, 0])
        for index, embedding in enumerate(self.castling):
            vector = vector + embedding(state[:, 1 + index])
        vector = vector + self.en_passant(state[:, 5])
        vector = vector + self.halfmove(state[:, 6].clamp(max=HALFMOVE_CAP))
        moves = self.history_from(history_from) + self.history_to(history_to)
        parts = torch.cat([vector[:, None], board, moves + self.history_position], dim=1)
        if self.r["arch"] == "transformer":
            tokens = self.encoder(parts)[:, :65]
        else:
            flat = self.input_norm(parts.flatten(1))
            h = self._run_trunk(flat, self.inproj, self.blocks, self.norms)
            tokens = self.outproj(h).view(-1, 65, d)
        if self.r["input_residual"]:
            tokens = tokens + self.skip_scale * parts[:, :65]
        summary = tokens[:, 0]
        if self.r["bilinear_head"]:
            f = self.f_proj(tokens[:, 1:65])
            t2 = self.t_proj(tokens[:, 1:65])
            base = (f @ t2.transpose(1, 2) / 128 ** 0.5).flatten(1)
        elif self.r["untied_readout"]:
            base = torch.einsum("bsd,sdt->bst", tokens[:, 1:65], self.to_square_bank).flatten(1)
        else:
            base = self.to_square(tokens[:, 1:65]).flatten(1)
        policy = torch.cat((base, self.promotions(summary)), dim=1)
        if self.r["two_tower"]:
            v_summary = self.v_out(self._run_trunk(flat, self.v_inproj, self.v_blocks, self.v_norms))
        else:
            v_summary = summary
        out = {"policy": policy, "value": self.value(v_summary)}
        if self.aux_plies_head is not None:
            out["aux_plies"] = self.aux_plies_head(summary)
        if self.aux_material_head is not None:
            out["aux_material"] = self.aux_material_head(summary)
        if self.aux_next_head is not None:
            out["aux_next"] = self.aux_next_head(summary)
        if self.aux_masked_head is not None and masked_positions is not None:
            picked = tokens[:, 1:65].gather(
                1, masked_positions[:, :, None].expand(-1, -1, d)
            )
            out["aux_masked"] = self.aux_masked_head(picked)
        return out


def newton_schulz(g, steps=5):
    x = g.bfloat16()
    x = x / (x.norm() + 1e-7)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445 * x + (-4.7750 * a + 2.0315 * (a @ a)) @ x
    return (x.T if transposed else x).to(g.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95):
        super().__init__(params, {"lr": lr, "momentum": momentum})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state.setdefault(p, {})
                buf = state.setdefault("momentum", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                update = newton_schulz(buf) if p.ndim == 2 else buf
                scale = max(1.0, p.shape[0] / p.shape[1]) ** 0.5 if p.ndim == 2 else 1.0
                p.add_(update, alpha=-group["lr"] * scale)


def build_optimizer(model, r):
    if r["optimizer"] == "sgd":
        return [torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=r["wd"])], None
    if r["optimizer"] == "muon2":
        trunk_w = [p for n, p in model.named_parameters()
                   if p.ndim == 2 and any(k in n for k in ("inproj", "outproj", "blocks"))]
        rest = [p for n, p in model.named_parameters()
                if not (p.ndim == 2 and any(k in n for k in ("inproj", "outproj", "blocks")))]
        return [Muon(trunk_w, lr=0.01), torch.optim.AdamW(rest, lr=r["lr"], weight_decay=r["wd"])], None
    if r["optimizer"] == "sfree":
        import schedulefree
        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=r["lr"], weight_decay=r["wd"])
        opt.train()
        return [opt], "sfree"
    if r["embed_lr_scale"] != 1.0:
        embeds, rest = [], []
        for n, p in model.named_parameters():
            (embeds if isinstance(getattr(model, n.split(".")[0], None), (nn.Embedding, nn.ModuleList)) or "embed" in n or n in ("square",) else rest).append(p)
        groups = [
            {"params": embeds, "lr": r["lr"] * r["embed_lr_scale"]},
            {"params": rest, "lr": r["lr"]},
        ]
        return [torch.optim.AdamW(groups, weight_decay=r["wd"], betas=tuple(r["betas"]), eps=r["eps"],
                                  fused=torch.cuda.is_available())], None
    fused_ok = torch.cuda.is_available() and r["arch"] != "cnn"  # fused + channels_last corrupts updates
    return [torch.optim.AdamW(model.parameters(), lr=r["lr"], weight_decay=r["wd"],
                              betas=tuple(r["betas"]), eps=r["eps"],
                              fused=fused_ok)], None


def lr_scale(r, step, total):
    warmup = max(1, int(total * r["warmup"])) if r["warmup"] > 0 else 0
    if step < warmup:
        return step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    if r["schedule"] == "cosine":
        progress = (progress * r["cycles"]) % 1.0 if r["cycles"] > 1 else progress
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        return r["cosine_floor"] + (1 - r["cosine_floor"]) * cos
    if r["schedule"] == "wsd":
        return 1.0 if progress < 0.8 else 1 - (progress - 0.8) / 0.2
    if r["schedule"] == "step":
        return 0.1 if progress > 0.8 else 1.0
    return 1.0


def forward_flops(r):
    """Analytic forward FLOPs per position for the biased-attention transformer (None elsewhere).

    Counts the matmuls that differ between recipes; embeddings and elementwise ops are ignored.
    Training cost is taken as 3x forward (backward is ~2x), the usual convention.
    """
    if r["arch"] != "transformer" or not r["attn_bias"]:
        return None
    d, layers, n = r["d_model"], r["layers"], 65 + 8
    projections = len(set(TIE_ROLES[r["qkv_tie"] or "none"]))
    per_layer = (
        n * d * d * (projections + 1)          # q/k/v projections (deduplicated) + output
        + 2 * n * n * d                        # scores and the weighted sum over values
        + 2 * n * d * d * r["ffn_ratio"]       # feed-forward, both matrices
    )
    if r["bilinear_head"]:
        head = 2 * 65 * d * 128 + 64 * 64 * 128 + d * PROMOTION_MOVES + 3 * d
    else:
        head = 65 * d * 64 + d * PROMOTION_MOVES + 3 * d
    return 2 * (layers * per_layer + head)


def load_shard(path, offset):
    parquet = pq.ParquetFile(path)
    columns = ["game_id", "squares", "state", "target", "result",
               "history_from", "history_to", "ply", "plies_remaining", "future_material"]
    chunks, seen = [], {}
    for batch in parquet.iter_batches(columns=columns):
        ids = batch["game_id"].to_pylist()
        ordinals = np.empty(len(ids), dtype=np.int64)
        for row, gid in enumerate(ids):
            if gid not in seen:
                seen[gid] = len(seen) + offset
            ordinals[row] = seen[gid]
        chunk = {"game_ordinal": ordinals}
        dtypes = {"squares": np.uint8, "state": np.uint8, "history_from": np.uint8,
                  "history_to": np.uint8, "target": np.int16, "result": np.uint8,
                  "ply": np.int16, "plies_remaining": np.int16, "future_material": np.int16}
        for name in columns[1:]:
            arr = batch[name].to_numpy(zero_copy_only=False)
            chunk[name] = (np.stack(arr) if arr.dtype == object else arr).astype(dtypes[name])
        chunks.append(chunk)
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}, len(seen)


def upload_with_retry(api, path_or_fileobj, path_in_repo, attempts=3):
    """Upload to the dataset repo, retrying with backoff; on final failure print a loud
    json line and return False (training must survive a flaky upload, never except:pass).
    LOCAL_CKPT_DIR redirects to local files for offline tests."""
    local_dir = os.environ.get("LOCAL_CKPT_DIR", "")
    if local_dir:
        dest = os.path.join(local_dir, path_in_repo)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if isinstance(path_or_fileobj, (bytes, bytearray)):
            with open(dest, "wb") as f:
                f.write(path_or_fileobj)
        else:
            shutil.copyfile(path_or_fileobj, dest)
        return True
    for attempt in range(attempts):
        try:
            api.upload_file(path_or_fileobj=path_or_fileobj, path_in_repo=path_in_repo,
                            repo_id=DATASET, repo_type="dataset")
            return True
        except Exception as exc:
            if attempt + 1 < attempts:
                time.sleep(5 * 2 ** attempt)
            else:
                print(json.dumps({"upload_failed": path_in_repo, "attempts": attempts,
                                  "error": repr(exc)}), flush=True)
    return False


def main():
    r = {**DEFAULTS, **json.loads(os.environ.get("RECIPE", "{}"))}
    api = HfApi()
    started = time.perf_counter()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.manual_seed(r["seed"])
    smoke_games = int(os.environ.get("SMOKE_GAMES", "0"))

    if r["shard_set"] != "legacy":
        shard_list = SHARD_SETS[r["shard_set"]]
    else:
        shard_list = SHARDS + ([BIG_SHARD] if r["big_shard"] or r["mega_corpus"] else [])
        if r["mega_corpus"]:
            shard_list += ["shards/enriched-chunk3.parquet", "shards/enriched-chunk4.parquet"]
    assert not (r["flip"] and r["aux_next_move"]), "next-move aux not flip-aware"
    offset, parts = 0, []
    local_dir = os.environ.get("LOCAL_SHARDS", "")
    for shard in shard_list:
        path = os.path.join(local_dir, os.path.basename(shard)) if local_dir else hf_hub_download(DATASET, shard, repo_type="dataset")
        merged, games = load_shard(path, offset)
        offset += games
        parts.append(merged)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    # next-move targets: valid only when the following row belongs to the same game
    nxt = np.full(len(data["target"]), -100, dtype=np.int16)
    same = data["game_ordinal"][1:] == data["game_ordinal"][:-1]
    nxt[:-1][same] = data["target"][1:][same]
    data["next_target"] = nxt
    if smoke_games:
        keep = data["game_ordinal"] < smoke_games
        data = {k: v[keep] for k, v in data.items()}
        offset = smoke_games
    rng = np.random.default_rng(r["seed"])
    # A frozen external val shard makes metrics comparable across arms trained on
    # differently-filtered data; the internal 10% game holdout is population-specific.
    val_games = np.zeros(offset, dtype=bool) if r["val_shard"] else rng.random(offset) < 0.1
    mask = val_games[data["game_ordinal"]]
    row_ok = (data["ply"] >= r["ply_min"]) & (data["ply"] < r["ply_max"])
    train_mask = ~mask & row_ok
    if r["subsample"] < 1.0:
        train_mask &= rng.random(len(mask)) < r["subsample"]
    keys = [k for k in data if k != "game_ordinal"]
    train_np = {k: data[k][train_mask].copy() for k in keys}
    if r["val_shard"]:
        vpath = (
            os.path.join(local_dir, os.path.basename(r["val_shard"]))
            if local_dir
            else hf_hub_download(DATASET, r["val_shard"], repo_type="dataset")
        )
        vdata, _ = load_shard(vpath, 0)
        vnxt = np.full(len(vdata["target"]), -100, dtype=np.int16)
        vsame = vdata["game_ordinal"][1:] == vdata["game_ordinal"][:-1]
        vnxt[:-1][vsame] = vdata["target"][1:][vsame]
        vdata["next_target"] = vnxt
        val_np = {k: vdata[k].copy() for k in keys}
    else:
        val_np = {k: data[k][mask].copy() for k in keys}
    if r["flip"]:
        flip_in_place(train_np)
        flip_in_place(val_np)
    train = {k: torch.from_numpy(v).to(device) for k, v in train_np.items()}
    val = {k: torch.from_numpy(v).to(device) for k, v in val_np.items()}
    if r["label_noise"] > 0:
        noise = torch.rand(len(train["target"]), device=device) < r["label_noise"]
        train["target"] = torch.where(
            noise, torch.randint(0, 4096, (len(train["target"]),), device=device), train["target"]
        )
    for part in (train, val):
        if r["history_k"] < 8:
            pad = torch.full_like(part["history_from"][:, : 8 - r["history_k"]], 64)
            part["history_from"] = torch.cat([pad, part["history_from"][:, 8 - r["history_k"]:]], dim=1)
            part["history_to"] = torch.cat([pad, part["history_to"][:, 8 - r["history_k"]:]], dim=1)
        if r["shuffle_history"]:
            perm = torch.randperm(8, device=device)
            part["history_from"] = part["history_from"][:, perm]
            part["history_to"] = part["history_to"][:, perm]
        if r["zero_halfmove"]:
            part["state"] = part["state"].clone()
            part["state"][:, 6] = 0
        if r["zero_castling"]:
            part["state"] = part["state"].clone()
            part["state"][:, 1:5] = 0

    weights = None
    if r["elite_mix"] > 0:
        elite, _ = load_shard(hf_hub_download(DATASET, ELITE_SHARD, repo_type="dataset"), 0)
        elite["next_target"] = np.full(len(elite["target"]), -100, dtype=np.int64)
        elite_t = {k: torch.from_numpy(elite[k]).to(device) for k in keys}
        n_main = len(train["target"])
        train = {k: torch.cat([train[k], elite_t[k]]) for k in keys}
        weights = np.concatenate([
            np.full(n_main, (1 - r["elite_mix"]) / n_main),
            np.full(len(elite_t["target"]), r["elite_mix"] / len(elite_t["target"])),
        ])
    elif r["endgame_oversample"]:
        ply = train["ply"].cpu().numpy()
        weights = np.where(ply >= 60, 2.0, 1.0)
        weights = weights / weights.sum()

    model = TinyPolicy(r).to(device)
    if r["init_from"]:
        prior = torch.load(hf_hub_download(DATASET, r["init_from"], repo_type="dataset"),
                           map_location=device, weights_only=True)
        model.load_state_dict(prior["model"])
    teacher = None
    if r["distill"]:
        saved = torch.load(hf_hub_download(TEACHER_REPO, "57-cloud.pt"), map_location=device, weights_only=True)
        t_conf = {**DEFAULTS, "hidden": saved["config"]["mlp_hidden"], "dropout": 0.0, "compile": False}
        teacher = TinyPolicy(t_conf).to(device)
        teacher.load_state_dict(saved["model"], strict=False)
        teacher.eval()
    optimizers, opt_mode = build_optimizer(model, r)
    # Full resume: restore model+optimizer+step+RNG, then replay the identical rng stream
    # for the data order and skip already-completed steps. LR stays a pure function of step.
    start_step, base_positions, base_flops, nonfinite_skips = 0, 0, 0, 0
    if r["resume_from"]:
        local_ckpt = os.environ.get("LOCAL_CKPT_DIR", "")
        ck_path = (os.path.join(local_ckpt, r["resume_from"]) if local_ckpt
                   else hf_hub_download(DATASET, r["resume_from"], repo_type="dataset"))
        saved = torch.load(ck_path, map_location=device, weights_only=True)
        model.load_state_dict(saved["model"])
        for opt, sd in zip(optimizers, saved["optimizers"]):
            opt.load_state_dict(sd)
        start_step = saved["step"]
        base_positions = saved["positions_seen"]
        base_flops = saved["train_flops"]
        nonfinite_skips = saved.get("nonfinite_skips", 0)
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([s.cpu() for s in saved["cuda_rng"]])
        print(json.dumps({"resumed_from": r["resume_from"], "start_step": start_step,
                          "base_positions": base_positions, "base_flops": base_flops}), flush=True)
    stepper = (
        torch.compile(model, mode=r["compile_mode"] or None)
        if r["compile"] and device.type == "cuda" else model
    )
    ema_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if r["ema"] > 0 else None
    swa_state, swa_count = None, 0

    n_train = len(train["target"])
    total_steps = r["epochs"] * math.ceil(n_train / r["batch"])
    if r["max_steps"]:
        total_steps = r["max_steps"]
    elif r["time_budget_s"]:
        total_steps = int(20 * r["time_budget_s"])   # provisional; replaced by the measured rate
    # calibration window starts after compile/warmup so the measured rate is the steady-state one
    CALIB_FROM, CALIB_TO = 20, 70
    calibrated = not r["time_budget_s"]
    mark_time = None
    train_started = time.perf_counter()
    stop = False
    step = 0
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if device.type != "cpu" else torch.autocast("cpu", enabled=False)
    for epoch in range(r["epochs"]):
        if weights is not None:
            order_np = rng.choice(n_train, size=n_train, replace=True, p=weights)
        else:
            order_np = rng.permutation(n_train)
        order = torch.from_numpy(order_np).to(device)
        for start in range(0, len(order), r["batch"]):
            if step < start_step:  # resume replay: rng stream already consumed identically
                step += 1
                continue
            batch = order[start : start + r["batch"]]
            scale = lr_scale(r, step, total_steps)
            if opt_mode != "sfree":
                for opt in optimizers:
                    for group in opt.param_groups:
                        base = group.get("base_lr")
                        if base is None:
                            group["base_lr"] = base = group["lr"]
                        group["lr"] = base * scale
            squares_in = train["squares"][batch].long()
            if r["input_square_dropout"] > 0:
                drop_mask = torch.rand_like(squares_in, dtype=torch.float) < r["input_square_dropout"]
                squares_in = torch.where(drop_mask, torch.zeros_like(squares_in), squares_in)
            masked_positions = None
            masked_truth = None
            if r["aux_masked"]:
                masked_positions = torch.randint(0, 64, (len(batch), 4), device=device)
                masked_truth = squares_in.gather(1, masked_positions)
                squares_in = squares_in.scatter(1, masked_positions, 13)
            with autocast:
                out = stepper(
                    squares_in, train["state"][batch].long(),
                    train["history_from"][batch].long(), train["history_to"][batch].long(),
                    masked_positions,
                )
                if r["focal"]:
                    ce = torch.nn.functional.cross_entropy(out["policy"], train["target"][batch].long(), reduction="none")
                    p = torch.exp(-ce)
                    loss = ((1 - p) ** 2 * ce).mean()
                else:
                    loss = torch.nn.functional.cross_entropy(
                        out["policy"], train["target"][batch].long(), label_smoothing=r["label_smoothing"]
                    )
                if r["entropy_bonus"] > 0:
                    logp = torch.log_softmax(out["policy"], dim=1)
                    loss = loss - r["entropy_bonus"] * (-(logp.exp() * logp).sum(1).mean())
                if r["value_weight"] > 0:
                    result = train["result"][batch].long()
                    if r["value_mode"] == "bce":
                        target_p = 1.0 - result.float() / 2.0
                        v_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            out["value"][:, 0], target_p, reduction="none")
                    elif r["value_mode"] == "mse":
                        v_loss = (out["value"][:, 0] - (1.0 - result.float())) ** 2
                    elif r["value_mode"] == "smooth":
                        v_loss = torch.nn.functional.cross_entropy(
                            out["value"], result, label_smoothing=0.1, reduction="none")
                    else:
                        v_loss = torch.nn.functional.cross_entropy(out["value"], result, reduction="none")
                    sample_w = torch.ones_like(v_loss)
                    if r["value_ply_weight"]:
                        sample_w = (train["ply"][batch].float() / 60).clamp(0.2, 1.0)
                    if r["value_decisive_only"]:
                        sample_w = sample_w * (result != 1).float()
                    loss = loss + r["value_weight"] * (v_loss * sample_w).mean()
                if r["aux_plies"]:
                    loss = loss + 0.25 * torch.nn.functional.cross_entropy(
                        out["aux_plies"], (train["plies_remaining"][batch].long() // 10).clamp(0, 7))
                if r["aux_material"]:
                    loss = loss + 0.25 * torch.nn.functional.cross_entropy(
                        out["aux_material"], (train["future_material"][batch].long().clamp(-20, 20) + 20))
                if r["aux_next_move"]:
                    loss = loss + 0.25 * torch.nn.functional.cross_entropy(
                        out["aux_next"], train["next_target"][batch].long(), ignore_index=-100)
                if r["aux_masked"]:
                    loss = loss + 0.25 * torch.nn.functional.cross_entropy(
                        out["aux_masked"].flatten(0, 1), masked_truth.long().flatten())
                if teacher is not None:
                    with torch.no_grad():
                        ref = teacher(
                            train["squares"][batch].long(), train["state"][batch].long(),
                            train["history_from"][batch].long(), train["history_to"][batch].long())
                    loss = 0.5 * loss + torch.nn.functional.kl_div(
                        torch.log_softmax(out["policy"] / 2.0, dim=1),
                        torch.softmax(ref["policy"] / 2.0, dim=1), reduction="batchmean")
            if step % 500 == 0 and not bool(torch.isfinite(loss.detach())):
                print(json.dumps({"tripwire": "non-finite loss", "step": step}), flush=True)
                raise RuntimeError(f"DIVERGED: non-finite loss at step {step}/{total_steps}")
            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            if r["grad_noise"] > 0:
                with torch.no_grad():
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.add_(torch.randn_like(p.grad), alpha=r["grad_noise"])
            skip_step = False
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), r["clip"] if r["clip"] > 0 else float("inf"))
            if not torch.isfinite(norm):  # one bad gradient must not poison every parameter
                skip_step = True
                nonfinite_skips += 1
                print(json.dumps({"nonfinite_grad_skip": True, "step": step,
                                  "total_skips": nonfinite_skips}), flush=True)
                for opt in optimizers:
                    opt.zero_grad()
            if not skip_step:
                for opt in optimizers:
                    opt.step()
            if ema_state is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema_state[k].mul_(r["ema"]).add_(v.float(), alpha=1 - r["ema"])
            if (r["save_ckpt"] and r["ckpt_every_frac"] == 0 and not smoke_games
                    and step > 0 and step % max(1, total_steps // 4) == 0):
                mark = round(100 * step / total_steps)
                torch.save({"sweep_recipe": r, "config": {}, "model": model.state_dict()}, "/tmp/partial.pt")
                upload_with_retry(api, "/tmp/partial.pt", f"results3/{r['id']}.partial{mark}.pt")
            if r["swa"] and step >= int(total_steps * 0.8):
                with torch.no_grad():
                    if swa_state is None:
                        swa_state = {k: v.detach().clone().float() for k, v in model.state_dict().items()}
                        swa_count = 1
                    else:
                        swa_count += 1
                        for k, v in model.state_dict().items():
                            swa_state[k].add_((v.float() - swa_state[k]) / swa_count)
            step += 1
            ckpt_every = int(total_steps * r["ckpt_every_frac"])
            if ckpt_every > 0 and not smoke_games and step % ckpt_every == 0 and step < total_steps:
                fwd = forward_flops(r)
                torch.save({
                    "sweep_recipe": r, "config": {}, "model": model.state_dict(),
                    "optimizers": [opt.state_dict() for opt in optimizers],
                    "step": step,
                    "positions_seen": base_positions + (step - start_step) * r["batch"],
                    "train_flops": base_flops + (3 * fwd * (step - start_step) * r["batch"] if fwd else 0),
                    "nonfinite_skips": nonfinite_skips,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                }, "/tmp/resume.pt")
                mark = round(100 * step / total_steps)
                # per-mark file so nothing is ever overwritten, plus a rolling latest for resume
                upload_with_retry(api, "/tmp/resume.pt", f"results3/{r['id']}.partial{mark}.pt")
                upload_with_retry(api, "/tmp/resume.pt", f"results3/{r['id']}.resume.pt")
            if not calibrated:
                if step == CALIB_FROM:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    mark_time = time.perf_counter()
                elif step == CALIB_TO:
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    now = time.perf_counter()
                    rate = (CALIB_TO - CALIB_FROM) / max(1e-6, now - mark_time)
                    remaining = r["time_budget_s"] - (now - train_started)
                    total_steps = CALIB_TO + max(1, int(rate * remaining))
                    calibrated = True
                    print(json.dumps({"steps_per_second": round(rate, 2),
                                      "calibrated_total_steps": total_steps}), flush=True)
            if step >= total_steps:
                stop = True
                break
        print(json.dumps({"epoch": epoch + 1, "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
        if stop:
            break

    train_elapsed = time.perf_counter() - train_started
    if opt_mode == "sfree":
        optimizers[0].eval()
    final_state = ema_state or swa_state
    if final_state is not None:
        model.load_state_dict({k: v.to(dtype) for (k, v), dtype in
                               zip(final_state.items(), [p.dtype for p in model.state_dict().values()])})
    model.eval()
    correct = value_correct = 0
    loss_sum = 0.0
    count = len(val["target"])
    with torch.no_grad():
        for start in range(0, count, 8192):
            b = slice(start, start + 8192)
            out = model(val["squares"][b].long(), val["state"][b].long(),
                        val["history_from"][b].long(), val["history_to"][b].long())
            loss_sum += torch.nn.functional.cross_entropy(out["policy"], val["target"][b].long(), reduction="sum").item()
            correct += (out["policy"].argmax(1) == val["target"][b].long()).sum().item()
            if r["value_mode"] in ("ce", "smooth"):
                value_correct += (out["value"].argmax(1) == val["result"][b].long()).sum().item()
            else:
                predicted = (out["value"][:, 0] > 0.5 if r["value_mode"] == "mse"
                             else out["value"][:, 0] > 0)
                value_correct += (predicted == (val["result"][b].long() == 0)).sum().item()
    metrics = {
        "id": r["id"], "recipe": r,
        "parameters": sum(p.numel() for p in model.parameters()),
        "val_loss": loss_sum / count, "val_top1": correct / count, "value_top1": value_correct / count,
        "train_positions": n_train, "steps": step,
        "positions_seen": base_positions + (step - start_step) * r["batch"],
        "nonfinite_skips": nonfinite_skips,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "train_seconds": round(train_elapsed, 1),
    }
    if r["resume_from"]:
        metrics["resumed_from_step"] = start_step
    fwd = forward_flops(r)
    if fwd is not None:
        metrics["forward_flops_per_position"] = fwd
        metrics["train_flops"] = base_flops + 3 * fwd * (step - start_step) * r["batch"]
    print(json.dumps(metrics), flush=True)
    if smoke_games:
        return
    upload_with_retry(api, json.dumps(metrics, indent=2).encode(), f"results3/{r['id']}.json")
    if r["save_ckpt"]:
        torch.save({"sweep_recipe": r, "config": {}, "model": model.state_dict()}, "/tmp/ckpt.pt")
        upload_with_retry(api, "/tmp/ckpt.pt", f"results3/{r['id']}.pt")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            rid = {**DEFAULTS, **json.loads(os.environ.get("RECIPE", "{}"))}["id"]
            upload_with_retry(HfApi(), json.dumps({"id": rid, "error": error}).encode(),
                              f"results3/{rid}.json", attempts=1)
        except Exception:
            pass
        raise
