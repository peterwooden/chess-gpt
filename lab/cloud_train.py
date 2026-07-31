# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "pyarrow", "huggingface_hub"]
# ///
"""Standalone cloud trainer: the lab MLP composite, state-dict compatible with lab.model.TinyPolicy."""

import json
import os
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn

DATASET = "peterwooden/chess-gpt-lab-shards"
SHARDS = ["shards/enriched-240k.parquet", "shards/enriched-fresh240k.parquet"]
OUTPUT_REPO = "peterwooden/chess-gpt-cloud-57"
CONFIG = {
    "d_model": 128, "layers": 2, "heads": 4, "ff_mult": 4, "dropout": 0.1,
    "arch": "mlp", "mlp_hidden": 1152, "per_square_readout": True, "moe": False,
    "geo_bias": False, "piece_value_init": False, "state_token": True,
    "material_feature": False, "repetition_feature": False, "history": 8,
    "aux_material": False, "aux_plies": False,
}
EPOCHS, BATCH, LR, VALUE_WEIGHT, SEED = 6, 1024, 1.2e-3, 1.0, 20260730

HALFMOVE_CAP = 100
BASE_MOVES, PROMOTION_MOVES = 64 * 64, 176
MOVE_CLASSES = BASE_MOVES + PROMOTION_MOVES


class TinyPolicy(nn.Module):
    """MLP branch of lab.model.TinyPolicy with identical module names."""

    def __init__(self, d_model, layers, mlp_hidden, dropout, history, **_):
        super().__init__()
        self.history = history
        self.piece = nn.Embedding(13, d_model)
        self.square = nn.Parameter(torch.zeros(64, d_model))
        self.turn = nn.Embedding(2, d_model)
        self.castling = nn.ModuleList(nn.Embedding(2, d_model) for _ in range(4))
        self.en_passant = nn.Embedding(65, d_model)
        self.halfmove = nn.Embedding(HALFMOVE_CAP + 1, d_model)
        self.history_from = nn.Embedding(65, d_model)
        self.history_to = nn.Embedding(65, d_model)
        self.history_position = nn.Parameter(torch.zeros(history, d_model))
        width = (65 + history) * d_model
        blocks = []
        for _ in range(layers):
            blocks += [nn.Linear(width, mlp_hidden), nn.ReLU(), nn.Dropout(dropout)]
            width = mlp_hidden
        blocks.append(nn.Linear(width, 65 * d_model))
        self.mlp = nn.Sequential(*blocks)
        self.to_square = nn.Linear(d_model, 64)
        self.promotions = nn.Linear(d_model, PROMOTION_MOVES)
        self.value = nn.Linear(d_model, 3)

    def forward(self, squares, state, history_from, history_to):
        board = self.piece(squares) + self.square
        vector = self.turn(state[:, 0])
        for index, embedding in enumerate(self.castling):
            vector = vector + embedding(state[:, 1 + index])
        vector = vector + self.en_passant(state[:, 5])
        vector = vector + self.halfmove(state[:, 6].clamp(max=HALFMOVE_CAP))
        moves = self.history_from(history_from) + self.history_to(history_to)
        parts = [vector[:, None], board, moves + self.history_position]
        tokens = self.mlp(torch.cat(parts, dim=1).flatten(1)).view(-1, 65, board.shape[-1])
        summary = tokens[:, 0]
        base = self.to_square(tokens[:, 1:65]).flatten(1)
        policy = torch.cat((base, self.promotions(summary)), dim=1)
        return {"policy": policy, "value": self.value(summary)}


def load_shard(path, ordinal_offset):
    parquet = pq.ParquetFile(path)
    columns = ["game_id", "squares", "state", "target", "result", "history_from", "history_to"]
    chunks, seen = [], {}
    for batch in parquet.iter_batches(columns=columns):
        ids = batch["game_id"].to_pylist()
        ordinals = np.empty(len(ids), dtype=np.int64)
        for row, gid in enumerate(ids):
            if gid not in seen:
                seen[gid] = len(seen) + ordinal_offset
            ordinals[row] = seen[gid]
        chunks.append({
            "squares": np.stack(batch["squares"].to_numpy(zero_copy_only=False)),
            "state": np.stack(batch["state"].to_numpy(zero_copy_only=False)),
            "target": batch["target"].to_numpy().astype(np.int64),
            "result": batch["result"].to_numpy().astype(np.int64),
            "history_from": np.stack(batch["history_from"].to_numpy(zero_copy_only=False)).astype(np.int64),
            "history_to": np.stack(batch["history_to"].to_numpy(zero_copy_only=False)).astype(np.int64),
            "game_ordinal": ordinals,
        })
    merged = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    return merged, len(seen)


def evaluate(model, tensors, device):
    model.eval()
    count = len(tensors["target"])
    correct = value_correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, count, 8192):
            batch = slice(start, start + 8192)
            out = model(
                tensors["squares"][batch].long(), tensors["state"][batch].long(),
                tensors["history_from"][batch].long(), tensors["history_to"][batch].long(),
            )
            moves = tensors["target"][batch]
            loss_sum += torch.nn.functional.cross_entropy(out["policy"], moves, reduction="sum").item()
            correct += (out["policy"].argmax(1) == moves).sum().item()
            value_correct += (out["value"].argmax(1) == tensors["result"][batch]).sum().item()
    model.train()
    return {"loss": loss_sum / count, "top1": correct / count, "value_top1": value_correct / count}


def main():
    started = time.perf_counter()
    device = torch.device("cuda")
    torch.manual_seed(SEED)
    offset = 0
    parts = []
    for shard in SHARDS:
        path = hf_hub_download(DATASET, shard, repo_type="dataset")
        merged, games = load_shard(path, offset)
        offset += games
        parts.append(merged)
        print(json.dumps({"loaded": shard, "games": games}), flush=True)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    rng = np.random.default_rng(SEED)
    validation_games = rng.random(offset) < 0.1
    mask = validation_games[data["game_ordinal"]]
    keys = ["squares", "state", "target", "result", "history_from", "history_to"]
    train = {k: torch.from_numpy(data[k][~mask]).to(device) for k in keys}
    val = {k: torch.from_numpy(data[k][mask]).to(device) for k in keys}
    print(json.dumps({"train_positions": len(train["target"]), "val_positions": len(val["target"])}), flush=True)

    model = TinyPolicy(**CONFIG).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
    generator = np.random.default_rng(SEED)
    for epoch in range(EPOCHS):
        order = torch.from_numpy(generator.permutation(len(train["target"]))).to(device)
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    train["squares"][batch].long(), train["state"][batch].long(),
                    train["history_from"][batch].long(), train["history_to"][batch].long(),
                )
                loss = torch.nn.functional.cross_entropy(out["policy"], train["target"][batch])
                loss = loss + VALUE_WEIGHT * torch.nn.functional.cross_entropy(
                    out["value"], train["result"][batch]
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(json.dumps({"epoch": epoch + 1, "elapsed": round(time.perf_counter() - started, 1)}), flush=True)

    metrics = {
        "config": CONFIG,
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "value_weight": VALUE_WEIGHT, "seed": SEED,
        "train": evaluate(model, train, device),
        "validation": evaluate(model, val, device),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    torch.save({"config": CONFIG, "model": model.state_dict()}, "/tmp/57-cloud.pt")
    with open("/tmp/57-cloud.json", "w") as handle:
        json.dump(metrics, handle, indent=2)
    api = HfApi()
    api.create_repo(OUTPUT_REPO, private=True, exist_ok=True)
    api.upload_file(path_or_fileobj="/tmp/57-cloud.pt", path_in_repo="57-cloud.pt", repo_id=OUTPUT_REPO)
    api.upload_file(path_or_fileobj="/tmp/57-cloud.json", path_in_repo="57-cloud.json", repo_id=OUTPUT_REPO)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
