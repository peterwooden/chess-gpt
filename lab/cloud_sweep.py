# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "pyarrow", "huggingface_hub", "schedulefree"]
# ///
"""Fleet-2 parameterized trainer: recipe via RECIPE env JSON, metrics pushed to the shard repo."""

import json
import math
import os
import time
import traceback

import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn

DATASET = "peterwooden/chess-gpt-lab-shards"
SHARDS = ["shards/enriched-240k.parquet", "shards/enriched-fresh240k.parquet"]
ELITE_SHARD = "shards/winner2000-enriched.parquet"
TEACHER_REPO = "peterwooden/chess-gpt-cloud-57"
HALFMOVE_CAP, PROMOTION_MOVES = 100, 176

DEFAULTS = {
    "id": "control", "optimizer": "adamw", "lr": 1.2e-3, "wd": 0.01, "betas": [0.9, 0.999],
    "clip": 0.0, "schedule": "const", "warmup": 0.0, "batch": 1024, "epochs": 2,
    "ema": 0.0, "activation": "relu", "residual": False, "block_norm": False,
    "layers": 2, "hidden": 1152, "dropout": 0.1, "label_smoothing": 0.0,
    "value_weight": 1.0, "value_ply_weight": False, "value_decisive_only": False,
    "aux_plies": False, "aux_material": False, "elite_mix": 0.0,
    "endgame_oversample": False, "curriculum": "none", "distill": False,
    "compile": False, "seed": 20260730, "save_ckpt": False,
}


class TinyPolicy(nn.Module):
    """Superset of lab.model.TinyPolicy's MLP branch; core module names stay compatible."""

    def __init__(self, r):
        super().__init__()
        d, hidden, history = 128, r["hidden"], 8
        self.history = history
        self.use_repetition = False  # match-harness compatibility
        self.piece = nn.Embedding(13, d)
        self.square = nn.Parameter(torch.zeros(64, d))
        self.turn = nn.Embedding(2, d)
        self.castling = nn.ModuleList(nn.Embedding(2, d) for _ in range(4))
        self.en_passant = nn.Embedding(65, d)
        self.halfmove = nn.Embedding(HALFMOVE_CAP + 1, d)
        self.history_from = nn.Embedding(65, d)
        self.history_to = nn.Embedding(65, d)
        self.history_position = nn.Parameter(torch.zeros(history, d))
        act = {"relu": nn.ReLU, "gelu": nn.GELU}.get(r["activation"])
        self.relu2 = r["activation"] == "relu2"
        self.residual = r["residual"]
        width = (65 + history) * d
        self.inproj = nn.Linear(width, hidden)
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(r["layers"] - 1):
            self.blocks.append(nn.Linear(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden) if r["block_norm"] else nn.Identity())
        self.act = act() if act else nn.ReLU()
        self.drop = nn.Dropout(r["dropout"])
        self.outproj = nn.Linear(hidden, 65 * d)
        self.to_square = nn.Linear(d, 64)
        self.promotions = nn.Linear(d, PROMOTION_MOVES)
        self.value = nn.Linear(d, 3)
        self.aux_plies_head = nn.Linear(d, 8) if r["aux_plies"] else None
        self.aux_material_head = nn.Linear(d, 41) if r["aux_material"] else None

    def _act(self, x):
        return torch.relu(x) ** 2 if self.relu2 else self.act(x)

    def forward(self, squares, state, history_from, history_to):
        board = self.piece(squares) + self.square
        vector = self.turn(state[:, 0])
        for index, embedding in enumerate(self.castling):
            vector = vector + embedding(state[:, 1 + index])
        vector = vector + self.en_passant(state[:, 5])
        vector = vector + self.halfmove(state[:, 6].clamp(max=HALFMOVE_CAP))
        moves = self.history_from(history_from) + self.history_to(history_to)
        flat = torch.cat([vector[:, None], board, moves + self.history_position], dim=1).flatten(1)
        h = self.drop(self._act(self.inproj(flat)))
        for block, norm in zip(self.blocks, self.norms):
            update = self.drop(self._act(block(norm(h))))
            h = h + update if self.residual else update
        tokens = self.outproj(h).view(-1, 65, 128)
        summary = tokens[:, 0]
        policy = torch.cat((self.to_square(tokens[:, 1:65]).flatten(1), self.promotions(summary)), dim=1)
        out = {"policy": policy, "value": self.value(summary)}
        if self.aux_plies_head is not None:
            out["aux_plies"] = self.aux_plies_head(summary)
        if self.aux_material_head is not None:
            out["aux_material"] = self.aux_material_head(summary)
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


class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), wd=0.01):
        super().__init__(params, {"lr": lr, "betas": betas, "wd": wd})

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            b1, b2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state.setdefault(p, {})
                buf = state.setdefault("momentum", torch.zeros_like(p))
                p.mul_(1 - group["lr"] * group["wd"])
                p.add_(torch.sign(buf.mul(b1).add(p.grad, alpha=1 - b1)), alpha=-group["lr"])
                buf.mul_(b2).add_(p.grad, alpha=1 - b2)


def build_optimizer(model, r):
    if r["optimizer"] == "muon":
        hidden = [p for n, p in model.named_parameters() if p.ndim == 2 and "proj" in n or "blocks" in n]
        rest = [p for n, p in model.named_parameters() if not (p.ndim == 2 and ("proj" in n or "blocks" in n))]
        return [Muon(hidden, lr=0.02), torch.optim.AdamW(rest, lr=r["lr"], weight_decay=r["wd"])], None
    if r["optimizer"] == "lion":
        return [Lion(model.parameters(), lr=r["lr"] / 4, wd=r["wd"] * 10)], None
    if r["optimizer"] == "sfree":
        import schedulefree

        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=r["lr"], weight_decay=r["wd"])
        opt.train()
        return [opt], "sfree"
    return [
        torch.optim.AdamW(
            model.parameters(), lr=r["lr"], weight_decay=r["wd"],
            betas=tuple(r["betas"]), fused=torch.cuda.is_available(),
        )
    ], None


def lr_scale(r, step, total):
    warmup = max(1, int(total * r["warmup"])) if r["warmup"] > 0 else 0
    if step < warmup:
        return step / warmup
    if r["schedule"] == "cosine":
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    if r["schedule"] == "wsd":
        decay_start = int(total * 0.8)
        return 1.0 if step < decay_start else 1 - (step - decay_start) / max(1, total - decay_start)
    if r["schedule"] == "step":
        return 0.1 if step > total * 0.8 else 1.0
    return 1.0


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
        for name in columns[1:]:
            arr = batch[name].to_numpy(zero_copy_only=False)
            chunk[name] = (np.stack(arr) if arr.dtype == object else arr).astype(np.int64)
        chunks.append(chunk)
    return {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}, len(seen)


def main():
    r = {**DEFAULTS, **json.loads(os.environ.get("RECIPE", "{}"))}
    api = HfApi()
    started = time.perf_counter()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    torch.manual_seed(r["seed"])
    smoke_games = int(os.environ.get("SMOKE_GAMES", "0"))

    offset, parts = 0, []
    for shard in SHARDS:
        merged, games = load_shard(hf_hub_download(DATASET, shard, repo_type="dataset"), offset)
        offset += games
        parts.append(merged)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    if smoke_games:
        keep = data["game_ordinal"] < smoke_games
        data = {k: v[keep] for k, v in data.items()}
        offset = smoke_games
    rng = np.random.default_rng(r["seed"])
    val_games = rng.random(offset) < 0.1
    mask = val_games[data["game_ordinal"]]
    keys = [k for k in data if k != "game_ordinal"]
    train = {k: torch.from_numpy(data[k][~mask]).to(device) for k in keys}
    val = {k: torch.from_numpy(data[k][mask]).to(device) for k in keys}
    train_order_meta = {"total": data["ply"][~mask] + data["plies_remaining"][~mask]}
    if r["elite_mix"] > 0:
        elite, _ = load_shard(hf_hub_download(DATASET, ELITE_SHARD, repo_type="dataset"), 0)
        elite_t = {k: torch.from_numpy(elite[k]).to(device) for k in keys}
        train = {k: torch.cat([train[k], elite_t[k]]) for k in keys}
        n_main = len(train["target"]) - len(elite_t["target"])
        weights = np.concatenate([
            np.full(n_main, (1 - r["elite_mix"]) / n_main),
            np.full(len(elite_t["target"]), r["elite_mix"] / len(elite_t["target"])),
        ])
    elif r["endgame_oversample"]:
        ply = data["ply"][~mask]
        weights = np.where(ply >= 60, 2.0, 1.0)
        weights = weights / weights.sum()
    else:
        weights = None

    model = TinyPolicy(r).to(device)
    teacher = None
    if r["distill"]:
        saved = torch.load(hf_hub_download(TEACHER_REPO, "57-cloud.pt"), map_location=device, weights_only=True)
        t_conf = {**DEFAULTS, "hidden": saved["config"]["mlp_hidden"], "dropout": 0.0}
        teacher = TinyPolicy(t_conf).to(device)
        teacher.load_state_dict(saved["model"])
        teacher.eval()
    optimizers, opt_mode = build_optimizer(model, r)
    stepper = torch.compile(model) if r["compile"] else model
    ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()} if r["ema"] > 0 else None

    n_train = len(train["target"])
    total_steps = r["epochs"] * math.ceil(n_train / r["batch"])
    step = 0
    autocast = torch.autocast(device.type, dtype=torch.bfloat16) if device.type != "cpu" else torch.autocast("cpu", enabled=False)
    for epoch in range(r["epochs"]):
        if r["curriculum"] != "none":
            noise = rng.random(n_train) * 40
            order_np = np.argsort(train_order_meta["total"] + noise)
            if r["curriculum"] == "long_first":
                order_np = order_np[::-1].copy()
        elif weights is not None:
            order_np = rng.choice(len(train["target"]), size=n_train, replace=True, p=weights)
        else:
            order_np = rng.permutation(n_train)
        order = torch.from_numpy(order_np).to(device)
        for start in range(0, len(order), r["batch"]):
            batch = order[start : start + r["batch"]]
            scale = lr_scale(r, step, total_steps)
            if opt_mode != "sfree":
                for opt in optimizers:
                    for group in opt.param_groups:
                        base = group.get("base_lr")
                        if base is None:
                            group["base_lr"] = base = group["lr"]
                        group["lr"] = base * scale
            with autocast:
                out = stepper(
                    train["squares"][batch].long(), train["state"][batch].long(),
                    train["history_from"][batch].long(), train["history_to"][batch].long(),
                )
                loss = torch.nn.functional.cross_entropy(
                    out["policy"], train["target"][batch], label_smoothing=r["label_smoothing"]
                )
                if r["value_weight"] > 0:
                    v_loss = torch.nn.functional.cross_entropy(
                        out["value"], train["result"][batch], reduction="none"
                    )
                    sample_w = torch.ones_like(v_loss)
                    if r["value_ply_weight"]:
                        sample_w = (train["ply"][batch].float() / 60).clamp(0.2, 1.0)
                    if r["value_decisive_only"]:
                        sample_w = sample_w * (train["result"][batch] != 1).float()
                    loss = loss + r["value_weight"] * (v_loss * sample_w).mean()
                if r["aux_plies"]:
                    loss = loss + 0.25 * torch.nn.functional.cross_entropy(
                        out["aux_plies"], (train["plies_remaining"][batch] // 10).clamp(0, 7)
                    )
                if r["aux_material"]:
                    loss = loss + 0.25 * torch.nn.functional.cross_entropy(
                        out["aux_material"], (train["future_material"][batch].clamp(-20, 20) + 20)
                    )
                if teacher is not None:
                    with torch.no_grad():
                        ref = teacher(
                            train["squares"][batch].long(), train["state"][batch].long(),
                            train["history_from"][batch].long(), train["history_to"][batch].long(),
                        )
                    loss = 0.5 * loss + torch.nn.functional.kl_div(
                        torch.log_softmax(out["policy"] / 2.0, dim=1),
                        torch.softmax(ref["policy"] / 2.0, dim=1), reduction="batchmean",
                    )
            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            if r["clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), r["clip"])
            for opt in optimizers:
                opt.step()
            if ema_state is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        ema_state[k].mul_(r["ema"]).add_(v, alpha=1 - r["ema"])
            step += 1
        print(json.dumps({"epoch": epoch + 1, "elapsed": round(time.perf_counter() - started, 1)}), flush=True)

    if opt_mode == "sfree":
        optimizers[0].eval()
    if ema_state is not None:
        model.load_state_dict(ema_state)
    model.eval()
    correct = value_correct = 0
    loss_sum = 0.0
    count = len(val["target"])
    with torch.no_grad():
        for start in range(0, count, 8192):
            b = slice(start, start + 8192)
            out = model(
                val["squares"][b].long(), val["state"][b].long(),
                val["history_from"][b].long(), val["history_to"][b].long(),
            )
            loss_sum += torch.nn.functional.cross_entropy(out["policy"], val["target"][b], reduction="sum").item()
            correct += (out["policy"].argmax(1) == val["target"][b]).sum().item()
            value_correct += (out["value"].argmax(1) == val["result"][b]).sum().item()
    metrics = {
        "id": r["id"], "recipe": r,
        "parameters": sum(p.numel() for p in model.parameters()),
        "val_loss": loss_sum / count, "val_top1": correct / count, "value_top1": value_correct / count,
        "train_positions": n_train, "steps": step,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(metrics), flush=True)
    if smoke_games:
        return
    api.upload_file(
        path_or_fileobj=json.dumps(metrics, indent=2).encode(),
        path_in_repo=f"results2/{r['id']}.json", repo_id=DATASET, repo_type="dataset",
    )
    if r["save_ckpt"]:
        torch.save({"sweep_recipe": r, "config": {
            "d_model": 128, "layers": r["layers"], "heads": 4, "ff_mult": 4, "dropout": r["dropout"],
            "arch": "mlp", "mlp_hidden": r["hidden"], "per_square_readout": True, "moe": False,
            "geo_bias": False, "piece_value_init": False, "state_token": True,
            "material_feature": False, "repetition_feature": False, "history": 8,
            "aux_material": False, "aux_plies": False,
        }, "model": model.state_dict()}, "/tmp/ckpt.pt")
        api.upload_file(path_or_fileobj="/tmp/ckpt.pt", path_in_repo=f"results2/{r['id']}.pt",
                        repo_id=DATASET, repo_type="dataset")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            rid = {**DEFAULTS, **json.loads(os.environ.get("RECIPE", "{}"))}["id"]
            HfApi().upload_file(
                path_or_fileobj=json.dumps({"id": rid, "error": error}).encode(),
                path_in_repo=f"results2/{rid}.json", repo_id=DATASET, repo_type="dataset",
            )
        except Exception:
            pass
        raise
