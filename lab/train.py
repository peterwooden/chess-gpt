"""Train one lab policy variant and report honest numbers."""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from lab.data import load_slice, validation_mask
from lab.model import TinyPolicy

MODEL_FLAGS = (
    "d_model", "layers", "heads", "ff_mult", "dropout", "arch", "mlp_hidden",
    "per_square_readout", "moe", "geo_bias", "piece_value_init", "state_token",
    "material_feature", "repetition_feature", "history", "aux_material", "aux_plies",
)


def batch_inputs(tensors: dict[str, torch.Tensor], batch, config: dict) -> dict:
    inputs = {
        "squares": tensors["squares"][batch].long(),
        "state": tensors["state"][batch].long(),
    }
    if config["history"]:
        inputs["history_from"] = tensors["history_from"][batch].long()
        inputs["history_to"] = tensors["history_to"][batch].long()
    if config["repetition_feature"]:
        inputs["repetition"] = tensors["repetition"][batch].long()
    return inputs


def losses(output: dict, tensors: dict, batch, args) -> torch.Tensor:
    loss = torch.nn.functional.cross_entropy(
        output["policy"], tensors["target"][batch], label_smoothing=args.label_smoothing
    ) * args.policy_weight
    if args.value_weight > 0:
        loss = loss + args.value_weight * torch.nn.functional.cross_entropy(
            output["value"], tensors["result"][batch]
        )
    if args.aux_material:
        target = (tensors["future_material"][batch].clamp(-20, 20) + 20).long()
        loss = loss + args.aux_weight * torch.nn.functional.cross_entropy(
            output["aux_material"], target
        )
    if args.aux_plies:
        target = (tensors["plies_remaining"][batch] // 10).clamp(0, 7).long()
        loss = loss + args.aux_weight * torch.nn.functional.cross_entropy(
            output["aux_plies"], target
        )
    return loss


def evaluate(model, tensors, config, args, batch_size: int = 4096) -> dict[str, float]:
    model.eval()
    count = len(tensors["target"])
    policy_loss = 0.0
    policy_correct = value_correct = 0
    with torch.no_grad():
        for start in range(0, count, batch_size):
            batch = slice(start, start + batch_size)
            output = model(**batch_inputs(tensors, batch, config))
            moves = tensors["target"][batch]
            policy_loss += torch.nn.functional.cross_entropy(
                output["policy"], moves, reduction="sum"
            ).item()
            policy_correct += (output["policy"].argmax(dim=1) == moves).sum().item()
            if args.value_weight > 0:
                value_correct += (
                    output["value"].argmax(dim=1) == tensors["result"][batch]
                ).sum().item()
    model.train()
    metrics = {"loss": policy_loss / count, "top1": policy_correct / count}
    if args.value_weight > 0:
        metrics["value_top1"] = value_correct / count
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--games", type=int, default=10_000)
    parser.add_argument("--split", choices=("position", "game"), default="game")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1.2e-3)
    parser.add_argument("--schedule", choices=("const", "cosine"), default="const")
    parser.add_argument("--warmup-frac", type=float, default=0.0)
    # model flags
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ff-mult", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--arch", choices=("transformer", "mlp"), default="transformer")
    parser.add_argument("--mlp-hidden", type=int, default=512)
    parser.add_argument("--per-square-readout", action="store_true")
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--geo-bias", action="store_true")
    parser.add_argument("--piece-value-init", action="store_true")
    parser.add_argument("--no-state-token", dest="state_token", action="store_false")
    parser.add_argument("--material-feature", action="store_true")
    parser.add_argument("--repetition-feature", action="store_true")
    parser.add_argument("--history", type=int, default=0)
    # objectives
    parser.add_argument("--policy-weight", type=float, default=1.0)
    parser.add_argument("--value-weight", type=float, default=0.0)
    parser.add_argument("--aux-material", action="store_true")
    parser.add_argument("--aux-plies", action="store_true")
    parser.add_argument("--aux-weight", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    # stages
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--distill-weight", type=float, default=1.0)
    # speed
    parser.add_argument("--fused-adam", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    data = load_slice(args.data, args.games)
    validation = validation_mask(data, args.split, args.validation_fraction, args.seed)
    arrays = {"squares": data.squares, "state": data.state, "target": data.target}
    if data.result is not None:
        arrays["result"] = data.result
    arrays.update(data.extras)
    needed = {"result"} if args.value_weight > 0 else set()
    if args.history:
        needed |= {"history_from", "history_to"}
    if args.repetition_feature:
        needed |= {"repetition"}
    if args.aux_material:
        needed |= {"future_material"}
    if args.aux_plies:
        needed |= {"plies_remaining"}
    missing = needed - arrays.keys()
    if missing:
        raise SystemExit(f"data shard lacks required columns: {sorted(missing)}")
    train = {k: torch.from_numpy(v[~validation]).to(device) for k, v in arrays.items()}
    val = {k: torch.from_numpy(v[validation]).to(device) for k, v in arrays.items()}
    if args.history:  # keep only the last K prior moves
        for part in (train, val):
            part["history_from"] = part["history_from"][:, -args.history :]
            part["history_to"] = part["history_to"][:, -args.history :]

    config = {flag: getattr(args, flag) for flag in MODEL_FLAGS}
    model = TinyPolicy(**config).to(device)
    if args.init_checkpoint:
        saved = torch.load(args.init_checkpoint, weights_only=True)
        model.load_state_dict(saved["model"] if "config" in saved else saved, strict=False)
    teacher = None
    if args.teacher:
        saved = torch.load(args.teacher, weights_only=True)
        teacher = TinyPolicy(**saved["config"]).to(device)
        teacher.load_state_dict(saved["model"])
        teacher.eval()
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, fused=args.fused_adam
    )
    total_steps = args.epochs * math.ceil(len(train["target"]) / args.batch_size)

    def lr_lambda(step: int) -> float:
        warmup = max(1, int(total_steps * args.warmup_frac))
        if step < warmup and args.warmup_frac > 0:
            return step / warmup
        if args.schedule == "cosine":
            progress = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    stepper = torch.compile(model) if args.compile else model
    autocast = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if args.precision == "bf16"
        else nullcontext()
    )

    steps = 0
    generator = np.random.default_rng(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        order = torch.from_numpy(generator.permutation(len(train["target"]))).to(device)
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            with autocast:
                output = stepper(**batch_inputs(train, batch, config))
                loss = losses(output, train, batch, args)
                if teacher is not None:
                    with torch.no_grad():
                        reference = teacher(**batch_inputs(train, batch, config))
                    loss = loss + args.distill_weight * torch.nn.functional.kl_div(
                        torch.log_softmax(output["policy"] / 2.0, dim=1),
                        torch.softmax(reference["policy"] / 2.0, dim=1),
                        reduction="batchmean",
                    )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            steps += 1
        # Deadline insurance: every epoch leaves a packageable checkpoint behind.
        torch.save(
            {"config": config, "model": model.state_dict()}, args.output.with_suffix(".pt")
        )
        print(
            json.dumps(
                {"epoch": epoch + 1, "elapsed": round(time.perf_counter() - started, 1)}
            ),
            flush=True,
        )

    metrics = {
        "games": data.games,
        "train_positions": int(len(train["target"])),
        "validation_positions": int(len(val["target"])),
        "parameters": parameters,
        "config": config,
        "recipe": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "schedule": args.schedule,
            "warmup_frac": args.warmup_frac, "policy_weight": args.policy_weight,
            "value_weight": args.value_weight, "aux_weight": args.aux_weight,
            "label_smoothing": args.label_smoothing, "split": args.split,
            "fused_adam": args.fused_adam, "compiled": args.compile,
            "precision": args.precision, "seed": args.seed,
            "init_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            "teacher": str(args.teacher) if args.teacher else None,
        },
        "steps": steps,
        # Evaluation always runs the uncompiled model in fp32 so metrics stay comparable.
        "train": evaluate(model, train, config, args),
        "validation": evaluate(model, val, config, args),
        "wall_seconds": round(time.perf_counter() - started, 1),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    torch.save({"config": config, "model": model.state_dict()}, args.output.with_suffix(".pt"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
