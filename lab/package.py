"""Export a lab checkpoint as a self-contained tournament package-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from chess_gpt.snapshot_model import PROMOTION_UCI_MOVES
from chess_gpt.snapshot_package import (
    PACKAGE_LIMIT_BYTES,
    PACKAGE_SCHEMA,
    _bundle_entrypoint,
    _descriptor,
)
from lab.model import TinyPolicy


class _OnnxInputs(nn.Module):
    def __init__(self, model: TinyPolicy) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        squares: torch.Tensor,
        state: torch.Tensor,
        history_from: torch.Tensor,
        history_to: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.model(
            squares.long(), state.long(), history_from.long(), history_to.long()
        )
        return output["policy"], output["value"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--entrypoint-source",
        type=Path,
        default=Path("adapters/lab-value-search/entry.source.js"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = saved["config"]
    model = TinyPolicy(**config)
    model.load_state_dict(saved["model"])
    model.eval()
    history = config.get("history", 0)
    if history != 8:
        raise SystemExit("this adapter expects a history-8 checkpoint")

    args.output.mkdir(parents=True, exist_ok=True)
    model_path = args.output / "model.onnx"
    torch.backends.mha.set_fastpath_enabled(False)
    torch.onnx.export(
        _OnnxInputs(model),
        (
            torch.zeros((1, 64), dtype=torch.int32),
            torch.zeros((1, 7), dtype=torch.int32),
            torch.full((1, history), 64, dtype=torch.int32),
            torch.full((1, history), 64, dtype=torch.int32),
        ),
        model_path,
        input_names=["squares", "state", "history_from", "history_to"],
        output_names=["policy", "value"],
        dynamic_axes={
            name: {0: "batch"}
            for name in ("squares", "state", "history_from", "history_to", "policy", "value")
        },
        opset_version=17,
        dynamo=False,
    )
    model_bytes = model_path.read_bytes()
    vocabulary_bytes = (
        json.dumps(
            {"promotion_uci_moves": PROMOTION_UCI_MOVES},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    (args.output / "vocabulary.json").write_bytes(vocabulary_bytes)
    entrypoint_bytes = _bundle_entrypoint(args.entrypoint_source, args.output / "entry.js")
    manifest: dict[str, Any] = {
        "schema": PACKAGE_SCHEMA,
        "name": args.name,
        "entrypoint": _descriptor("entry.js", entrypoint_bytes),
        "artifacts": {
            "model": _descriptor("model.onnx", model_bytes),
            "vocabulary": _descriptor("vocabulary.json", vocabulary_bytes),
        },
        "config": {
            "architecture": "lab-tinypolicy",
            "model_config": config,
            "decode": "value-search-1ply-contempt-0.15",
        },
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (args.output / "manifest.json").write_bytes(manifest_bytes)
    total = sum(
        len(part) for part in (manifest_bytes, entrypoint_bytes, model_bytes, vocabulary_bytes)
    )
    if total > PACKAGE_LIMIT_BYTES:
        raise SystemExit(f"canonical package is {total:,} bytes, over the limit")
    print(json.dumps({"canonical_package_bytes": total, "output": str(args.output)}))


if __name__ == "__main__":
    main()
