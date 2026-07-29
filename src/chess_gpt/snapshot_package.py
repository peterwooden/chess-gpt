"""Export trained snapshot policies as self-contained tournament packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch import nn

from chess_gpt.snapshot_model import PROMOTION_UCI_MOVES, ModelConfig, SnapshotPolicy

PACKAGE_SCHEMA = "chess-gpt-package-v1"
PACKAGE_LIMIT_BYTES = 100_000_000


class _OnnxInputs(nn.Module):
    def __init__(self, model: SnapshotPolicy) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, squares: torch.Tensor, state: torch.Tensor, phase: torch.Tensor
    ) -> torch.Tensor:
        return self.model(squares.long(), state.long(), phase.long())


def _descriptor(path: str, payload: bytes) -> dict[str, str | int]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _bundle_entrypoint(source: Path, destination: Path) -> bytes:
    esbuild = Path("site/node_modules/.bin/esbuild")
    chess_js = Path("site/node_modules/chess.js/dist/esm/chess.js").resolve()
    if not esbuild.is_file():
        raise FileNotFoundError("run npm install in site/ before packaging (esbuild is missing)")
    if not chess_js.is_file():
        raise FileNotFoundError("run npm install in site/ before packaging (chess.js is missing)")
    subprocess.run(
        [
            str(esbuild),
            str(source),
            "--bundle",
            "--format=esm",
            "--platform=browser",
            "--target=es2022",
            "--minify",
            "--legal-comments=inline",
            f"--alias:chess.js={chess_js}",
            f"--outfile={destination}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return destination.read_bytes()


def export_snapshot_package(
    *, checkpoint: Path, entrypoint_source: Path, output: Path
) -> dict[str, Any]:
    """Export ONNX, bundled chess rules, vocabulary, and a verified manifest."""
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if raw.get("model_type") != "board_snapshot_policy":
        raise ValueError("checkpoint is not a board snapshot policy")
    config = ModelConfig(**raw["model_config"])
    model = SnapshotPolicy(config)
    model.load_state_dict(raw["state_dict"])
    model.eval()
    wrapper = _OnnxInputs(model)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "model.onnx"
    torch.backends.mha.set_fastpath_enabled(False)
    torch.onnx.export(
        wrapper,
        (
            torch.zeros((1, 64), dtype=torch.int32),
            torch.zeros((1, 7), dtype=torch.int32),
            torch.zeros((1,), dtype=torch.int32),
        ),
        model_path,
        input_names=["squares", "state", "phase"],
        output_names=["logits"],
        dynamic_axes={
            "squares": {0: "batch"},
            "state": {0: "batch"},
            "phase": {0: "batch"},
            "logits": {0: "batch"},
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
    (output / "vocabulary.json").write_bytes(vocabulary_bytes)
    entrypoint_bytes = _bundle_entrypoint(entrypoint_source, output / "entry.js")
    manifest: dict[str, Any] = {
        "schema": PACKAGE_SCHEMA,
        "name": raw["train_config"]["experiment_id"],
        "entrypoint": _descriptor("entry.js", entrypoint_bytes),
        "artifacts": {
            "model": _descriptor("model.onnx", model_bytes),
            "vocabulary": _descriptor("vocabulary.json", vocabulary_bytes),
        },
        "config": {"architecture": config.architecture},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (output / "manifest.json").write_bytes(manifest_bytes)
    package_bytes = len(manifest_bytes) + len(entrypoint_bytes) + len(model_bytes) + len(
        vocabulary_bytes
    )
    if package_bytes > PACKAGE_LIMIT_BYTES:
        raise ValueError(f"canonical package is {package_bytes:,} bytes, over the limit")
    manifest["canonical_package_bytes"] = package_bytes
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--entrypoint-source",
        type=Path,
        default=Path("adapters/board-policy/entry.source.js"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export_snapshot_package(
        checkpoint=args.checkpoint,
        entrypoint_source=args.entrypoint_source,
        output=args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
