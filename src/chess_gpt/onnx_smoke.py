"""Build the minimal ONNX package used to verify the browser runtime path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

PACKAGE_SCHEMA = "chess-gpt-package-v1"


def _descriptor(path: str, payload: bytes) -> dict[str, str | int]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def build_smoke_model() -> bytes:
    """Return a one-operation model whose output equals its scalar input."""
    graph = helper.make_graph(
        nodes=[helper.make_node("Add", ["input", "zero"], ["output"])],
        name="chess-gpt-onnx-runtime-smoke",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        initializer=[numpy_helper.from_array(np.array([0], dtype=np.float32), name="zero")],
    )
    model = helper.make_model(
        graph,
        producer_name="chess-gpt",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    # Keep the fixture compatible with older runtimes; it uses no newer IR features.
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString(deterministic=True)


def export_onnx_smoke_package(*, entrypoint: Path, output: Path) -> dict[str, Any]:
    """Create a complete package-v1 fixture that must initialize ONNX Runtime."""
    entrypoint_bytes = entrypoint.read_bytes()
    model_bytes = build_smoke_model()
    manifest: dict[str, Any] = {
        "schema": PACKAGE_SCHEMA,
        "name": "onnx-runtime-smoke",
        "entrypoint": _descriptor("entry.js", entrypoint_bytes),
        "artifacts": {"model": _descriptor("model.onnx", model_bytes)},
        "config": {},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()

    output.mkdir(parents=True, exist_ok=True)
    (output / "entry.js").write_bytes(entrypoint_bytes)
    (output / "model.onnx").write_bytes(model_bytes)
    (output / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entrypoint",
        type=Path,
        default=Path("adapters/onnx-smoke/entry.js"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_onnx_smoke_package(entrypoint=args.entrypoint, output=args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
