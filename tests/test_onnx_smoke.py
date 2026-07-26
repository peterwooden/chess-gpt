from __future__ import annotations

import hashlib
from pathlib import Path

import onnx

from chess_gpt.onnx_smoke import build_smoke_model, export_onnx_smoke_package


def test_smoke_model_is_valid_and_minimal() -> None:
    model_bytes = build_smoke_model()
    model = onnx.load_model_from_string(model_bytes)

    onnx.checker.check_model(model)
    assert len(model.graph.node) == 1
    assert model.graph.node[0].op_type == "Add"
    assert len(model_bytes) < 1_000


def test_exported_smoke_package_is_complete_and_self_verifying(tmp_path: Path) -> None:
    entrypoint = Path("adapters/onnx-smoke/entry.js")
    output = tmp_path / "browser"

    manifest = export_onnx_smoke_package(entrypoint=entrypoint, output=output)

    assert manifest["schema"] == "chess-gpt-package-v1"
    assert manifest["name"] == "onnx-runtime-smoke"
    for descriptor in [manifest["entrypoint"], *manifest["artifacts"].values()]:
        payload = (output / descriptor["path"]).read_bytes()
        assert descriptor["bytes"] == len(payload)
        assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
