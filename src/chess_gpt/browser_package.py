"""Export reproducible ChessGPT browser packages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

PACKAGE_SCHEMA = "chess-gpt-package-v1"
PACKAGE_LIMIT_BYTES = 100_000_000


def _descriptor(path: str, payload: bytes) -> dict[str, str | int]:
    return {
        "path": path,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def export_ngram_package(
    *,
    checkpoint: Path,
    entrypoint: Path,
    output: Path,
) -> dict[str, Any]:
    """Turn the canonical state inside an n-gram checkpoint into package v1."""
    with gzip.open(checkpoint, "rb") as file:
        model_bytes = file.read()
    try:
        state = json.loads(model_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("checkpoint does not contain valid JSON") from error
    if not isinstance(state, dict) or state.get("model_type") != "san_backoff_ngram":
        raise ValueError("checkpoint is not a SAN backoff n-gram model")

    entrypoint_bytes = entrypoint.read_bytes()
    metadata = state.get("metadata")
    experiment_id = metadata.get("experiment_id") if isinstance(metadata, dict) else None
    name = experiment_id if isinstance(experiment_id, str) and experiment_id else checkpoint.stem
    manifest: dict[str, Any] = {
        "schema": PACKAGE_SCHEMA,
        "name": name,
        "entrypoint": _descriptor("entry.js", entrypoint_bytes),
        "artifacts": {"model": _descriptor("model.json", model_bytes)},
        "config": {},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    package_bytes = len(manifest_bytes) + len(entrypoint_bytes) + len(model_bytes)
    if package_bytes > PACKAGE_LIMIT_BYTES:
        raise ValueError(
            f"canonical package is {package_bytes:,} bytes, "
            f"over the {PACKAGE_LIMIT_BYTES:,}-byte limit"
        )

    output.mkdir(parents=True, exist_ok=True)
    (output / "entry.js").write_bytes(entrypoint_bytes)
    (output / "model.json").write_bytes(model_bytes)
    (output / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = export_ngram_package(
        checkpoint=args.checkpoint,
        entrypoint=args.entrypoint,
        output=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
