"""Rebuild a published package-v1 with a new entrypoint, reusing its artifacts.

The checkpoint that produced a published package is not always at hand, and
re-exporting ONNX to change only the adapter would risk a different model digest
for no reason. This takes the published `browser/` directory as the source of
truth for `model.onnx` and `vocabulary.json`, bundles a new entrypoint against
them, and writes a fresh manifest. The model artifacts are copied byte for byte,
so the new package is the same weights behind different search telemetry.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from chess_gpt.snapshot_package import (
    PACKAGE_LIMIT_BYTES,
    PACKAGE_SCHEMA,
    _bundle_entrypoint,
    _descriptor,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="published browser/ directory")
    parser.add_argument("--entrypoint-source", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--decode", required=True, help="manifest config.decode value")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_manifest = json.loads((args.source / "manifest.json").read_text())
    if source_manifest["schema"] != PACKAGE_SCHEMA:
        raise SystemExit(f"unsupported source schema: {source_manifest['schema']}")

    args.output.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for name, descriptor in source_manifest["artifacts"].items():
        payload = (args.source / descriptor["path"]).read_bytes()
        if len(payload) != descriptor["bytes"]:
            raise SystemExit(f"{descriptor['path']} does not match its published byte length")
        (args.output / descriptor["path"]).write_bytes(payload)
        rebuilt = _descriptor(descriptor["path"], payload)
        if rebuilt["sha256"] != descriptor["sha256"]:
            raise SystemExit(f"{descriptor['path']} does not match its published digest")
        artifacts[name] = rebuilt

    # esbuild reads a bare relative path as a bare package import, not a file.
    entrypoint_bytes = _bundle_entrypoint(
        args.entrypoint_source.resolve(), args.output / "entry.js"
    )
    manifest = {
        "schema": PACKAGE_SCHEMA,
        "name": args.name,
        "entrypoint": _descriptor("entry.js", entrypoint_bytes),
        "artifacts": artifacts,
        "config": {**source_manifest.get("config", {}), "decode": args.decode},
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    (args.output / "manifest.json").write_bytes(manifest_bytes)
    total = (
        len(manifest_bytes)
        + len(entrypoint_bytes)
        + sum(descriptor["bytes"] for descriptor in artifacts.values())
    )
    if total > PACKAGE_LIMIT_BYTES:
        raise SystemExit(f"canonical package is {total:,} bytes, over the limit")
    print(json.dumps({"canonical_package_bytes": total, "output": str(args.output)}))


if __name__ == "__main__":
    main()
