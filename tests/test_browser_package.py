from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

from chess_gpt.browser_package import export_ngram_package


def test_export_ngram_package_is_complete_and_self_verifying(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.json.gz"
    state = {
        "format_version": 1,
        "model_type": "san_backoff_ngram",
        "order": 1,
        "metadata": {"experiment_id": "tiny-ngram"},
        "ngrams": {"1": {"e4": [["e5", 2]]}},
        "side_counts": {"0": [["e4", 3]], "1": [["e5", 2]]},
    }
    canonical = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
    with gzip.GzipFile(filename=checkpoint, mode="wb", mtime=0) as file:
        file.write(canonical)

    entrypoint = tmp_path / "entry.js"
    entrypoint.write_text("export async function loadPackage() {}\n")
    output = tmp_path / "browser"

    manifest = export_ngram_package(
        checkpoint=checkpoint,
        entrypoint=entrypoint,
        output=output,
    )

    assert manifest["schema"] == "chess-gpt-package-v1"
    assert manifest["name"] == "tiny-ngram"
    assert (output / "model.json").read_bytes() == canonical
    for descriptor in [manifest["entrypoint"], *manifest["artifacts"].values()]:
        payload = (output / descriptor["path"]).read_bytes()
        assert descriptor["bytes"] == len(payload)
        assert descriptor["sha256"] == hashlib.sha256(payload).hexdigest()
    assert sum(path.stat().st_size for path in output.iterdir()) <= 100_000_000


def test_export_ngram_package_rejects_wrong_checkpoint_type(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.json.gz"
    with gzip.open(checkpoint, "wt") as file:
        json.dump({"format_version": 1, "model_type": "something_else"}, file)
    entrypoint = tmp_path / "entry.js"
    entrypoint.write_text("export async function loadPackage() {}\n")

    try:
        export_ngram_package(
            checkpoint=checkpoint,
            entrypoint=entrypoint,
            output=tmp_path / "browser",
        )
    except ValueError as error:
        assert "SAN backoff n-gram" in str(error)
    else:
        raise AssertionError("wrong checkpoint type was accepted")
