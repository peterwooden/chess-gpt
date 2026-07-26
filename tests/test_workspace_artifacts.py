from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_learning_site_internal_routes_resolve() -> None:
    documents = [
        ROOT / "site/app/page.tsx",
        ROOT / "site/app/chapter-1/data-splits/lesson-client.tsx",
        ROOT / "site/app/glossary/page.tsx",
    ]
    assert all(document.is_file() for document in documents)

    for document in documents:
        for link in re.findall(r'href=["\']([^"\']+)["\']', document.read_text()):
            if link.startswith(("http://", "https://", "#")):
                continue
            route = link.split("?", 1)[0].split("#", 1)[0]
            assert route.startswith("/"), f"{document}: unexpected relative route {link}"
            route_page = ROOT / "site/app" / route.removeprefix("/") / "page.tsx"
            if route == "/":
                route_page = ROOT / "site/app/page.tsx"
            assert route_page.is_file(), f"{document}: missing route {link}"


def test_tournament_dataset_is_frozen_by_month_and_checksum() -> None:
    with (ROOT / "data/dataset.toml").open("rb") as file:
        manifest = tomllib.load(file)

    assert manifest["dataset"]["status"] == "frozen"
    files = manifest["files"]
    assert [(item["month"], item["split"]) for item in files] == [
        ("2026-01", "train"),
        ("2026-02", "train"),
        ("2026-03", "train"),
        ("2026-04", "validation"),
    ]
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in files)


def test_tournament_rules_are_five_bold_labelled_bullets() -> None:
    rules = (ROOT / "docs/TOURNAMENT_RULES.md").read_text()
    summary = rules.split("## Technical appendix", 1)[0]
    bullets = [line for line in summary.splitlines() if line.startswith("- ")]

    assert len(bullets) == 5
    assert all(line.startswith("- **") for line in bullets)


def test_tournament_rules_include_unified_package_contract() -> None:
    rules = (ROOT / "docs/TOURNAMENT_RULES.md").read_text()
    for required_term in (
        "public Hugging Face",
        "chess-gpt-package-v1",
        "100,000,000",
        "loadPackage",
        "legalMoves",
        "dedicated Web Worker",
        "immediate game loss",
    ):
        assert required_term in rules


def test_first_experiment_records_the_observed_result() -> None:
    path = ROOT / "experiments/0000-local-autodiff-smoke.toml"
    with path.open("rb") as file:
        manifest = tomllib.load(file)

    assert manifest["experiment"]["id"] == path.stem
    assert manifest["experiment"]["status"] == "completed"
    assert manifest["result"]["accepted"] is True


def test_first_playable_baseline_records_a_passing_result() -> None:
    path = ROOT / "experiments/0001-basic-san-ngram.toml"
    with path.open("rb") as file:
        manifest = tomllib.load(file)

    result = manifest["result"]
    assert manifest["experiment"]["id"] == path.stem
    assert manifest["experiment"]["status"] == "completed"
    assert result["validation_legal_move_rate"] == 1.0
    assert (
        result["validation_top1_accuracy"]
        > result["validation_deterministic_fallback_top1_accuracy"]
    )
    assert result["canonical_learned_state_bytes"] <= 100_000_000
    assert result["functional_san_cli"] is True
    assert result["acceptance_passed"] is True
    assert manifest["publication"]["revision"] == "fecf413cfe0e5dab427c4cec7a78aafa4410aa65"
    assert manifest["publication"]["browser_artifact_url"].endswith("/model.json.gz")
    assert manifest["publication"]["package_v1_revision"] == (
        "bea221167728c33f0a5df54051cd27717cae6586"
    )
    assert manifest["publication"]["package_v1_canonical_bytes"] == 6_689_698
    assert manifest["package_validation"]["code_revision"] == (
        "987533c34125a89a4b8f2bed0535b2f517ff953c"
    )
    assert manifest["package_validation"]["actual_training_flops"] == 0
    assert manifest["package_validation"]["external_cost_usd"] == 0.0


def test_curriculum_tracks_prediction_driven_progress() -> None:
    curriculum = (ROOT / "CURRICULUM.md").read_text()

    assert curriculum.count("## Chapter ") == 10
    assert "- [x] Placement diagnostic completed on the learning site" in curriculum
    assert (ROOT / "learning-records/0002-placement-diagnostic.md").is_file()
    assert "hill climbing" in curriculum.lower()
    assert "prediction" in curriculum.lower()
    assert "five hours per week" in curriculum.lower()
    assert curriculum.count("### RL extension ") == 4
    assert "Part III §15.4" in curriculum
    assert "Part III §17.1" in curriculum
    assert "- [x] Adaptive Chapter 1 missions generated" in curriculum
    assert (ROOT / "docs/CHAPTER_1_PLAN.md").is_file()
    assert (ROOT / "site/app/chapter-1/data-splits/lesson-client.tsx").is_file()
    assert (ROOT / "site/app/glossary/page.tsx").is_file()
    assert curriculum.count("**Further reading**") == 14
    assert curriculum.count("- **Primary:**") == 14
    assert curriculum.count("- **Secondary:**") == 14

    reading_lines = [
        line
        for line in curriculum.splitlines()
        if line.startswith(("- **Primary:**", "- **Secondary:**"))
    ]
    assert len(reading_lines) == 28
    assert all(
        1 <= len(re.findall(r"\[[^]]+\]\([^)]+\)", line)) <= 3
        for line in reading_lines
    )
