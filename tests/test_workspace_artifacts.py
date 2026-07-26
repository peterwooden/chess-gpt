from __future__ import annotations

import re
import tomllib
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag not in {"a", "link"}:
            return
        for name, value in attrs:
            if name == "href" and value is not None:
                self.links.append(value)


def test_local_lesson_links_resolve() -> None:
    documents = [*ROOT.glob("lessons/*.html"), *ROOT.glob("reference/*.html")]
    assert documents

    for document in documents:
        parser = LinkCollector()
        parser.feed(document.read_text())
        for link in parser.links:
            if link.startswith(("http://", "https://", "#")):
                continue
            target = link.split("#", 1)[0]
            assert (document.parent / target).resolve().exists(), f"{document}: missing {link}"


def test_candidate_dataset_is_not_mistaken_for_a_freeze() -> None:
    with (ROOT / "data/dataset-candidate.toml").open("rb") as file:
        manifest = tomllib.load(file)

    assert manifest["dataset"]["status"] == "candidate-not-frozen"
    assert manifest["compatibility"]["compatible"] is False


def test_tournament_rules_are_five_bold_labelled_bullets() -> None:
    rules = (ROOT / "docs/TOURNAMENT_RULES_DRAFT.md").read_text().splitlines()
    bullets = [line for line in rules if line.startswith("- ")]

    assert len(bullets) == 5
    assert all(line.startswith("- **") for line in bullets)


def test_arena_hugging_face_requirements_stay_brief_and_complete() -> None:
    requirements = (ROOT / "docs/ARENA_HUGGING_FACE_REQUIREMENTS.md").read_text()
    bullets = [line for line in requirements.splitlines() if line.startswith("- ")]

    assert 5 <= len(bullets) <= 8
    assert all(line.startswith("- **") for line in bullets)
    for required_term in (
        "public Hugging Face",
        "ONNX",
        "SAN",
        "input_ids",
        "SHA-256",
        "150 MB",
        "40-character-commit-sha",
    ):
        assert required_term in requirements


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
    assert (ROOT / "lessons/0001-honest-chess-data-splits.html").is_file()
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
