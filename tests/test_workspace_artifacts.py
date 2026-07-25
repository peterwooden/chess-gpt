from __future__ import annotations

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


def test_first_experiment_records_the_observed_result() -> None:
    path = ROOT / "experiments/0000-local-autodiff-smoke.toml"
    with path.open("rb") as file:
        manifest = tomllib.load(file)

    assert manifest["experiment"]["id"] == path.stem
    assert manifest["experiment"]["status"] == "completed"
    assert manifest["result"]["accepted"] is True
