"""Shared helpers for the evaluation scripts: dataset loading and gold-evidence matching."""

import json
import re
from dataclasses import dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
DATASET = EVAL_DIR / "questions.json"
RESULTS_DIR = EVAL_DIR / "results"


def results_dir(split: str) -> Path:
    return RESULTS_DIR / split

_RANGE = re.compile(r"^(?P<path>[^:#]+):(?P<start>\d+)-(?P<end>\d+)$")


@dataclass(frozen=True)
class Gold:
    """A piece of evidence that answers a question: a file line range or a history ref."""

    ref: str

    @property
    def is_history(self) -> bool:
        return bool(re.match(r"^(PR#|issue#|discussion#|release:|commit:)", self.ref, re.IGNORECASE))

    def matches(self, ref: str) -> bool:
        """True if a retrieved/cited ref covers this gold evidence (line ranges must overlap)."""
        if self.is_history:
            return ref.lower() == self.ref.lower()
        gold, other = _RANGE.match(self.ref), _RANGE.match(ref)
        if gold is None:  # whole-file gold
            return ref.split(":", 1)[0] == self.ref
        if other is None or other["path"] != gold["path"]:
            return False
        return int(other["start"]) <= int(gold["end"]) and int(gold["start"]) <= int(other["end"])


def load_questions(repo: str | None = None, ids: list[str] | None = None, split: str | None = None) -> list[dict]:
    questions = json.loads(DATASET.read_text())["questions"]
    if split:
        questions = [q for q in questions if q.get("split") == split]
    if repo:
        questions = [q for q in questions if q["repo"] == repo]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    return questions


def gold_items(question: dict) -> list[Gold]:
    return [Gold(ref) for ref in question.get("gold_evidence", [])]


def any_match(gold: list[Gold], refs: list[str]) -> bool:
    return any(g.matches(r) for g in gold for r in refs)
