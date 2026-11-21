"""Hand-written search questions, checked against a manifest before they are used.

Unlike the golden set, these are written by hand and committed: what a person would type is
the thing being measured, and a generator would copy the descriptions' wording. Each expected
model must exist in the manifest, so a question cannot silently go stale.
"""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

import yaml

Category = Literal["described", "column", "jargon", "undescribed", "unanswerable"]
CATEGORIES: tuple[str, ...] = get_args(Category)
SEARCH_SET_VERSION = 1

WORD = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "the and for are was were how what which where who when does did each every with from "
    "that this there have has had our can you get see one any all per about into while do we "
    "is a an of in on to by at it be as or i my me us they them their its".split()
)


class SearchSetError(ValueError):
    pass


@dataclass(frozen=True)
class SearchQuestion:
    id: str
    category: Category
    question: str
    expected: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class SearchSet:
    project: str
    questions: tuple[SearchQuestion, ...]
    sha: str


def load_search_set(path: Path) -> SearchSet:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or data.get("version") != SEARCH_SET_VERSION:
        raise SearchSetError(f"{path}: expected version {SEARCH_SET_VERSION}")
    problems: list[str] = []
    questions: list[SearchQuestion] = []
    seen: set[str] = set()
    for number, item in enumerate(data.get("questions") or [], start=1):
        qid = str(item.get("id", f"#{number}"))
        category = item.get("category")
        expected = tuple(item.get("expected") or ())
        if qid in seen:
            problems.append(f"{qid}: duplicate id")
        seen.add(qid)
        if category not in CATEGORIES:
            problems.append(f"{qid}: category {category!r} is not one of {', '.join(CATEGORIES)}")
        if not str(item.get("question", "")).strip():
            problems.append(f"{qid}: no question")
        if (category == "unanswerable") != (not expected):
            problems.append(f"{qid}: only unanswerable questions may have no expected models")
        if len(set(expected)) != len(expected):
            problems.append(f"{qid}: an expected model is listed twice")
        questions.append(
            SearchQuestion(
                qid, category, str(item.get("question", "")), expected, item.get("notes", "")
            )
        )
    if not questions:
        problems.append("no questions")
    if problems:
        raise SearchSetError(f"{path}:\n" + "\n".join(f"  - {p}" for p in problems))
    return SearchSet(str(data["project"]), tuple(questions), hashlib.sha256(raw).hexdigest()[:12])


def model_ids(manifest: Mapping, project: str) -> dict[str, str]:
    """Model name -> unique_id for one project. Names are unique within a dbt project."""
    return {
        node["name"]: unique_id
        for unique_id, node in manifest["nodes"].items()
        if node["resource_type"] == "model" and node["package_name"] == project
    }


def check_against_manifest(search_set: SearchSet, manifest: Mapping) -> dict[str, tuple[str, ...]]:
    """Question id -> expected unique_ids; raises if any expected model is missing."""
    ids = model_ids(manifest, search_set.project)
    if not ids:
        raise SearchSetError(f"the manifest has no models in project {search_set.project!r}")
    missing = [
        f"{question.id}: no model named {name!r}"
        for question in search_set.questions
        for name in question.expected
        if name not in ids
    ]
    if missing:
        raise SearchSetError("stale search set:\n" + "\n".join(f"  - {m}" for m in missing))
    return {q.id: tuple(ids[name] for name in q.expected) for q in search_set.questions}


def words(text: str) -> set[str]:
    return {w for w in WORD.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS}


def description_overlap(question: SearchQuestion, manifest: Mapping, project: str) -> float | None:
    """Share of the question's words found in the best expected model's name and description.

    High overlap means keyword search can find it, so it tests little that embeddings add.
    """
    asked = words(question.question)
    if not question.expected or not asked:
        return None
    ids = model_ids(manifest, project)
    best = 0.0
    for name in question.expected:
        node = manifest["nodes"][ids[name]]
        found = words(f"{node['name'].replace('_', ' ')} {node.get('description', '')}")
        best = max(best, len(asked & found) / len(asked))
    return best
