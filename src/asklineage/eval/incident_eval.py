"""Score the incident search methods on hand-written questions (step 3c, ADR 0015).

One table: hit@k and MRR per method, overall and by category. Expected incidents come from
the questions file, checked against the corpus so a question can't point at a missing or
non-incident document. Scoring reuses the search eval's.
"""

import hashlib
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, get_args

import psycopg
import yaml

from asklineage.corpus.docs import Corpus
from asklineage.embed.base import EmbeddingProvider
from asklineage.eval.search_eval import QuestionScore, score_question, summarize
from asklineage.incidents.search import Method, find_incidents

Category = Literal["named", "symptom", "loose", "decoy", "none"]
CATEGORIES: tuple[str, ...] = get_args(Category)
SET_VERSION = 1
K = 10
HIT_AT = (1, 3, 5)


class IncidentSetError(ValueError):
    pass


@dataclass(frozen=True)
class IncidentQuestion:
    id: str
    category: Category
    question: str
    expected: tuple[str, ...]


@dataclass(frozen=True)
class IncidentSet:
    project: str
    questions: tuple[IncidentQuestion, ...]
    sha: str


def load_incident_set(path: Path) -> IncidentSet:
    raw = path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict) or data.get("version") != SET_VERSION:
        raise IncidentSetError(f"{path}: expected version {SET_VERSION}")
    problems: list[str] = []
    questions: list[IncidentQuestion] = []
    for number, item in enumerate(data.get("questions") or [], start=1):
        qid = str(item.get("id", f"#{number}"))
        category = item.get("category")
        expected = tuple(item.get("expected") or ())
        text = str(item.get("question", "")).strip()
        if qid in {q.id for q in questions}:
            problems.append(f"{qid}: duplicate id")
        if category not in CATEGORIES:
            problems.append(f"{qid}: category {category!r} is not one of {', '.join(CATEGORIES)}")
        if not text:
            problems.append(f"{qid}: no question")
        if (category == "none") != (not expected):
            problems.append(f"{qid}: only category none may have no expected incidents")
        questions.append(IncidentQuestion(qid, category, text, expected))
    if not questions:
        problems.append("no questions")
    if problems:
        raise IncidentSetError(f"{path}:\n" + "\n".join(f"  - {p}" for p in problems))
    return IncidentSet(str(data["project"]), tuple(questions), hashlib.sha256(raw).hexdigest()[:12])


def check_against_corpus(incident_set: IncidentSet, corpus: Corpus) -> None:
    if incident_set.project != corpus.project:
        raise IncidentSetError(
            f"questions are for {incident_set.project!r}, the corpus is {corpus.project!r}"
        )
    incidents = {doc.id for doc in corpus.documents if doc.type == "incident"}
    missing = [
        f"{q.id}: no incident {doc_id!r} in the corpus"
        for q in incident_set.questions
        for doc_id in q.expected
        if doc_id not in incidents
    ]
    if missing:
        raise IncidentSetError(
            "stale incident questions:\n" + "\n".join(f"  - {m}" for m in missing)
        )


def run_method(
    conn: psycopg.Connection,
    incident_set: IncidentSet,
    method: Method,
    embedder: EmbeddingProvider | None,
) -> list[QuestionScore]:
    scores = []
    for q in incident_set.questions:
        started = time.perf_counter()
        hits = find_incidents(conn, q.question, method, embedder, project=incident_set.project, k=K)
        latency_ms = round((time.perf_counter() - started) * 1000)
        returned = [hit.doc_id for hit in hits]
        top = hits[0].score if hits else None
        scores.append(score_question(q.id, q.category, q.expected, returned, top, latency_ms))
    return scores


def run_incident_eval(
    conn: psycopg.Connection,
    incident_set: IncidentSet,
    embedder: EmbeddingProvider | None,
    methods: Sequence[Method],
    on_method: Callable[[Method], None] | None = None,
) -> dict[str, dict]:
    results = {}
    for method in methods:
        if on_method:
            on_method(method)
        scores = run_method(conn, incident_set, method, embedder)
        results[method] = {
            "summary": summarize(scores),
            "questions": [{**asdict(s), "top": list(s.top)} for s in scores],
        }
    return results


def render(results: Mapping[str, dict], header: str) -> str:
    cols = ["n", *(f"hit@{k}" for k in HIT_AT), "mrr"]
    lines = [header, "", f"{'method':<10} {'p50 ms':>6}  " + "  ".join(f"{c:>6}" for c in cols)]
    for method, result in results.items():
        overall = result["summary"]["overall"]
        p50 = result["summary"]["latency_p50_ms"]
        cells = "  ".join(f"{overall.get(c, ''):>6}" for c in cols)
        lines.append(f"{method:<10} {p50:>6.0f}  {cells}")

    answerable = [c for c in CATEGORIES if c != "none"]
    lines += [
        "",
        "hit@3 by category",
        f"{'method':<10} " + "  ".join(f"{c:>8}" for c in answerable),
    ]
    for method, result in results.items():
        by = result["summary"]["by_category"]
        cells = [f"{by[c]['hit@3']}/{by[c]['n']}" if c in by else "-" for c in answerable]
        lines.append(f"{method:<10} " + "  ".join(f"{cell:>8}" for cell in cells))

    lines += ["", "questions with no answer: results returned (top doc)"]
    for method, result in results.items():
        cells = [
            f"{q['id']} {len(q['top'])}+ ({q['top'][0]})" if q["top"] else f"{q['id']} none"
            for q in result["questions"]
            if not q["expected"]
        ]
        lines.append(f"{method:<10} " + ("; ".join(cells) or "-"))

    misses: dict[str, list[str]] = defaultdict(list)
    for method, result in results.items():
        for q in result["questions"]:
            if q["expected"] and not (q["rank"] and q["rank"] <= 3):
                misses[q["id"]].append(f"{method} ({q['top'][0] if q['top'] else 'nothing'})")
    lines += ["", "answerable questions outside the top 3: method (its top doc)"]
    lines += [f"  {qid}: {', '.join(m)}" for qid, m in sorted(misses.items())] or ["  none"]
    return "\n".join(lines) + "\n"
