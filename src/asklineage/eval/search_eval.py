"""Score search on the hand-written questions: one run per (mode, chunk kinds) configuration.

A hit is any expected model in the top k. A family hit also counts a node up to FAMILY_HOPS
upstream of an expected model (its staging model or source): the right data, one layer early.
It measures how much lineage-aware ranking could gain; it is not the headline metric.
Unanswerable questions are scored separately: what a
"nothing found" cutoff on the top score would catch, and what it would wrongly block.
"""

import statistics
import time
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass

import psycopg

from asklineage.embed.base import EmbeddingProvider
from asklineage.embed.chunks import ChunkKind
from asklineage.eval.search_set import SearchSet
from asklineage.search.query import Mode, search

K = 10
FAMILY_HOPS = 2
HIT_AT = (1, 3, 5, 10)
KIND_SETS: tuple[tuple[ChunkKind, ...], ...] = (
    ("summary",),
    ("summary", "columns"),
    ("summary", "columns", "sql"),
)


@dataclass(frozen=True)
class Config:
    mode: Mode
    kinds: tuple[ChunkKind, ...]
    prefer: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        lineage = "+lineage" if self.prefer else ""
        return f"{self.mode}{lineage}:{'+'.join(self.kinds)}"


@dataclass(frozen=True)
class QuestionScore:
    id: str
    category: str
    rank: int | None
    family_rank: int | None
    found: int
    expected: int
    top_score: float | None
    top: tuple[str, ...]
    latency_ms: int

    @property
    def answerable(self) -> bool:
        return self.expected > 0


def score_question(
    qid: str,
    category: str,
    expected: Sequence[str],
    returned: Sequence[str],
    top_score: float | None,
    latency_ms: int = 0,
    family: frozenset[str] = frozenset(),
) -> QuestionScore:
    wanted = set(expected)
    ranks = [i for i, uid in enumerate(returned, start=1) if uid in wanted]
    kin = wanted | family
    family_ranks = [i for i, uid in enumerate(returned, start=1) if uid in kin]
    return QuestionScore(
        qid,
        category,
        ranks[0] if ranks else None,
        family_ranks[0] if family_ranks else None,
        len(ranks),
        len(wanted),
        top_score,
        tuple(returned[:3]),
        latency_ms,
    )


def summarize(scores: Sequence[QuestionScore]) -> dict:
    answerable = [s for s in scores if s.answerable]
    by_category: dict[str, list[QuestionScore]] = defaultdict(list)
    for score in answerable:
        by_category[score.category].append(score)
    return {
        "overall": _metrics(answerable),
        "by_category": {c: _metrics(group) for c, group in sorted(by_category.items())},
        "cutoff": cutoff_table(scores),
        "latency_p50_ms": statistics.median(s.latency_ms for s in scores) if scores else 0,
    }


def _metrics(scores: Sequence[QuestionScore]) -> dict:
    n = len(scores)
    if not n:
        return {"n": 0}
    hits = {f"hit@{k}": sum(1 for s in scores if s.rank and s.rank <= k) for k in HIT_AT}
    return {
        "n": n,
        **hits,
        "mrr": round(sum(1 / s.rank for s in scores if s.rank) / n, 3),
        "recall@10": round(sum(s.found / s.expected for s in scores) / n, 3),
        "fam@1": sum(1 for s in scores if s.family_rank == 1),
        "fam@5": sum(1 for s in scores if s.family_rank and s.family_rank <= 5),
    }


def families(
    parents: Mapping[str, frozenset[str]], expected: Mapping[str, tuple[str, ...]]
) -> dict[str, frozenset[str]]:
    """Question id -> nodes up to FAMILY_HOPS upstream of its expected models."""
    result = {}
    for qid, roots in expected.items():
        seen: set[str] = set()
        frontier = set(roots)
        for _ in range(FAMILY_HOPS):
            frontier = {p for node in frontier for p in parents.get(node, ())} - seen
            seen |= frontier
        result[qid] = frozenset(seen - set(roots))
    return result


def cutoff_table(scores: Sequence[QuestionScore]) -> list[dict]:
    """For each candidate cutoff: unanswerable questions caught vs answerable ones blocked.

    An answerable question is blocked when its top score is below the cutoff, whether or not
    its top hit was right; that is what a user would see.
    """
    unanswerable = [s.top_score for s in scores if not s.answerable]
    answerable = [s.top_score for s in scores if s.answerable]
    if not unanswerable or not answerable:
        return []
    candidates = sorted({x for x in unanswerable if x is not None})
    rows = []
    for cutoff in candidates:
        # Just above an unanswerable top score, so that question is caught.
        bar = cutoff + 1e-9
        rows.append(
            {
                "cutoff": round(cutoff, 4),
                "caught": sum(1 for x in unanswerable if x is None or x < bar),
                "of_unanswerable": len(unanswerable),
                "blocked": sum(1 for x in answerable if x is None or x < bar),
                "of_answerable": len(answerable),
            }
        )
    return rows


def run_config(
    conn: psycopg.Connection,
    search_set: SearchSet,
    expected: Mapping[str, tuple[str, ...]],
    embedder: EmbeddingProvider | None,
    config: Config,
    family: Mapping[str, frozenset[str]] | None = None,
) -> list[QuestionScore]:
    scores = []
    for question in search_set.questions:
        started = time.perf_counter()
        result = search(
            conn,
            question.question,
            None if config.mode == "keyword" else embedder,
            project=search_set.project,
            kinds=config.kinds,
            k=K,
            keyword_only=config.mode == "keyword",
            prefer=config.prefer,
        )
        if result.mode != config.mode:
            raise RuntimeError(f"{config.label} ran as {result.mode}: {result.note}")
        latency_ms = round((time.perf_counter() - started) * 1000)
        returned = [hit.unique_id for hit in result.hits]
        top = result.hits[0].score if result.hits else None
        scores.append(
            score_question(
                question.id,
                question.category,
                expected[question.id],
                returned,
                top,
                latency_ms,
                (family or {}).get(question.id, frozenset()),
            )
        )
    return scores


def run_search_eval(
    conn: psycopg.Connection,
    search_set: SearchSet,
    expected: Mapping[str, tuple[str, ...]],
    embedder: EmbeddingProvider | None,
    configs: Sequence[Config],
    on_config: Callable[[Config], None] | None = None,
    family: Mapping[str, frozenset[str]] | None = None,
) -> dict[str, dict]:
    results = {}
    for config in configs:
        if on_config:
            on_config(config)
        scores = run_config(conn, search_set, expected, embedder, config, family)
        results[config.label] = {
            "summary": summarize(scores),
            "questions": [{**asdict(s), "top": list(s.top)} for s in scores],
        }
    return results


def render(results: Mapping[str, dict], header: str) -> str:
    lines = [header, ""]
    cols = ["n", *(f"hit@{k}" for k in HIT_AT), "mrr", "recall@10", "fam@1", "fam@5"]
    lines.append(f"{'config':<38} {'p50 ms':>6}  " + "  ".join(f"{c:>9}" for c in cols))
    for label, result in results.items():
        overall = result["summary"]["overall"]
        p50 = result["summary"]["latency_p50_ms"]
        lines.append(
            f"{label:<38} {p50:>6.0f}  " + "  ".join(f"{overall.get(c, ''):>9}" for c in cols)
        )
    lines += ["", "hit@5 by category"]
    categories = sorted({c for r in results.values() for c in r["summary"]["by_category"]})
    lines.append(f"{'config':<38} " + "  ".join(f"{c:>11}" for c in categories))
    for label, result in results.items():
        by = result["summary"]["by_category"]
        cells = [f"{by[c]['hit@5']}/{by[c]['n']}" if c in by else "-" for c in categories]
        lines.append(f"{label:<38} " + "  ".join(f"{cell:>11}" for cell in cells))
    lines += [
        "",
        "nothing-found cutoff on the top score (caught unanswerable / blocked answerable)",
    ]
    for label, result in results.items():
        rows = result["summary"]["cutoff"]
        cells = [
            f"{r['cutoff']}: {r['caught']}/{r['of_unanswerable']} caught, "
            f"{r['blocked']}/{r['of_answerable']} blocked"
            for r in rows
        ]
        lines.append(f"{label:<38} " + ("; ".join(cells) or "-"))
    misses = defaultdict(list)
    for label, result in results.items():
        for q in result["questions"]:
            if q["expected"] and not (q["rank"] and q["rank"] <= 5):
                misses[q["id"]].append(label)
    lines += ["", "answerable questions outside the top 5 (configs)"]
    lines += [f"  {qid}: {', '.join(labels)}" for qid, labels in sorted(misses.items())] or [
        "  none"
    ]
    return "\n".join(lines) + "\n"
