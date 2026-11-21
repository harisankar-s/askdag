from collections import Counter
from pathlib import Path

import psycopg
import pytest
from synthetic_manifest import build_manifest

from asklineage.answer.generate import Attempt, Violation
from asklineage.answer.lineage_answer import HighlightStatus
from asklineage.api.flow import run_ask
from asklineage.api.models import AskRequest, AttemptInfo, HighlightInfo, LineageResponse
from asklineage.eval.golden import NameIndex, build_golden
from asklineage.eval.highlights import named_nodes, score_highlights
from asklineage.eval.report import render, summarize
from asklineage.eval.run import run_info
from asklineage.eval.score import score_turn
from asklineage.eval.truth import TruthGraph, truth_graph
from asklineage.ingest.load import load_manifest
from asklineage.ingest.manifest import parse_manifest
from asklineage.ingest.owners import OwnerRule

RULES = (
    OwnerRule("models/intermediate/*", "team_a"),
    OwnerRule("models/marts/*", "team_b"),
)


@pytest.fixture
def graph(conn: psycopg.Connection) -> TruthGraph:
    manifest = build_manifest()
    load_manifest(conn, parse_manifest(manifest, RULES))
    return truth_graph(manifest)


@pytest.fixture
def answer(conn: psycopg.Connection, graph: TruthGraph) -> LineageResponse:
    return run_ask(conn, None, AskRequest(node="int_hub", project="synthetic"), max_named=20)


def with_highlights(
    answer: LineageResponse, claims: list[tuple[str, str, str]], first_rules: tuple[str, ...] = ()
) -> LineageResponse:
    attempts = [AttemptInfo(stage="highlights", passed=not first_rules, rules=list(first_rules))]
    if first_rules:
        attempts.append(AttemptInfo(stage="highlights", passed=True))
    highlights = [HighlightInfo(text="t", upstream=u, downstream=d, hops=h) for u, d, h in claims]
    status = HighlightStatus.PASSED_AFTER_RETRY if first_rules else HighlightStatus.PASSED
    return answer.model_copy(
        update={"highlights": highlights, "attempts": attempts, "highlight_status": status}
    )


def rows_at(answer: LineageResponse, depth: int) -> list[str]:
    return [row.display_name for row in answer.table if row.depth == depth]


def test_highlights_are_checked_against_the_manifest(
    answer: LineageResponse, graph: TruthGraph
) -> None:
    child, grandchild = rows_at(answer, 1)[0], rows_at(answer, 2)[0]
    claims = [
        ("int_hub", child, "direct"),
        ("int_hub", grandchild, "indirect"),
        (child, "int_hub", "direct"),  # backwards
    ]
    scored = score_highlights(with_highlights(answer, claims), graph, 20, "seed")
    assert (scored.claims, scored.manifest_false) == (3, 1)


def test_false_relations_are_told_apart_from_output_rules(
    answer: LineageResponse, graph: TruthGraph
) -> None:
    claims = [("int_hub", rows_at(answer, 1)[0], "direct")]
    false = with_highlights(answer, claims, ("no_such_edge", "unsupported_number"))
    style = with_highlights(answer, claims, ("unsupported_number",))
    assert score_highlights(false, graph, 20, "seed").hallucinated_first
    assert not score_highlights(style, graph, 20, "seed").hallucinated_first
    assert score_highlights(answer, graph, 20, "seed") is None  # no LLM, nothing to score


def test_one_team_picks_score_below_the_code_baseline(
    answer: LineageResponse, graph: TruthGraph
) -> None:
    majority = Counter(row.owner for row in answer.table).most_common(1)[0][0]
    owner_of = {row.display_name: row.owner for row in answer.table}
    same_team = [name for name in rows_at(answer, 1) if owner_of[name] == majority][:2]
    assert len(same_team) == 2
    claims = [("int_hub", name, "direct") for name in same_team]
    scored = score_highlights(with_highlights(answer, claims), graph, 20, "seed")
    assert scored.llm.minority_hit is False
    assert scored.code.minority_hit is True
    assert scored.code.nodes == tuple(n.name for n in named_nodes(answer, 20)[:2])
    assert len(scored.random.nodes) == 2


def test_attempt_info_keeps_the_rule_of_each_violation() -> None:
    attempt = Attempt(violations=(Violation(1, "no_such_edge", "a does not feed b"),))
    info = AttemptInfo.from_attempt("highlights", attempt)
    assert info.rules == ["no_such_edge"]
    assert info.violations == ["claim 1: a does not feed b"]


def test_the_report_shows_grounding_and_usefulness(
    conn: psycopg.Connection, answer: LineageResponse, graph: TruthGraph
) -> None:
    manifest = build_manifest()
    owners = {row.unique_id: row.owner for row in answer.table}
    golden = build_golden(manifest, graph, owners)
    item = next(item for item in golden.items if item.id == "s-hub-down")
    response = with_highlights(answer, [("int_hub", rows_at(answer, 1)[0], "direct")])
    score = score_turn(
        item,
        1,
        item.turns[0],
        response,
        llm=True,
        graph=graph,
        index=NameIndex(manifest, graph),
        owners=owners,
    )
    summary = summarize([score], llm=True)
    assert summary["highlight_grounding"]["claims"] == 1
    info = run_info(Path("m.json"), b"{}", golden, llm_provider="x", llm_model="m", max_named=20)
    report = render(info, golden, summary, [score])
    assert "### Usefulness" in report
    assert "Answers with passed highlights: 1." in report
