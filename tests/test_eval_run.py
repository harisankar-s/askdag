from pathlib import Path

import psycopg
import pytest
from synthetic_manifest import build_manifest

from asklineage.api.flow import run_ask
from asklineage.api.models import AbstainResponse, AskRequest, ClarificationResponse
from asklineage.eval.conformance import loaded_owners
from asklineage.eval.golden import GoldenSet, NameIndex, build_golden
from asklineage.eval.report import fmt, render, summarize
from asklineage.eval.run import EvalContext, run_golden, run_info, write_run
from asklineage.eval.score import TurnScore, score_turn
from asklineage.eval.truth import truth_graph
from asklineage.ingest.load import load_manifest
from asklineage.ingest.manifest import parse_manifest

HUB = "model.synthetic.int_hub"


@pytest.fixture
def context(conn: psycopg.Connection) -> EvalContext:
    manifest = build_manifest()
    load_manifest(conn, parse_manifest(manifest))
    graph = truth_graph(manifest)
    owners = loaded_owners(conn, graph.project_name)
    return EvalContext(conn, None, graph, NameIndex(manifest, graph), owners, max_named=20)


@pytest.fixture
def golden(context: EvalContext) -> GoldenSet:
    return build_golden(build_manifest(), context.graph, context.owners)


@pytest.fixture
def scores(context: EvalContext, golden: GoldenSet) -> list[TurnScore]:
    return run_golden(context, golden)


def by_id(golden: GoldenSet, item_id: str):
    return next(item for item in golden.items if item.id == item_id)


def score(context: EvalContext, golden: GoldenSet, item_id: str, turn: int, response, **kw):
    item = by_id(golden, item_id)
    return score_turn(
        item,
        turn,
        item.turns[turn - 1],
        response,
        llm=True,
        graph=context.graph,
        index=context.index,
        owners=context.owners,
        **kw,
    )


def hub_answer(context: EvalContext):
    request = AskRequest(node="int_hub", project="synthetic")
    return run_ask(context.conn, None, request, max_named=20)


def test_without_an_llm_every_structured_turn_is_right(scores: list[TurnScore]) -> None:
    wrong = [s for s in scores if s.outcome not in ("correct", "blocked")]
    assert wrong == [], [(s.item_id, s.turn, s.problems) for s in wrong]
    blocked = [(s.item_id, s.turn) for s in scores if s.outcome == "blocked"]
    # The typo is free text, so without an LLM there is no clarification to choose from.
    assert blocked == [("t-typo-choose", 2)]
    lineage = [s for s in scores if s.kind == "lineage"]
    assert lineage and all(s.lineage_ok and s.facts_ok for s in lineage)


def test_without_an_llm_free_text_abstains_with_llm_off(scores: list[TurnScore]) -> None:
    free = [s for s in scores if s.free_text]
    assert free and all(s.kind == "abstain" and s.reason == "llm_off" for s in free)
    summary = summarize(scores, llm=False)
    count, total = summary["llm_off"]
    assert count == total == len(free)
    assert summary["bugs"] == []


def test_a_wrong_table_or_count_is_a_silent_wrong_answer_and_a_bug(
    context: EvalContext, golden: GoldenSet
) -> None:
    answer = hub_answer(context)
    short = answer.model_copy(update={"table": answer.table[1:]})
    scored = score(context, golden, "s-hub-down", 1, short)
    assert (scored.outcome, scored.lineage_ok, scored.facts_ok) == ("silent_wrong", False, True)

    facts = answer.facts.model_copy(update={"total": answer.facts.total + 1})
    miscounted = score(context, golden, "s-hub-down", 1, answer.model_copy(update={"facts": facts}))
    assert (miscounted.outcome, miscounted.facts_ok) == ("silent_wrong", False)
    assert summarize([scored, miscounted], llm=True)["bugs"] == ["s-hub-down turn 1"] * 2


def test_a_follow_up_answered_about_the_previous_node_is_flagged(
    context: EvalContext, golden: GoldenSet
) -> None:
    scored = score(context, golden, "t-new-node", 2, hub_answer(context), previous_node=HUB)
    assert scored.outcome == "silent_wrong"
    assert scored.reused_previous
    assert scored.node_ok is False and scored.lineage_ok


def test_non_answers_are_safe_wrong_acceptable_or_silent(
    context: EvalContext, golden: GoldenSet
) -> None:
    abstained = AbstainResponse(reason="not_lineage", message="m")
    not_found = ClarificationResponse(reason="not_found", query="q", message="m", candidates=[])
    assert score(context, golden, "f-hub-down", 1, abstained).outcome == "safe_wrong"
    assert score(context, golden, "vague", 1, not_found).outcome == "acceptable"
    assert score(context, golden, "off-sql", 1, hub_answer(context)).outcome == "silent_wrong"
    typo = score(context, golden, "f-typo-up", 1, not_found)
    assert (typo.outcome, typo.suggestion_rank) == ("correct", 0)


def test_a_run_writes_its_files_and_report(
    tmp_path: Path, golden: GoldenSet, scores: list[TurnScore]
) -> None:
    info = run_info(
        Path("synthetic.json"), b"{}", golden, llm_provider="none", llm_model=None, max_named=20
    )
    summary = summarize(scores, llm=False)
    report = render(info, golden, summary, scores)
    assert golden.sha in report and "## Without an LLM" in report
    run_dir = write_run(tmp_path, info, golden, scores, summary, report)
    names = sorted(path.name for path in run_dir.iterdir())
    assert names == ["golden.json", "report.md", "results.jsonl", "summary.json"]
    assert len((run_dir / "results.jsonl").read_text().splitlines()) == len(scores)
    # The LLM sections render too, even with nothing to show.
    assert "## Follow-ups" in render(info, golden, summarize(scores, llm=True), scores)


def test_shares_print_a_percentage_only_from_twenty_turns() -> None:
    assert fmt((4, 5)) == "4/5"
    assert fmt((18, 20)) == "18/20 (90%)"
    assert fmt((0, 0)) == "–"
