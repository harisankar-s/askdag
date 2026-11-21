from dataclasses import replace

from lineage_examples import EXAMPLE_LINEAGE, EXAMPLE_MAX_NAMED, EXAMPLE_ROOT

from asklineage.answer.lineage_answer import answer_lineage
from asklineage.answer.prompt import EXAMPLE_HIGHLIGHTS
from asklineage.cli import format_summary
from asklineage.llm.base import LLMError
from asklineage.llm.scripted import ScriptedProvider

GOOD = EXAMPLE_HIGHLIGHTS.model_dump()
BAD_TEXT = "`mart_revenue` reads directly from `int_order_items`."
BAD = {
    "claims": [
        {
            "upstream": "int_order_items",
            "downstream": "mart_revenue",
            "hops": "direct",
            "text": BAD_TEXT,
        }
    ]
}


def summarize(replies, lineage=EXAMPLE_LINEAGE, verbose=False):
    provider = None if replies is None else ScriptedProvider(replies, model="qwen2.5:7b")
    answer = answer_lineage(lineage, EXAMPLE_ROOT, provider, max_named=EXAMPLE_MAX_NAMED)
    return format_summary(answer, verbose=verbose)


def test_summary_then_grounded_highlights():
    output = summarize([GOOD])

    assert output.startswith("Summary\n  5 nodes depend on `stg_orders` within 2 hops")
    assert "\nHighlights (grounded: qwen2.5:7b, 0 ms)\n" in output
    assert "  - `fct_orders` (finance) reads directly from `stg_orders`." in output


def test_retry_is_labelled():
    assert "Highlights (grounded on retry: qwen2.5:7b, 2 attempts" in summarize([BAD, GOOD])


def test_rejected_highlights_keep_the_summary_and_point_to_verbose():
    output = summarize([BAD, BAD])

    assert output.startswith("Summary\n  5 nodes depend on")
    hint = "run with --verbose to see why"
    assert f"Highlights: none (rejected by the check 2 times; {hint})" in output


def test_verbose_lists_each_attempt_with_the_failing_sentence():
    output = summarize(["not json", BAD], verbose=True)

    assert "attempt 1: 1 problem (reply was not valid JSON for the schema)" in output
    assert "attempt 2: 1 problem (qwen2.5:7b, 0 ms)" in output
    assert (
        "    - claim 1: `int_order_items` does not feed `mart_revenue` directly\n"
        f"      said: {BAD_TEXT!r}"
    ) in output
    assert "run with --verbose" not in output


def test_unreachable_llm_is_labelled():
    output = summarize([LLMError("Cannot reach http://localhost:11434/v1")], verbose=True)

    assert "Highlights: none (the LLM could not be reached)" in output
    assert "attempt 1: error: Cannot reach" in output


def test_off_and_empty_labels():
    assert "Highlights: off (no LLM configured)" in summarize(None)
    empty = replace(EXAMPLE_LINEAGE, nodes=(), edges=(), truncated=False)
    assert summarize([GOOD], lineage=empty) == "Summary\n  Nothing reads from `stg_orders`.\n"


def test_lines_wrap_at_100_columns_without_splitting_hyphenated_words():
    answer = answer_lineage(EXAMPLE_LINEAGE, EXAMPLE_ROOT, None, max_named=EXAMPLE_MAX_NAMED)
    summary = "word " * 18 + "transit-database-team owns these nodes."

    output = format_summary(replace(answer, summary=summary))

    assert "transit-database-team" in output
    assert all(len(line) <= 100 for line in output.splitlines())
