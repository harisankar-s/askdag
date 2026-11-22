from dataclasses import replace

from lineage_examples import EXAMPLE_LINEAGE, EXAMPLE_ROOT

from asklineage.answer.lineage_answer import answer_lineage
from asklineage.cli import format_summary


def test_summary_then_highlights_with_owner_tags():
    output = format_summary(answer_lineage(EXAMPLE_LINEAGE, EXAMPLE_ROOT))

    assert output.startswith("Summary\n  ")
    assert "\nHighlights\n  - `mart_weekly` (no owner recorded) depends on" in output


def test_an_empty_result_has_only_the_summary():
    empty = replace(EXAMPLE_LINEAGE, nodes=(), edges=(), truncated=False)

    assert format_summary(answer_lineage(empty, EXAMPLE_ROOT)) == (
        "Summary\n  Nothing reads from `stg_orders`.\n"
    )


def test_lines_wrap_at_100_columns_without_splitting_hyphenated_words():
    answer = answer_lineage(EXAMPLE_LINEAGE, EXAMPLE_ROOT)
    summary = "word " * 18 + "transit-database-team owns these nodes."

    output = format_summary(replace(answer, summary=summary))

    assert "transit-database-team" in output
    assert all(len(line) <= 100 for line in output.splitlines())
