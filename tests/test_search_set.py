from pathlib import Path

import pytest

from asklineage.eval.search_set import (
    CATEGORIES,
    SearchSetError,
    check_against_manifest,
    description_overlap,
    load_search_set,
)

COMMITTED = Path(__file__).parent.parent / "config" / "search_questions.calitp.yml"

MANIFEST = {
    "nodes": {
        "model.shop.fct_orders": {
            "name": "fct_orders",
            "resource_type": "model",
            "package_name": "shop",
            "description": "One row per order placed in the web shop.",
        },
        "model.shop.dim_customers": {
            "name": "dim_customers",
            "resource_type": "model",
            "package_name": "shop",
            "description": "",
        },
        "seed.shop.country_codes": {
            "name": "country_codes",
            "resource_type": "seed",
            "package_name": "shop",
            "description": "",
        },
    }
}


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "questions.yml"
    path.write_text("version: 1\nproject: shop\nquestions:\n" + body)
    return path


VALID = """
  - id: q1
    category: described
    question: Where are the shop orders?
    expected: [fct_orders]
  - id: q2
    category: unanswerable
    question: Where are the warehouse robots?
    expected: []
"""


def test_valid_set_resolves_names_to_unique_ids(tmp_path):
    search_set = load_search_set(write(tmp_path, VALID))

    expected = check_against_manifest(search_set, MANIFEST)

    assert expected == {"q1": ("model.shop.fct_orders",), "q2": ()}
    assert len(search_set.sha) == 12


def test_missing_model_makes_the_set_stale(tmp_path):
    search_set = load_search_set(write(tmp_path, VALID.replace("[fct_orders]", "[fct_gone]")))

    with pytest.raises(SearchSetError, match="no model named 'fct_gone'"):
        check_against_manifest(search_set, MANIFEST)


def test_a_seed_is_not_a_model(tmp_path):
    search_set = load_search_set(write(tmp_path, VALID.replace("[fct_orders]", "[country_codes]")))

    with pytest.raises(SearchSetError, match="country_codes"):
        check_against_manifest(search_set, MANIFEST)


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        (VALID + VALID, "duplicate id"),
        (VALID.replace("category: described", "category: vague"), "not one of"),
        (VALID.replace("expected: [fct_orders]", "expected: []"), "only unanswerable"),
        (VALID.replace("expected: []", "expected: [fct_orders]"), "only unanswerable"),
        (VALID.replace("[fct_orders]", "[fct_orders, fct_orders]"), "listed twice"),
        (VALID.replace("Where are the shop orders?", "''"), "no question"),
    ],
)
def test_malformed_sets_are_refused(tmp_path, body, problem):
    with pytest.raises(SearchSetError, match=problem):
        load_search_set(write(tmp_path, body))


def test_wrong_version_is_refused(tmp_path):
    path = tmp_path / "questions.yml"
    path.write_text("version: 2\nproject: shop\nquestions: []\n")

    with pytest.raises(SearchSetError, match="version 1"):
        load_search_set(path)


def test_overlap_counts_question_words_in_name_and_description(tmp_path):
    q1, q2 = load_search_set(write(tmp_path, VALID)).questions

    # "shop" and "orders" are in the description; "where" and "the" are stopwords.
    assert description_overlap(q1, MANIFEST, "shop") == 1.0
    assert description_overlap(q2, MANIFEST, "shop") is None


def test_committed_calitp_set_is_well_formed():
    search_set = load_search_set(COMMITTED)

    categories = {question.category for question in search_set.questions}
    assert categories == set(CATEGORIES)
    assert search_set.project == "calitp_warehouse"
