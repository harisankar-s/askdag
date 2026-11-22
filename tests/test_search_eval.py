from dataclasses import replace

import pytest
from helpers import TEST_PROJECT, make_node

from asklineage.embed.base import index_name
from asklineage.embed.hashing import HashingEmbedder
from asklineage.embed.store import sync_chunks
from asklineage.eval.search_eval import (
    KIND_SETS,
    Config,
    WarmEmbedder,
    cutoff_table,
    families,
    percentile,
    render,
    run_search_eval,
    score_question,
    summarize,
)
from asklineage.eval.search_set import SearchQuestion, SearchSet
from asklineage.search.scan import variants


def test_rank_is_the_first_expected_hit():
    score = score_question("q", "column", ["b", "c"], ["a", "c", "b"], 0.7)

    assert (score.rank, score.found, score.expected) == (2, 2, 2)
    assert score.top == ("a", "c", "b")


def test_a_miss_has_no_rank():
    assert score_question("q", "column", ["z"], ["a", "b"], 0.7).rank is None


def test_summary_counts_hits_mrr_and_recall_over_answerable_only():
    scores = [
        score_question("a", "described", ["x"], ["x"], 0.9),
        score_question("b", "column", ["y", "z"], ["p", "q", "y"], 0.8),
        score_question("c", "column", ["w"], ["p"], 0.7),
        score_question("n", "unanswerable", [], ["p"], 0.6),
    ]

    overall = summarize(scores)["overall"]

    assert overall == {
        "n": 3,
        "hit@1": 1,
        "hit@3": 2,
        "hit@5": 2,
        "hit@10": 2,
        "mrr": round((1 + 1 / 3) / 3, 3),
        "recall@10": round((1 + 0.5 + 0) / 3, 3),
        "fam@1": 1,
        "fam@5": 2,
    }
    assert summarize(scores)["by_category"]["column"]["hit@3"] == 1


def test_cutoff_table_trades_caught_against_blocked():
    scores = [
        score_question("a", "described", ["x"], ["x"], 0.80),
        score_question("b", "column", ["y"], ["y"], 0.65),
        score_question("n1", "unanswerable", [], ["p"], 0.60),
        score_question("n2", "unanswerable", [], ["p"], 0.70),
    ]

    assert cutoff_table(scores) == [
        {"cutoff": 0.6, "caught": 1, "of_unanswerable": 2, "blocked": 0, "of_answerable": 2},
        {"cutoff": 0.7, "caught": 2, "of_unanswerable": 2, "blocked": 1, "of_answerable": 2},
    ]


def test_index_name_includes_the_document_prefix():
    embedder = HashingEmbedder()
    assert index_name(embedder) == "hashing-768"
    embedder.document_prefix = "search_document: "
    assert index_name(embedder) == "hashing-768 [search_document:]"


def test_run_search_eval_on_postgres(conn, load_graph):
    stops = replace(make_node("dim_stops"), description="One row per bus stop location.")
    fares = replace(make_node("fct_fares"), description="One row per card payment.")
    load_graph([stops, fares], [])
    embedder = HashingEmbedder()
    sync_chunks(conn, TEST_PROJECT, embedder)
    search_set = SearchSet(
        TEST_PROJECT,
        (
            SearchQuestion("q1", "described", "bus stop location", ("dim_stops",)),
            SearchQuestion("n1", "unanswerable", "rider survey results", ()),
        ),
        "sha",
    )
    expected = {"q1": (stops.unique_id,), "n1": ()}
    configs = [Config(mode, kinds) for mode in ("semantic", "keyword") for kinds in KIND_SETS]

    results = run_search_eval(conn, search_set, expected, embedder, configs)

    assert list(results) == [c.label for c in configs]
    assert results["semantic:summary"]["summary"]["overall"]["hit@1"] == 1
    assert results["keyword:summary"]["summary"]["overall"]["hit@1"] == 1
    assert "hit@5 by category" in render(results, "header")


def test_family_hit_counts_an_upstream_node_but_the_strict_rank_does_not():
    score = score_question(
        "q", "undescribed", ["mart"], ["stg", "other", "mart"], 0.7, family=frozenset({"stg"})
    )

    assert (score.rank, score.family_rank) == (3, 1)


def test_families_walk_up_two_hops_only():
    parents = {
        "mart": frozenset({"stg"}),
        "stg": frozenset({"source"}),
        "source": frozenset({"raw"}),
    }

    assert families(parents, {"q": ("mart",), "n": ()}) == {
        "q": frozenset({"stg", "source"}),
        "n": frozenset(),
    }


def test_warm_embedder_embeds_each_question_once_and_times_it():
    calls = []

    class Counting(HashingEmbedder):
        def embed(self, texts, *, kind):
            calls.append(tuple(texts))
            return super().embed(texts, kind=kind)

    inner = Counting()
    warm = WarmEmbedder(inner, ["a", "b", "a"])

    assert warm.embed(["b"], kind="query").vectors == inner.embed(["b"], kind="query").vectors
    assert calls[:2] == [("a",), ("b",)] and len(warm.latencies_ms) == 2
    assert index_name(warm) == index_name(inner)
    with pytest.raises(KeyError):
        warm.embed(["unseen"], kind="query")
    with pytest.raises(ValueError, match="question vectors"):
        warm.embed(["a"], kind="document")


def test_percentile_is_nearest_rank():
    values = [float(v) for v in range(1, 21)]

    assert (percentile(values, 0.5), percentile(values, 0.95)) == (10.0, 19.0)
    assert percentile([], 0.95) == 0.0


def test_exact_runs_once_and_the_other_scans_at_each_overfetch():
    assert variants(["exact", "topn", "hnsw"], [2, 8]) == [
        ("exact", 2),
        ("topn", 2),
        ("topn", 8),
        ("hnsw", 2),
        ("hnsw", 8),
    ]


def test_scan_configs_get_their_own_labels(conn, load_graph):
    stops = replace(make_node("dim_stops"), description="One row per bus stop location.")
    load_graph([stops], [])
    embedder = HashingEmbedder()
    sync_chunks(conn, TEST_PROJECT, embedder)
    search_set = SearchSet(
        TEST_PROJECT,
        (SearchQuestion("q1", "described", "bus stop location", ("dim_stops",)),),
        "sha",
    )
    full = KIND_SETS[-1]
    configs = [Config("semantic", full, (), scan, 2) for scan in ("exact", "topn", "hnsw")]

    results = run_search_eval(
        conn,
        search_set,
        {"q1": (stops.unique_id,)},
        WarmEmbedder(embedder, ["bus stop location"]),
        configs,
    )

    assert list(results) == [
        "semantic:summary+columns+sql",
        "semantic[topn x2]:summary+columns+sql",
        "semantic[hnsw x2]:summary+columns+sql",
    ]
    assert all(r["summary"]["overall"]["hit@1"] == 1 for r in results.values())
    assert all("latency_p95_ms" in r["summary"] for r in results.values())
