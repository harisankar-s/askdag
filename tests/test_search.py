from dataclasses import replace

import pytest
from helpers import TEST_PROJECT, make_node

from asklineage.embed.hashing import HashingEmbedder
from asklineage.embed.store import sync_chunks
from asklineage.ingest.load import load_manifest
from asklineage.ingest.manifest import Column, ParsedManifest
from asklineage.search.query import search

STOPS = replace(
    make_node("dim_stops", owner="gtfs-team"), description="One row per bus stop location."
)
FARES = replace(make_node("fct_fare_payments"), description="One row per card payment.")
BREAKDOWNS = replace(
    make_node("fct_breakdowns"), raw_code="select major_mechanical_failures from stg"
)
COLUMNS = (
    Column(
        STOPS.unique_id, "wheelchair_boarding", "Whether riders can board in a wheelchair.", None
    ),
)


@pytest.fixture
def loaded(conn):
    load_manifest(
        conn,
        ParsedManifest(
            project_name=TEST_PROJECT,
            dbt_version="1.10.8",
            schema_version="v12",
            generated_at=None,
            invocation_id=None,
            nodes=(STOPS, FARES, BREAKDOWNS),
            edges=(),
            columns=COLUMNS,
            tests=(),
        ),
    )
    return conn


@pytest.fixture
def embedded(loaded):
    embedder = HashingEmbedder()
    sync_chunks(loaded, TEST_PROJECT, embedder)
    return loaded, embedder


def names(result):
    return [hit.name for hit in result.hits]


def test_semantic_search_ranks_the_matching_node_first(embedded):
    conn, embedder = embedded

    result = search(conn, "bus stop location", embedder, project=TEST_PROJECT)

    assert result.mode == "semantic"
    assert result.model == "hashing-768"
    assert names(result)[0] == "dim_stops"
    assert result.hits[0].owner == "gtfs-team"
    assert len(names(result)) == len(set(names(result))) == 3


def test_kinds_choose_which_chunks_can_match(embedded):
    conn, embedder = embedded

    with_sql = search(conn, "mechanical failures", embedder, project=TEST_PROJECT)
    summary_only = search(
        conn, "mechanical failures", embedder, project=TEST_PROJECT, kinds=["summary"]
    )

    assert names(with_sql)[0] == "fct_breakdowns"
    assert with_sql.hits[0].kind == "sql"
    assert {hit.kind for hit in summary_only.hits} == {"summary"}


def test_chunks_from_another_model_are_ignored(embedded):
    conn, _ = embedded
    other = HashingEmbedder()
    other.model = "other-model"

    result = search(conn, "bus stop", other, project=TEST_PROJECT)

    assert result.mode == "keyword"
    assert "asklineage embed" in result.note


def test_without_an_embedder_search_uses_keywords(loaded):
    result = search(loaded, "Where are the bus stops?", None, project=TEST_PROJECT)

    assert (result.mode, result.note) == ("keyword", "EMBED_PROVIDER=none")
    assert names(result) == ["dim_stops"]


def test_keyword_search_matches_any_word_and_column_text(loaded):
    result = search(loaded, "wheelchair payment", None, project=TEST_PROJECT)

    assert set(names(result)) == {"dim_stops", "fct_fare_payments"}


def test_keyword_search_reads_sql_only_when_asked(loaded):
    with_sql = search(loaded, "mechanical failures", None, project=TEST_PROJECT)
    without = search(
        loaded, "mechanical failures", None, project=TEST_PROJECT, kinds=["summary", "columns"]
    )

    assert names(with_sql) == ["fct_breakdowns"]
    assert names(without) == []


def test_keyword_only_skips_embeddings(embedded):
    conn, embedder = embedded

    result = search(conn, "bus stop", embedder, project=TEST_PROJECT, keyword_only=True)

    assert (result.mode, result.note) == ("keyword", None)


def test_a_question_of_only_stopwords_finds_nothing(loaded):
    assert search(loaded, "where is the", None, project=TEST_PROJECT).hits == ()


def load_layers(conn, marts: int = 1):
    stg = replace(
        make_node("stg_breakdowns"),
        file_path="models/staging/stg_breakdowns.sql",
        description="Mechanical failures of vehicles in service.",
    )
    mart_nodes = [
        replace(make_node(f"fct_mart_{i}"), file_path=f"models/mart/fct_mart_{i}.sql")
        for i in range(marts)
    ]
    load_manifest(
        conn,
        ParsedManifest(
            project_name=TEST_PROJECT,
            dbt_version="1.10.8",
            schema_version="v12",
            generated_at=None,
            invocation_id=None,
            nodes=(stg, *mart_nodes),
            edges=tuple((stg.unique_id, m.unique_id) for m in mart_nodes),
            columns=(),
            tests=(),
        ),
    )
    embedder = HashingEmbedder()
    sync_chunks(conn, TEST_PROJECT, embedder)
    return embedder


def test_a_matching_staging_model_lifts_the_mart_it_feeds(conn):
    embedder = load_layers(conn)
    question = "mechanical failures of vehicles"

    plain = search(conn, question, embedder, project=TEST_PROJECT)
    lifted = search(conn, question, embedder, project=TEST_PROJECT, prefer=["models/mart/*"])

    assert names(plain) == ["stg_breakdowns", "fct_mart_0"]
    assert names(lifted) == ["fct_mart_0", "stg_breakdowns"]
    assert lifted.hits[0].via == "stg_breakdowns"
    assert lifted.hits[0].score == pytest.approx(plain.hits[0].score)
    assert lifted.hits[1].score == pytest.approx(plain.hits[0].score * 0.98)


def test_one_upstream_node_lifts_at_most_three_marts(conn):
    embedder = load_layers(conn, marts=5)

    result = search(
        conn, "mechanical failures", embedder, project=TEST_PROJECT, prefer=["models/mart/*"]
    )

    assert sum(1 for hit in result.hits if hit.via) == 3
