from dataclasses import replace

import pytest
from helpers import TEST_PROJECT, make_node

from asklineage.embed.base import EmbeddingError
from asklineage.embed.hashing import HashingEmbedder
from asklineage.embed.store import project_chunks, sync_chunks


class CountingEmbedder(HashingEmbedder):
    def __init__(self, dimensions: int = 768, model: str | None = None) -> None:
        super().__init__(dimensions)
        self.model = model or self.model
        self.texts: list[str] = []

    def embed(self, texts, *, kind):
        self.texts += texts
        return super().embed(texts, kind=kind)


def nodes(description: str = "One row per stop."):
    stops = replace(make_node("dim_stops"), description=description, raw_code="select 1")
    return [stops, make_node("fct_trips")], [(stops.unique_id, "model.lineage_test.fct_trips")]


def stored(conn):
    return conn.execute(
        "SELECT node_id, kind, model FROM node_chunks WHERE project_name = %s ORDER BY 1, 2",
        (TEST_PROJECT,),
    ).fetchall()


def test_first_sync_embeds_every_chunk(conn, load_graph):
    load_graph(*nodes())
    embedder = CountingEmbedder()

    report = sync_chunks(conn, TEST_PROJECT, embedder)

    assert (report.embedded, report.kept, report.deleted) == (3, 0, 0)
    assert report.chunks == {"summary": 2, "sql": 1}
    assert [(r["node_id"], r["kind"]) for r in stored(conn)] == [
        ("model.lineage_test.dim_stops", "sql"),
        ("model.lineage_test.dim_stops", "summary"),
        ("model.lineage_test.fct_trips", "summary"),
    ]


def test_second_sync_embeds_nothing(conn, load_graph):
    load_graph(*nodes())
    sync_chunks(conn, TEST_PROJECT, CountingEmbedder())
    embedder = CountingEmbedder()

    report = sync_chunks(conn, TEST_PROJECT, embedder)

    assert (report.embedded, report.kept) == (0, 3)
    assert embedder.texts == []


def test_reloading_the_manifest_keeps_embeddings_and_redoes_only_changed_text(conn, load_graph):
    load_graph(*nodes())
    sync_chunks(conn, TEST_PROJECT, CountingEmbedder())
    load_graph(*nodes(description="One row per bus stop."))
    embedder = CountingEmbedder()

    report = sync_chunks(conn, TEST_PROJECT, embedder)

    assert (report.embedded, report.kept) == (1, 2)
    assert embedder.texts == ["dim_stops (model)\nOne row per bus stop."]


def test_chunks_of_removed_nodes_are_deleted(conn, load_graph):
    load_graph(*nodes())
    sync_chunks(conn, TEST_PROJECT, CountingEmbedder())
    load_graph([make_node("fct_trips")], [])

    report = sync_chunks(conn, TEST_PROJECT, CountingEmbedder())

    assert report.deleted == 2
    assert [r["node_id"] for r in stored(conn)] == ["model.lineage_test.fct_trips"]


def test_a_new_model_re_embeds_everything(conn, load_graph):
    load_graph(*nodes())
    sync_chunks(conn, TEST_PROJECT, CountingEmbedder())

    report = sync_chunks(conn, TEST_PROJECT, CountingEmbedder(model="other-model"))

    assert report.embedded == 3
    assert {r["model"] for r in stored(conn)} == {"other-model"}


def test_other_projects_are_untouched(conn, load_graph):
    load_graph(*nodes())
    sync_chunks(conn, TEST_PROJECT, CountingEmbedder())

    report = sync_chunks(conn, "some_other_project", CountingEmbedder())

    assert (report.embedded, report.deleted) == (0, 0)
    assert len(stored(conn)) == 3


def test_dimension_mismatch_is_refused_before_embedding(conn, load_graph):
    load_graph(*nodes())
    embedder = CountingEmbedder(dimensions=16)

    with pytest.raises(EmbeddingError, match="768 dimensions"):
        sync_chunks(conn, TEST_PROJECT, embedder)
    assert embedder.texts == []


def test_project_chunks_reads_nodes_from_the_database(conn, load_graph):
    load_graph(*nodes())

    chunks = project_chunks(conn, TEST_PROJECT)

    assert [(c.node_id, c.kind) for c in chunks] == [
        ("model.lineage_test.dim_stops", "summary"),
        ("model.lineage_test.dim_stops", "sql"),
        ("model.lineage_test.fct_trips", "summary"),
    ]
