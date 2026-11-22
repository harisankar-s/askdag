from datetime import date
from pathlib import Path

import pytest
from helpers import TEST_PROJECT, make_node

from asklineage.corpus.docs import Corpus, Document
from asklineage.embed.base import EmbeddingError
from asklineage.embed.hashing import HashingEmbedder
from asklineage.incidents.chunks import chunk_incident, detect_nodes, sections, spellings
from asklineage.incidents.store import index_incidents

BODY = """## Summary

dim_stops lost rows after raw_gtfs.stops was late.

## Root cause

The   source stopped.

## Fix

Re-ran the load.
"""


def incident(doc_id: str = "inc-001", body: str = BODY, **meta) -> Document:
    return Document(
        id=doc_id,
        type="incident",
        title="Stops went missing",
        date=date(2025, 5, 7),
        mention="exact",
        node_roles=(("dim_stops", "symptom"),),
        body=body,
        path=Path("corpus/test/incidents") / f"{doc_id}.md",
        meta={"severity": "sev2", **meta},
    )


def test_sections_skip_text_before_the_first_heading_and_collapse_space():
    assert sections("intro\n## A\n\nx   y\n## B\n\n") == [("A", "x y")]


def test_each_section_is_a_chunk_with_the_incident_header():
    chunks = chunk_incident(incident())
    assert [c.section for c in chunks] == ["Summary", "Root cause", "Fix"]
    assert [c.ordinal for c in chunks] == [0, 1, 2]
    assert chunks[2].content == (
        "incident inc-001 · 2025-05-07 · sev2 · Stops went missing\nFix: Re-ran the load."
    )


def test_a_long_section_is_split_and_keeps_the_header():
    chunks = chunk_incident(incident(body="## Fix\n\n" + "word " * 400), limit=300)
    assert len(chunks) > 1
    assert all(c.content.startswith("incident inc-001") and len(c.content) <= 300 for c in chunks)
    assert {c.section for c in chunks} == {"Fix"}


KNOWN = spellings(
    [
        ("dim_stops", "model.p.dim_stops"),
        ("dim_stops_daily", "model.p.dim_stops_daily"),
        ("raw_gtfs.stops", "source.p.raw_gtfs.stops"),
        ("raw_gtfs.download_outcomes", "source.p.raw_gtfs.download_outcomes"),
        ("raw_a.shared_table", "source.p.raw_a.shared_table"),
        ("raw_b.shared_table", "source.p.raw_b.shared_table"),
    ]
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("dim_stops lost rows", ("model.p.dim_stops",)),
        ("dim_stops_daily only", ("model.p.dim_stops_daily",)),
        ("Is DIM_STOPS late? Then dim_stops.", ("model.p.dim_stops",)),
        ("raw_gtfs.stops was late", ("source.p.raw_gtfs.stops",)),
        ("the stops were late", ()),
        ("download_outcomes was empty", ("source.p.raw_gtfs.download_outcomes",)),
        ("raw_a.shared_table broke", ("source.p.raw_a.shared_table",)),
        ("shared_table broke", ()),
        ("raw_gtfs.stops then dim_stops", ("source.p.raw_gtfs.stops", "model.p.dim_stops")),
    ],
)
def test_detect_nodes_finds_exact_names_only(text, expected):
    assert detect_nodes(text, KNOWN) == expected


def stored(conn):
    docs = conn.execute(
        "SELECT doc_id, source, url, severity, node_ids FROM documents "
        "WHERE project_name = %s ORDER BY doc_id",
        (TEST_PROJECT,),
    ).fetchall()
    chunks = conn.execute(
        "SELECT doc_id, ordinal, model, embedding IS NOT NULL AS has_vector, "
        "search @@ plainto_tsquery('english', 'load') AS matches FROM doc_chunks "
        "WHERE project_name = %s ORDER BY doc_id, ordinal",
        (TEST_PROJECT,),
    ).fetchall()
    return docs, chunks


def corpus(*docs: Document) -> Corpus:
    return Corpus(TEST_PROJECT, 1, docs)


def test_index_links_detected_nodes_and_embeds_sections(conn, load_graph):
    stops = make_node("dim_stops")
    load_graph([stops, make_node("stops", "source", "raw_gtfs")], [])

    report = index_incidents(conn, corpus(incident()), HashingEmbedder(768))

    docs, chunks = stored(conn)
    assert docs == [
        {
            "doc_id": "inc-001",
            "source": "markdown",
            "url": "corpus/test/incidents/inc-001.md",
            "severity": "sev2",
            "node_ids": [stops.unique_id, f"source.{TEST_PROJECT}.raw_gtfs.stops"],
        }
    ]
    assert [c["has_vector"] for c in chunks] == [True, True, True]
    assert [c["matches"] for c in chunks] == [False, False, True]
    assert (report.documents, report.chunks, report.linked, report.links) == (1, 3, 1, 2)


def test_without_an_embedder_chunks_are_kept_for_keyword_search(conn, load_graph):
    load_graph([make_node("dim_stops")], [])

    report = index_incidents(conn, corpus(incident()), None)

    _, chunks = stored(conn)
    assert [(c["model"], c["has_vector"]) for c in chunks] == [(None, False)] * 3
    assert report.model is None


def test_a_reload_replaces_the_project(conn, load_graph):
    load_graph([make_node("dim_stops")], [])
    index_incidents(conn, corpus(incident("inc-001"), incident("inc-002")), None)

    index_incidents(conn, corpus(incident("inc-002")), None)

    docs, chunks = stored(conn)
    assert [d["doc_id"] for d in docs] == ["inc-002"]
    assert {c["doc_id"] for c in chunks} == {"inc-002"}


def test_a_dimension_mismatch_leaves_the_old_index(conn, load_graph):
    load_graph([make_node("dim_stops")], [])
    index_incidents(conn, corpus(incident()), None)

    with pytest.raises(EmbeddingError, match="768 dimensions"):
        index_incidents(conn, corpus(incident("inc-009")), HashingEmbedder(16))

    docs, _ = stored(conn)
    assert [d["doc_id"] for d in docs] == ["inc-001"]


def test_indexing_needs_loaded_nodes(conn):
    with pytest.raises(LookupError, match="load-manifest"):
        index_incidents(conn, Corpus("no_such_project", 1, (incident(),)), None)
