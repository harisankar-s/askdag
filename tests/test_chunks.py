import json
from dataclasses import replace
from pathlib import Path

from helpers import make_node

from asklineage.embed.chunks import CHUNK_CHARS, chunk_node, header
from asklineage.ingest.manifest import Column, parse_manifest

FIXTURE = Path(__file__).parent / "fixtures" / "mini_manifest.json"


def column(node_id: str, name: str, description: str = "") -> Column:
    return Column(node_id=node_id, name=name, description=description, data_type=None)


def test_header_names_the_node_its_type_and_folder():
    node = replace(make_node("dim_stops"), file_path="models/mart/gtfs/dim_stops.sql")
    source = make_node("stops", resource_type="source", source_name="gtfs")

    assert header(node) == "dim_stops (model, models/mart/gtfs)"
    assert header(source) == "gtfs.stops (source)"


def test_node_without_text_still_gets_a_summary_chunk():
    chunks = chunk_node(make_node("fct_breakdowns"), [])

    assert [(c.kind, c.ordinal, c.content) for c in chunks] == [
        ("summary", 0, "fct_breakdowns (model)")
    ]


def test_each_kind_gets_its_own_chunks_and_every_chunk_starts_with_the_header():
    node = replace(
        make_node("dim_stops"),
        description="  One row per\nstop. ",
        raw_code="select stop_id,\n    stop_lat from {{ ref('stg_stops') }}",
    )
    cols = [column(node.unique_id, "stop_id", "The stop."), column(node.unique_id, "stop_lat")]

    chunks = chunk_node(node, cols)

    assert [c.kind for c in chunks] == ["summary", "columns", "sql"]
    assert chunks[0].content == "dim_stops (model)\nOne row per stop."
    assert chunks[1].content == "dim_stops (model)\nstop id: The stop.\nstop lat"
    assert (
        chunks[2].content
        == "dim_stops (model)\nselect stop id, stop lat from {{ ref('stg stops') }}"
    )


def test_long_text_is_split_under_the_limit_without_losing_words():
    words = [f"word{i}" for i in range(2000)]
    node = replace(make_node("big"), description=" ".join(words))

    chunks = chunk_node(node, [])

    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(len(c.content) <= CHUNK_CHARS for c in chunks)
    rebuilt = " ".join(c.content.split("\n", 1)[1] for c in chunks).split()
    assert rebuilt == words


def test_an_oversized_single_item_is_cut():
    node = make_node("wide")
    chunks = chunk_node(node, [column(node.unique_id, "c", "x" * 5000)], limit=200)

    assert all(len(c.content) <= 200 for c in chunks if c.kind == "columns")
    assert sum(1 for c in chunks if c.kind == "columns") > 1


def test_sha_follows_content():
    (first,) = chunk_node(make_node("a"), [])
    (again,) = chunk_node(make_node("a"), [])
    (other,) = chunk_node(replace(make_node("a"), description="changed"), [])

    assert first.sha == again.sha != other.sha


def test_mini_manifest_chunks():
    parsed = parse_manifest(json.loads(FIXTURE.read_text()))
    for node in parsed.nodes:
        cols = [c for c in parsed.columns if c.node_id == node.unique_id]
        chunks = chunk_node(node, cols)
        assert chunks[0].kind == "summary"
        assert all(c.content.startswith(header(node)) for c in chunks)


def test_snake_case_is_split_in_columns_and_sql_but_not_in_the_header():
    node = replace(
        make_node("fct_breakdowns"),
        description="See major_mechanical_failures.",
        raw_code="select major_mechanical_failures, _valid_from",
    )
    summary, columns, sql = chunk_node(node, [column(node.unique_id, "occupancy_status")])

    assert summary.content == "fct_breakdowns (model)\nSee major_mechanical_failures."
    assert columns.content == "fct_breakdowns (model)\noccupancy status"
    assert sql.content == "fct_breakdowns (model)\nselect major mechanical failures, _valid from"
