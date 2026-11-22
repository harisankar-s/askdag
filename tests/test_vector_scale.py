import math
from dataclasses import replace

import pytest
from helpers import TEST_PROJECT, make_node

from asklineage.embed.base import index_name
from asklineage.embed.hashing import HashingEmbedder
from asklineage.embed.store import sync_chunks
from asklineage.eval.vector_scale import (
    FILL,
    TABLE,
    NoSourceVectorsError,
    create_sql,
    noise_amplitude,
    recalls,
    render,
    run_vector_scale,
)


@pytest.fixture
def chunks(conn, load_graph):
    nodes = [
        replace(make_node("dim_stops"), description="One row per bus stop location."),
        replace(make_node("fct_fares"), description="One row per card payment."),
    ]
    load_graph(nodes, [])
    embedder = HashingEmbedder()
    sync_chunks(conn, TEST_PROJECT, embedder)
    count = conn.execute(
        "SELECT count(*) AS n FROM node_chunks WHERE project_name = %s", (TEST_PROJECT,)
    ).fetchone()["n"]
    return embedder, count


def test_noise_of_the_given_length_spreads_evenly_over_dimensions():
    amplitude = noise_amplitude(0.3, 768)

    assert math.isclose(amplitude * math.sqrt(768 / 12), 0.3)


def test_padding_copies_the_chunks_once_exactly_then_with_noise_as_new_nodes(conn, chunks):
    embedder, count = chunks
    size = count * 3
    conn.execute(create_sql(768))
    conn.execute(
        FILL,
        {
            "model": index_name(embedder),
            "project": TEST_PROJECT,
            "size": size,
            "dims": 768,
            "amplitude": noise_amplitude(0.3, 768),
        },
    )

    rows = conn.execute(
        f"SELECT p.node_id, vector_norm(p.embedding) AS norm, "
        f"p.embedding <=> c.embedding AS moved "
        f"FROM {TABLE} p JOIN node_chunks c ON c.node_id = split_part(p.node_id, '#', 1) "
        f"AND c.kind = p.kind AND c.content = p.content"
    ).fetchall()

    assert len(rows) == size
    assert {r["node_id"].split("#")[1] for r in rows} == {"1", "2", "3"}
    (apart,) = (
        conn.execute(
            f"SELECT min(a.embedding <=> b.embedding) AS d FROM {TABLE} a JOIN {TABLE} b "
            f"ON a.kind = b.kind AND a.content = b.content "
            f"AND a.node_id LIKE '%#2' AND b.node_id LIKE '%#3' "
            f"AND split_part(a.node_id, '#', 1) = split_part(b.node_id, '#', 1)"
        )
        .fetchone()
        .values()
    )
    assert apart > 1e-4  # each copy gets its own noise
    assert all(math.isclose(r["norm"], 1, rel_tol=1e-4) for r in rows)
    first = [r["moved"] for r in rows if r["node_id"].endswith("#1")]
    noisy = [r["moved"] for r in rows if not r["node_id"].endswith("#1")]
    assert max(first) < 1e-6
    assert min(noisy) > 1e-4


def test_small_sizes_give_full_recall_and_leave_no_table(conn, chunks):
    embedder, count = chunks
    queries = embedder.embed(["bus stop", "card payment", "fares"], kind="query").vectors

    results = run_vector_scale(
        conn,
        index_name(embedder),
        queries,
        [count, count * 4],
        project=TEST_PROJECT,
        fetch=4,
        ef_searches=[40, 100],
    )

    assert [r.size for r in results] == [count, count * 4]
    assert [[h.ef_search for h in r.hnsw] for r in results] == [[40, 100], [40, 100]]
    assert all(h.recall == h.tie_recall == 1.0 for r in results for h in r.hnsw)
    assert all(h.p95_ms >= h.p50_ms for r in results for h in r.hnsw)
    assert conn.execute("SELECT to_regclass(%s) AS t", (TABLE,)).fetchone()["t"] is None
    assert "planner uses index" in render(results, "header", 5.0)


def test_tie_aware_recall_counts_an_equally_near_chunk():
    exact = [[(1, 0.10), (2, 0.20)]]
    approx = [[(1, 0.10), (3, 0.2005)]]  # 3 ties with 2, the exact second

    assert recalls(exact, approx) == (0.5, 1.0)
    assert recalls(exact, [[(1, 0.10), (4, 0.30)]]) == (0.5, 0.5)


def test_a_model_with_no_chunks_is_refused(conn):
    with pytest.raises(NoSourceVectorsError, match="run `embed`"):
        run_vector_scale(conn, "no-such-model", [[1.0] * 768], [10], project=None, fetch=4)


def test_ef_search_above_pgvectors_limit_is_refused(conn):
    with pytest.raises(ValueError, match="1000"):
        run_vector_scale(conn, "m", [[1.0] * 768], [10], project=None, fetch=4, ef_searches=[2000])
