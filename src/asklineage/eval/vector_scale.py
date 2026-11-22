"""At what size does an HNSW index pay off? (Phase 5a, ADR 0018)

Real data is too small to tell: on ~3k chunks a full scan beats the index. So this pads a
scratch copy of node_chunks to each size and times the one query an index changes, fetching
the nearest chunks (search/scan.py), with and without HNSW.

Padding: copy 1 is the real chunks; each later copy adds noise to every vector and counts as
new nodes, like the same project loaded again with different text. Rows keep their content so
a full scan reads rows as wide as the real ones. Real embeddings cluster; uniform random
vectors would be the worst case for HNSW and nothing like a dbt project.

Per size: exact (topn) latency, index build time and size, whether the planner picks the index
without being forced, and per ef_search HNSW latency and recall against the exact nearest chunks.
Two recalls: strict (the same chunks) and tie-aware (a chunk as near as the exact k-th, within
TIE_EPSILON, counts; the ann-benchmarks convention). Padded copies sit close together, so strict
recall partly measures which of several equally near chunks came back.
"""

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import psycopg

from asklineage.embed.store import vector_literal
from asklineage.eval.search_eval import elapsed_ms, percentile
from asklineage.search.scan import MAX_EF_SEARCH, Scan, vector_scan

TABLE = "vector_scale_chunks"
INDEX = "vector_scale_chunks_hnsw_idx"
BUILD_MEMORY = "1GB"  # HNSW builds far slower once the graph outgrows maintenance_work_mem
# A parallel build keeps the graph in shared memory, more than Docker's default 64 MB /dev/shm.
BUILD_WORKERS = 0
TIE_EPSILON = 1e-3


def create_sql(dims: int) -> str:
    return f"""
        CREATE UNLOGGED TABLE {TABLE} (
            id bigint PRIMARY KEY,
            node_id text NOT NULL,
            kind text NOT NULL,
            content text NOT NULL,
            embedding vector({int(dims)}) NOT NULL
        )
    """


# The noise subquery names g so it runs once per row, not once for the whole insert.
FILL = f"""
    WITH src AS (
        SELECT row_number() OVER (ORDER BY node_id, kind, ordinal) - 1 AS i,
               node_id, kind, content, l2_normalize(embedding) AS embedding
        FROM node_chunks
        WHERE model = %(model)s AND (%(project)s::text IS NULL OR project_name = %(project)s)
    ), counted AS (
        SELECT count(*) AS n FROM src
    )
    INSERT INTO {TABLE} (id, node_id, kind, content, embedding)
    SELECT g, s.node_id || '#' || (g / c.n + 1), s.kind, s.content,
           CASE WHEN g < c.n THEN s.embedding
                ELSE l2_normalize(s.embedding + (
                    SELECT array_agg((random() - 0.5) * %(amplitude)s)
                    FROM generate_series(1, %(dims)s)
                    WHERE g IS NOT NULL
                )::vector)
           END
    FROM generate_series(0, %(size)s - 1) AS g
    CROSS JOIN counted c
    JOIN src s ON s.i = g %% c.n
"""

NEAREST = f"""
    SELECT id, embedding <=> %(vector)s::vector AS distance FROM {TABLE}
    ORDER BY embedding <=> %(vector)s::vector
    LIMIT %(fetch)s
"""


class NoSourceVectorsError(LookupError):
    pass


@dataclass(frozen=True)
class HnswResult:
    ef_search: int
    p50_ms: float
    p95_ms: float
    recall: float  # share of the exact nearest chunks HNSW also returned, over all queries
    tie_recall: float  # share of HNSW's chunks at most the exact k-th distance (+ TIE_EPSILON)


@dataclass(frozen=True)
class SizeResult:
    size: int
    fill_s: float
    table_mb: float
    index_build_s: float
    index_mb: float
    exact_p50_ms: float
    exact_p95_ms: float
    planner_picks_index: bool  # with no settings forced
    hnsw: tuple[HnswResult, ...]


type Nearest = list[tuple[int, float]]  # (chunk id, distance), nearest first


def noise_amplitude(noise: float, dims: int) -> float:
    """Per-dimension spread of uniform noise whose vector length is about `noise`."""
    return noise * math.sqrt(12 / dims)


def source_count(conn: psycopg.Connection, model: str, project: str | None) -> int:
    row = conn.execute(
        "SELECT count(*) AS n FROM node_chunks "
        "WHERE model = %s AND (%s::text IS NULL OR project_name = %s)",
        (model, project, project),
    ).fetchone()
    return int(row["n"])


def run_vector_scale(
    conn: psycopg.Connection,
    model: str,
    queries: Sequence[Sequence[float]],
    sizes: Sequence[int],
    *,
    project: str | None,
    fetch: int,
    ef_searches: Sequence[int] = (40,),
    noise: float = 0.3,
    dims: int = 768,
    on_size: Callable[[int], None] | None = None,
) -> list[SizeResult]:
    """conn needs dict rows; autocommit keeps each size out of one long transaction.
    Each size builds, measures and drops its own table."""
    if max(ef_searches, default=0) > MAX_EF_SEARCH:
        raise ValueError(f"ef_search above {MAX_EF_SEARCH} is not allowed by pgvector")
    if not source_count(conn, model, project):
        raise NoSourceVectorsError(f"no node chunks embedded with {model!r}; run `embed`")
    vectors = [vector_literal(list(q)) for q in queries]
    conn.execute(f"SET maintenance_work_mem = '{BUILD_MEMORY}'")
    conn.execute(f"SET max_parallel_maintenance_workers = {BUILD_WORKERS}")
    results = []
    for size in sizes:
        if on_size:
            on_size(size)
        conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
        try:
            results.append(
                _measure(conn, model, vectors, size, project, fetch, ef_searches, noise, dims)
            )
        finally:
            conn.execute(f"DROP TABLE IF EXISTS {TABLE}")
    return results


def _measure(
    conn: psycopg.Connection,
    model: str,
    vectors: Sequence[str],
    size: int,
    project: str | None,
    fetch: int,
    ef_searches: Sequence[int],
    noise: float,
    dims: int,
) -> SizeResult:
    started = time.perf_counter()
    conn.execute(create_sql(dims))
    conn.execute(
        FILL,
        {
            "model": model,
            "project": project,
            "size": size,
            "dims": dims,
            "amplitude": noise_amplitude(noise, dims),
        },
    )
    conn.execute(f"ANALYZE {TABLE}")
    fill_s = round(time.perf_counter() - started, 1)

    exact_ms, exact = _time(conn, "topn", vectors, fetch)

    started = time.perf_counter()
    conn.execute(f"CREATE INDEX {INDEX} ON {TABLE} USING hnsw (embedding vector_cosine_ops)")
    index_build_s = round(time.perf_counter() - started, 1)

    hnsw = []
    for ef in ef_searches:
        hnsw_ms, approx = _time(conn, "hnsw", vectors, fetch, ef)
        strict, ties = recalls(exact, approx)
        hnsw.append(
            HnswResult(ef, percentile(hnsw_ms, 0.5), percentile(hnsw_ms, 0.95), strict, ties)
        )

    plan = conn.execute("EXPLAIN " + NEAREST, {"vector": vectors[0], "fetch": fetch}).fetchall()
    sizes = conn.execute(
        "SELECT pg_total_relation_size(%s::regclass) AS t, pg_relation_size(%s::regclass) AS i",
        (TABLE, INDEX),
    ).fetchone()
    return SizeResult(
        size=size,
        fill_s=fill_s,
        table_mb=_mb(sizes["t"] - sizes["i"]),
        index_build_s=index_build_s,
        index_mb=_mb(sizes["i"]),
        exact_p50_ms=percentile(exact_ms, 0.5),
        exact_p95_ms=percentile(exact_ms, 0.95),
        planner_picks_index=any(INDEX in row["QUERY PLAN"] for row in plan),
        hnsw=tuple(hnsw),
    )


def recalls(exact: Sequence[Nearest], approx: Sequence[Nearest]) -> tuple[float, float]:
    """(strict, tie-aware) recall over all queries; see the module docstring."""
    wanted = found = near_enough = 0
    for truth, got in zip(exact, approx, strict=True):
        wanted += len(truth)
        found += len({i for i, _ in truth} & {i for i, _ in got})
        if truth:
            bar = truth[-1][1] + TIE_EPSILON
            near_enough += sum(1 for _, distance in got[: len(truth)] if distance <= bar)
    if not wanted:
        return 1.0, 1.0
    return round(found / wanted, 4), round(near_enough / wanted, 4)


def _time(
    conn: psycopg.Connection,
    scan: Scan,
    vectors: Sequence[str],
    fetch: int,
    ef_search: int | None = None,
) -> tuple[list[float], list[Nearest]]:
    """Times only the query statement; the settings around it are not counted."""
    latencies: list[float] = []
    found: list[Nearest] = []
    with vector_scan(conn, scan, fetch, ef_search):
        conn.execute(NEAREST, {"vector": vectors[0], "fetch": fetch}).fetchall()  # warm-up
        for vector in vectors:
            started = time.perf_counter()
            rows = conn.execute(NEAREST, {"vector": vector, "fetch": fetch}).fetchall()
            latencies.append(elapsed_ms(started))
            found.append([(row["id"], float(row["distance"])) for row in rows])
    return latencies, found


def _mb(size_bytes: int) -> float:
    return round(size_bytes / 1024 / 1024, 1)


def render(results: Sequence[SizeResult], header: str, embed_p95_ms: float | None) -> str:
    lines = [
        header,
        "",
        f"{'chunks':>8} {'fill s':>7} {'table MB':>9} {'index s':>8} {'index MB':>9}  "
        f"{'exact p50':>9} {'exact p95':>9}  planner uses index",
    ]
    for r in results:
        lines.append(
            f"{r.size:>8} {r.fill_s:>7.1f} {r.table_mb:>9.1f} {r.index_build_s:>8.1f} "
            f"{r.index_mb:>9.1f}  {r.exact_p50_ms:>9.1f} {r.exact_p95_ms:>9.1f}  "
            f"{'yes' if r.planner_picks_index else 'no'}"
        )
    lines += [
        "",
        f"{'chunks':>8} {'ef_search':>9}  {'hnsw p50':>8} {'hnsw p95':>8}  "
        f"{'recall':>6} {'tie recall':>10}",
    ]
    for r in results:
        for h in r.hnsw:
            lines.append(
                f"{r.size:>8} {h.ef_search:>9}  {h.p50_ms:>8.1f} {h.p95_ms:>8.1f}  "
                f"{h.recall:>6.3f} {h.tie_recall:>10.3f}"
            )
    if embed_p95_ms is not None:
        over = [r.size for r in results if r.exact_p95_ms > embed_p95_ms]
        lines += [
            "",
            f"embedding a question: p95 {embed_p95_ms:.1f} ms. Exact p95 is above it at: "
            + (", ".join(str(s) for s in over) if over else "no size measured"),
        ]
    return "\n".join(lines) + "\n"
