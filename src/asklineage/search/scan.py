"""How a vector query finds its nearest chunks (Phase 5a).

- exact: score every chunk, keep each node's best. No index can serve this shape.
- topn: fetch the n nearest chunks (n = k x overfetch), then keep each node's best. Exact, but
  a node with many close chunks can crowd others out of the n.
- hnsw: the same SQL as topn, served by the HNSW index: approximate on top of the crowding.

topn and hnsw differ only in the plan, so the eval separates the two losses.
"""

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Literal, get_args

import psycopg

Scan = Literal["exact", "topn", "hnsw"]
SCANS: tuple[Scan, ...] = get_args(Scan)
MAX_EF_SEARCH = 1000  # pgvector's upper bound


def fetch_size(k: int, overfetch: int) -> int:
    return k * overfetch


def variants(scans: Sequence[Scan], overfetches: Sequence[int]) -> list[tuple[Scan, int]]:
    """exact once, since it ignores overfetch; topn and hnsw at each overfetch."""
    first = overfetches[0]
    return [
        pair
        for scan in dict.fromkeys(scans)
        for pair in ([(scan, first)] if scan == "exact" else [(scan, o) for o in overfetches])
    ]


@contextmanager
def vector_scan(
    conn: psycopg.Connection, scan: Scan, fetch: int, ef_search: int | None = None
) -> Iterator[None]:
    """Plan settings for one query, undone afterwards (on error the savepoint undoes them).
    ef_search: HNSW's candidate list; None uses the fetch, never below pgvector's 40."""
    if scan == "exact":
        yield
        return
    if scan == "topn":
        settings = {"enable_indexscan": "off"}
    else:
        # On small tables any scan plus a sort looks cheaper, which would measure topn twice;
        # only the vector index returns rows in distance order, so no sort forces it.
        # ef_search caps how many rows the index returns, so it must cover the fetch.
        settings = {
            "enable_seqscan": "off",
            "enable_sort": "off",
            "hnsw.ef_search": str(min(max(ef_search or fetch, 40), MAX_EF_SEARCH)),
            "hnsw.iterative_scan": "relaxed_order",
        }
    names = list(settings)
    # One statement each way: every round trip here would count against the scan being timed.
    with conn.transaction():
        conn.execute(
            "SELECT set_config(n, v, true) FROM unnest(%s::text[], %s::text[]) AS s(n, v)",
            (names, list(settings.values())),
        )
        yield
        conn.execute(
            "SELECT set_config(name, reset_val, true) FROM pg_settings WHERE name = ANY(%s)",
            (names,),
        )
