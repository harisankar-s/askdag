"""Upstream and downstream lineage: a depth-bounded walk over lineage_edges."""

from dataclasses import dataclass
from enum import StrEnum

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

DEFAULT_DEPTH = 5
MAX_DEPTH = 25


class Direction(StrEnum):
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"


@dataclass(frozen=True)
class LineageNode:
    unique_id: str
    name: str
    source_name: str | None
    resource_type: str
    depth: int
    owner: str | None
    owner_source: str | None

    @property
    def display_name(self) -> str:
        return f"{self.source_name}.{self.name}" if self.source_name else self.name


@dataclass(frozen=True)
class Lineage:
    root_id: str
    direction: Direction
    max_depth: int
    nodes: tuple[LineageNode, ...]
    edges: tuple[tuple[str, str], ...]
    truncated: bool


# UNION (not UNION ALL) keeps each (node, hops) pair once, so the walk is bounded by
# nodes x max_depth rows however many paths exist, and the depth cap also ends any cycle.
WALK = sql.SQL("""
    WITH RECURSIVE walk (node_id, hops) AS (
        SELECT {next}, 1 FROM lineage_edges WHERE {start} = %(root)s
        UNION
        SELECT e.{next}, w.hops + 1
        FROM walk w
        JOIN lineage_edges e ON e.{start} = w.node_id
        WHERE w.hops < %(max_depth)s
    )
    SELECT n.unique_id, n.name, n.source_name, n.resource_type, min(w.hops) AS depth,
           n.owner, n.owner_source
    FROM walk w
    JOIN nodes n ON n.unique_id = w.node_id
    WHERE w.node_id <> %(root)s
    GROUP BY n.unique_id
    ORDER BY depth, n.name, n.unique_id
""")

HAS_MORE = sql.SQL("""
    SELECT EXISTS (
        SELECT 1 FROM lineage_edges
        WHERE {start} = ANY(%(frontier)s) AND NOT ({next} = ANY(%(seen)s))
    ) AS has_more
""")

EDGES_BETWEEN = """
    SELECT parent_id, child_id FROM lineage_edges
    WHERE parent_id = ANY(%(ids)s) AND child_id = ANY(%(ids)s)
    ORDER BY parent_id, child_id
"""

PARENT, CHILD = sql.Identifier("parent_id"), sql.Identifier("child_id")
WALK_COLUMNS = {
    Direction.DOWNSTREAM: {"start": PARENT, "next": CHILD},
    Direction.UPSTREAM: {"start": CHILD, "next": PARENT},
}


def lineage(
    conn: psycopg.Connection,
    root_id: str,
    direction: Direction,
    max_depth: int = DEFAULT_DEPTH,
) -> Lineage:
    if not 1 <= max_depth <= MAX_DEPTH:
        raise ValueError(f"max_depth must be between 1 and {MAX_DEPTH}, got {max_depth}")
    columns = WALK_COLUMNS[direction]

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT 1 FROM nodes WHERE unique_id = %s", (root_id,))
        if cur.fetchone() is None:
            raise LookupError(f"No node with unique_id {root_id!r}")

        cur.execute(WALK.format(**columns), {"root": root_id, "max_depth": max_depth})
        nodes = tuple(LineageNode(**row) for row in cur.fetchall())
        seen = [root_id, *(node.unique_id for node in nodes)]

        cur.execute(EDGES_BETWEEN, {"ids": seen})
        edges = tuple((row["parent_id"], row["child_id"]) for row in cur.fetchall())

        # Only nodes at the cap can have unvisited neighbours; anything closer was fully expanded.
        frontier = [node.unique_id for node in nodes if node.depth == max_depth]
        truncated = False
        if frontier:
            cur.execute(HAS_MORE.format(**columns), {"frontier": frontier, "seen": seen})
            truncated = cur.fetchone()["has_more"]

    return Lineage(root_id, direction, max_depth, nodes, edges, truncated)
