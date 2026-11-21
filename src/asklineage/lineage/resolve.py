"""Turn a typed node name into exactly one node, or say why that isn't possible."""

from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row

MAX_SUGGESTIONS = 5

EXACT = """
    SELECT unique_id, name, resource_type, source_name, 1.0::float8 AS score
    FROM nodes
    WHERE (%(project)s::text IS NULL OR project_name = %(project)s)
      AND (
          unique_id = %(q)s
          OR lower(name) = lower(%(q)s)
          OR lower(source_name || '.' || name) = lower(%(q)s)
      )
    ORDER BY unique_id
"""

# `%` and `<%` use pg_trgm's default thresholds (0.3 similarity, 0.6 word similarity)
# and can use the trigram index; `%%` is psycopg's escape for a literal `%`.
FUZZY = """
    SELECT unique_id, name, resource_type, source_name,
           greatest(similarity(name, %(q)s), word_similarity(%(q)s, name))::float8 AS score
    FROM nodes
    WHERE (%(project)s::text IS NULL OR project_name = %(project)s)
      AND (name %% %(q)s OR %(q)s <%% name)
    ORDER BY score DESC, name, unique_id
    LIMIT %(limit)s
"""


DISPLAY_NAMES = """
    SELECT CASE WHEN source_name IS NULL THEN name ELSE source_name || '.' || name END AS name
    FROM nodes
    WHERE %(project)s::text IS NULL OR project_name = %(project)s
"""


@dataclass(frozen=True)
class Candidate:
    unique_id: str
    name: str
    resource_type: str
    source_name: str | None
    score: float

    @property
    def display_name(self) -> str:
        return f"{self.source_name}.{self.name}" if self.source_name else self.name


@dataclass(frozen=True)
class Resolved:
    node: Candidate


@dataclass(frozen=True)
class Ambiguous:
    query: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class NotFound:
    query: str
    suggestions: tuple[Candidate, ...]


type Resolution = Resolved | Ambiguous | NotFound


def resolve_node(conn: psycopg.Connection, query: str, project: str | None = None) -> Resolution:
    """Resolve on an exact id or name only; fuzzy matches come back as suggestions, never picks."""
    q = query.strip()
    if not q:
        return NotFound(query, ())

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(EXACT, {"q": q, "project": project})
        exact = [Candidate(**row) for row in cur.fetchall()]
        by_id = [c for c in exact if c.unique_id == q]
        if by_id:
            return Resolved(by_id[0])
        if len(exact) == 1:
            return Resolved(exact[0])
        if exact:
            return Ambiguous(q, tuple(exact))

        cur.execute(FUZZY, {"q": q, "project": project, "limit": MAX_SUGGESTIONS})
        return NotFound(q, tuple(Candidate(**row) for row in cur.fetchall()))


def node_display_names(conn: psycopg.Connection, project: str | None = None) -> frozenset[str]:
    """Every node's display name, so a summary that mentions a real node unmarked can be caught."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(DISPLAY_NAMES, {"project": project})
        return frozenset(row["name"] for row in cur.fetchall())
