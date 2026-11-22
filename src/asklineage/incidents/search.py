"""Find incidents for a question, five ways, so step 3c can compare them (ADR 0015).

- names: nodes the question names exactly, and their upstream, joined to each incident's
  detected node links. Precise, but blind to write-ups that don't name the node.
- keyword: Postgres full-text over chunks, any word may match (OR).
- vector: cosine over chunk embeddings.
- hybrid: keyword and vector merged by reciprocal rank fusion, which needs no score scaling.
- routed: names when the question names a node, hybrid for the rest of the list.

A document ranks by its best chunk. Scores are only comparable within one method.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, get_args

import psycopg
from psycopg.rows import dict_row, tuple_row

from asklineage.embed.base import EmbeddingProvider, index_name
from asklineage.embed.store import vector_literal
from asklineage.incidents.chunks import detect_nodes, spellings
from asklineage.incidents.store import ALL_DISPLAY_NAMES, DISPLAY_NAMES

Method = Literal["names", "keyword", "vector", "hybrid", "routed"]
METHODS: tuple[Method, ...] = get_args(Method)
UPSTREAM_HOPS = 4
CANDIDATES = 30
RRF_K = 60  # the usual constant; larger flattens the difference between ranks

NAMES = """
    WITH RECURSIVE up(node_id, hops) AS (
        SELECT unnest(%(ids)s::text[]), 0
        UNION
        SELECT e.parent_id, u.hops + 1
        FROM lineage_edges e JOIN up u ON e.child_id = u.node_id
        WHERE u.hops < %(hops)s
    ), nearest AS (
        SELECT node_id, min(hops) AS hops FROM up GROUP BY node_id
    )
    SELECT d.project_name AS project, d.doc_id, d.title, 1.0 / (1 + min(n.hops)) AS score,
           NULL::text AS section, NULL::text AS content
    FROM documents d JOIN nearest n ON n.node_id = ANY(d.node_ids)
    WHERE %(project)s::text IS NULL OR d.project_name = %(project)s
    GROUP BY d.project_name, d.doc_id, d.title, d.doc_date
    ORDER BY score DESC, d.doc_date DESC, d.doc_id
    LIMIT %(k)s
"""

KEYWORD = """
    WITH q AS (
        SELECT CASE WHEN t.text = '' THEN NULL ELSE t.text::tsquery END AS query
        FROM (SELECT replace(plainto_tsquery('english', %(question)s)::text, '&', '|')
              AS text) t
    ), best AS (
        SELECT DISTINCT ON (c.doc_id)
            c.project_name, c.doc_id, c.section, c.content,
            ts_rank_cd(c.search, q.query) AS score
        FROM doc_chunks c, q
        WHERE (%(project)s::text IS NULL OR c.project_name = %(project)s)
          AND c.search @@ q.query
        ORDER BY c.doc_id, score DESC, c.project_name, c.ordinal
    )
    SELECT b.project_name AS project, b.doc_id, d.title, b.score, b.section, b.content
    FROM best b JOIN documents d ON d.project_name = b.project_name AND d.doc_id = b.doc_id
    ORDER BY b.score DESC, b.doc_id
    LIMIT %(k)s
"""

VECTOR = """
    WITH best AS (
        SELECT DISTINCT ON (c.doc_id)
            c.project_name, c.doc_id, c.section, c.content,
            c.embedding <=> %(vector)s::vector AS distance
        FROM doc_chunks c
        WHERE (%(project)s::text IS NULL OR c.project_name = %(project)s)
          AND c.model = %(model)s
        ORDER BY c.doc_id, distance, c.project_name, c.ordinal
    )
    SELECT b.project_name AS project, b.doc_id, d.title, 1 - b.distance AS score, b.section,
           b.content
    FROM best b JOIN documents d ON d.project_name = b.project_name AND d.doc_id = b.doc_id
    ORDER BY b.distance, b.doc_id
    LIMIT %(k)s
"""


class NoVectorsError(LookupError):
    pass


@dataclass(frozen=True)
class DocHit:
    project: str
    doc_id: str
    title: str
    score: float
    section: str | None  # the best-matching chunk's section; None for name links
    content: str | None


def named_nodes(conn: psycopg.Connection, project: str | None, question: str) -> tuple[str, ...]:
    return detect_nodes(question, known_spellings(conn, project))


def known_spellings(conn: psycopg.Connection, project: str | None) -> dict[str, str]:
    with conn.cursor(row_factory=tuple_row) as cur:
        if project is None:
            return spellings(cur.execute(ALL_DISPLAY_NAMES).fetchall())
        return spellings(cur.execute(DISPLAY_NAMES, (project,)).fetchall())


def find_incidents(
    conn: psycopg.Connection,
    question: str,
    method: Method,
    embedder: EmbeddingProvider | None,
    *,
    project: str | None,
    k: int = 10,
) -> tuple[DocHit, ...]:
    """project None searches every loaded project."""
    params = {"project": project, "question": question, "k": k}
    if method == "names":
        ids = named_nodes(conn, project, question)
        return _rows(conn, NAMES, params | {"ids": list(ids), "hops": UPSTREAM_HOPS}) if ids else ()
    if method == "keyword":
        return _rows(conn, KEYWORD, params)
    if method == "vector":
        return _vector(conn, question, embedder, params)
    if method == "hybrid":
        return _hybrid(conn, question, embedder, params, k)
    if method == "routed":
        named = find_incidents(conn, question, "names", embedder, project=project, k=k)
        seen = {(hit.project, hit.doc_id) for hit in named}
        rest = _hybrid(conn, question, embedder, params, k)
        return (*named, *(h for h in rest if (h.project, h.doc_id) not in seen))[:k]
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def _vector(
    conn: psycopg.Connection, question: str, embedder: EmbeddingProvider | None, params: dict
) -> tuple[DocHit, ...]:
    if embedder is None:
        raise NoVectorsError("vector search needs an embedding model (EMBED_PROVIDER)")
    tag = index_name(embedder)
    with conn.cursor(row_factory=tuple_row) as cur:
        (present,) = cur.execute(
            "SELECT EXISTS (SELECT 1 FROM doc_chunks WHERE model = %s "
            "AND (%s::text IS NULL OR project_name = %s))",
            (tag, params["project"], params["project"]),
        ).fetchone()
    if not present:
        raise NoVectorsError(f"no incident chunks embedded with {tag!r}; run `load-docs`")
    vector = vector_literal(embedder.embed([question], kind="query").vectors[0])
    return _rows(conn, VECTOR, params | {"vector": vector, "model": tag})


def _hybrid(
    conn: psycopg.Connection,
    question: str,
    embedder: EmbeddingProvider | None,
    params: dict,
    k: int,
) -> tuple[DocHit, ...]:
    lists = [
        _rows(conn, KEYWORD, params | {"k": CANDIDATES}),
        _vector(conn, question, embedder, params | {"k": CANDIDATES}),
    ]
    return tuple(rrf(lists)[:k])


def rrf(lists: Sequence[Sequence[DocHit]], k: int = RRF_K) -> list[DocHit]:
    """Reciprocal rank fusion: each list adds 1 / (k + rank). The section and text shown are
    from the list where the document ranked best."""
    scores: dict[tuple[str, str], float] = {}
    best: dict[tuple[str, str], tuple[int, DocHit]] = {}
    for hits in lists:
        for rank, hit in enumerate(hits, start=1):
            key = (hit.project, hit.doc_id)
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            if key not in best or rank < best[key][0]:
                best[key] = (rank, hit)
    order = sorted(scores, key=lambda key: (-scores[key], key[1], key[0]))
    return [replace(best[key][1], score=scores[key]) for key in order]


def _rows(conn: psycopg.Connection, sql: str, params: dict) -> tuple[DocHit, ...]:
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(sql, params).fetchall()
    return tuple(DocHit(**{**row, "score": float(row["score"])}) for row in rows)


SECTIONS = """
    SELECT project_name, doc_id, section, content FROM doc_chunks
    WHERE (project_name, doc_id) IN (SELECT * FROM unnest(%(projects)s::text[], %(ids)s::text[]))
      AND section = ANY(%(sections)s)
    ORDER BY project_name, doc_id, ordinal
"""


def incident_sections(
    conn: psycopg.Connection, hits: Sequence[DocHit], sections: Sequence[str]
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """(project, doc_id) -> [(section, chunk text)] for the given sections, in document order."""
    found: dict[tuple[str, str], list[tuple[str, str]]] = {(h.project, h.doc_id): [] for h in hits}
    if not hits:
        return found
    params = {
        "projects": [h.project for h in hits],
        "ids": [h.doc_id for h in hits],
        "sections": list(sections),
    }
    with conn.cursor(row_factory=tuple_row) as cur:
        for project, doc_id, section, content in cur.execute(SECTIONS, params).fetchall():
            found[(project, doc_id)].append((section, content))
    return found
