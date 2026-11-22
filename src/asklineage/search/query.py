"""Find nodes by what a question describes: by meaning when embeddings exist, else by keywords.

The result always says which path ran. Scores from the two paths are on different scales and
are never compared; there is no "nothing found" threshold yet (the search eval sets one).

Lineage-aware ranking (semantic only, when preferred paths are given): a staging model or source
that matches well often sits one or two hops above the mart a user should query. Such a node
lifts its best preferred descendants to its own score, and non-preferred nodes are scored a
little lower, so the mart ranks first and the result says which node it came via.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fnmatch import fnmatch
from typing import Literal

import psycopg
from psycopg.rows import dict_row

from asklineage.embed.base import EmbeddingProvider, index_name
from asklineage.embed.chunks import CHUNK_KINDS, ChunkKind
from asklineage.embed.store import vector_literal
from asklineage.search.scan import Scan, fetch_size, vector_scan

type Mode = Literal["semantic", "keyword"]

SEMANTIC = """
    WITH best AS (
        SELECT DISTINCT ON (c.node_id)
            c.node_id, c.kind, c.content, c.embedding <=> %(vector)s::vector AS distance
        FROM node_chunks c
        WHERE c.model = %(model)s AND c.kind = ANY(%(kinds)s)
          AND (%(project)s::text IS NULL OR c.project_name = %(project)s)
          AND (%(ids)s::text[] IS NULL OR c.node_id = ANY(%(ids)s))
        ORDER BY c.node_id, distance
    )
    SELECT n.unique_id, n.name, n.source_name, n.resource_type, n.owner, n.owner_source,
           n.file_path, b.kind, b.content, 1 - b.distance AS score
    FROM best b JOIN nodes n ON n.unique_id = b.node_id
    ORDER BY b.distance, n.unique_id
    LIMIT %(k)s
"""

# topn and hnsw (search/scan.py): an index can serve only ORDER BY distance LIMIT n.
SEMANTIC_NEAREST = """
    WITH near AS (
        SELECT c.node_id, c.kind, c.content, c.embedding <=> %(vector)s::vector AS distance
        FROM node_chunks c
        WHERE c.model = %(model)s AND c.kind = ANY(%(kinds)s)
          AND (%(project)s::text IS NULL OR c.project_name = %(project)s)
        ORDER BY c.embedding <=> %(vector)s::vector
        LIMIT %(fetch)s
    ), best AS (
        SELECT DISTINCT ON (node_id) node_id, kind, content, distance
        FROM near
        ORDER BY node_id, distance
    )
    SELECT n.unique_id, n.name, n.source_name, n.resource_type, n.owner, n.owner_source,
           n.file_path, b.kind, b.content, 1 - b.distance AS score
    FROM best b JOIN nodes n ON n.unique_id = b.node_id
    ORDER BY b.distance, n.unique_id
    LIMIT %(k)s
"""

# Any word may match (OR), since a question rarely repeats every word of a description.
KEYWORD = """
    WITH docs AS (
        SELECT n.unique_id, n.name, n.source_name, n.resource_type, n.owner, n.owner_source,
               n.file_path,
               concat_ws(' ', replace(n.name, '_', ' '), n.source_name, n.description) AS summary,
               (SELECT string_agg(replace(c.name, '_', ' ') || ' ' || c.description, ' ')
                FROM node_columns c WHERE c.node_id = n.unique_id) AS columns,
               n.raw_code AS sql
        FROM nodes n
        WHERE %(project)s::text IS NULL OR n.project_name = %(project)s
    ), scored AS (
        SELECT d.*, ts_rank_cd(
                   to_tsvector('english', concat_ws(' ',
                       d.summary,
                       CASE WHEN 'columns' = ANY(%(kinds)s) THEN d.columns END,
                       CASE WHEN 'sql' = ANY(%(kinds)s) THEN d.sql END)),
                   q.query) AS score
        FROM docs d,
             LATERAL (SELECT replace(plainto_tsquery('english', %(question)s)::text, '&', '|')
                      AS text) t,
             LATERAL (SELECT CASE WHEN t.text = '' THEN NULL ELSE t.text::tsquery END
                      AS query) q
    )
    SELECT unique_id, name, source_name, resource_type, owner, owner_source, file_path,
           'summary' AS kind, summary AS content, score
    FROM scored WHERE score > 0
    ORDER BY score DESC, unique_id
    LIMIT %(k)s
"""

DOWNSTREAM_2 = """
    SELECT e1.parent_id AS up, e1.child_id AS down
    FROM lineage_edges e1 WHERE e1.parent_id = ANY(%(ids)s)
    UNION
    SELECT e1.parent_id, e2.child_id
    FROM lineage_edges e1 JOIN lineage_edges e2 ON e2.parent_id = e1.child_id
    WHERE e1.parent_id = ANY(%(ids)s)
"""

CANDIDATES = 30
# Non-preferred nodes keep 98% of their score, so a lifted mart outranks the node it came via.
LAYER_PENALTY = 0.98
# One upstream node lifts at most this many descendants, so a source feeding dozens of marts
# cannot fill the results on its own.
MAX_LIFTS = 3


@dataclass(frozen=True)
class Hit:
    unique_id: str
    name: str
    source_name: str | None
    resource_type: str
    owner: str | None
    owner_source: str | None
    file_path: str | None
    score: float
    kind: str
    content: str
    via: str | None = None

    @property
    def display_name(self) -> str:
        return f"{self.source_name}.{self.name}" if self.source_name else self.name


@dataclass(frozen=True)
class SearchResult:
    mode: Mode
    hits: tuple[Hit, ...]
    model: str | None = None
    note: str | None = None


def has_embeddings(conn: psycopg.Connection, model: str, project: str | None) -> bool:
    row = conn.execute(
        "SELECT EXISTS (SELECT 1 FROM node_chunks WHERE model = %s "
        "AND (%s::text IS NULL OR project_name = %s))",
        (model, project, project),
    ).fetchone()
    return bool(row[0] if isinstance(row, tuple) else next(iter(row.values())))


def search(
    conn: psycopg.Connection,
    question: str,
    embedder: EmbeddingProvider | None,
    *,
    project: str | None = None,
    kinds: Sequence[ChunkKind] = CHUNK_KINDS,
    k: int = 10,
    keyword_only: bool = False,
    prefer: Sequence[str] = (),
    scan: Scan = "topn",
    overfetch: int = 4,
    ef_search: int | None = None,
) -> SearchResult:
    """prefer: path patterns of the nodes users should query (marts); empty turns lifting off.
    scan, overfetch, ef_search: how the nearest chunks are found (search/scan.py)."""
    params = {"project": project, "kinds": list(kinds), "k": k, "question": question, "ids": None}
    note = None
    if embedder is not None and not keyword_only:
        tag = index_name(embedder)
        if has_embeddings(conn, tag, project):
            vector = embedder.embed([question], kind="query").vectors[0]
            params |= {"vector": vector_literal(vector), "model": tag}
            first = params if not prefer else params | {"k": CANDIDATES}
            candidates = nearest(conn, first, scan, overfetch, ef_search)
            if not prefer:
                return SearchResult("semantic", candidates, tag)
            return SearchResult("semantic", _lift(conn, candidates, params, prefer, k), tag)
        note = f"no embeddings tagged {tag!r} yet; run `asklineage embed`"
    elif embedder is None and not keyword_only:
        note = "EMBED_PROVIDER=none"
    return SearchResult("keyword", _rows(conn, KEYWORD, params), note=note)


def nearest(
    conn: psycopg.Connection,
    params: dict,
    scan: Scan,
    overfetch: int,
    ef_search: int | None = None,
) -> tuple[Hit, ...]:
    """The k nodes whose best chunk is nearest the question vector."""
    if scan == "exact":
        return _rows(conn, SEMANTIC, params)
    fetch = fetch_size(params["k"], overfetch)
    with vector_scan(conn, scan, fetch, ef_search):
        return _rows(conn, SEMANTIC_NEAREST, params | {"fetch": fetch})


def _lift(
    conn: psycopg.Connection,
    candidates: tuple[Hit, ...],
    params: dict,
    prefer: Sequence[str],
    k: int,
) -> tuple[Hit, ...]:
    def preferred(hit: Hit) -> bool:
        return bool(hit.file_path) and any(fnmatch(hit.file_path, p) for p in prefer)

    known = {hit.unique_id: hit for hit in candidates}
    ups = [hit.unique_id for hit in candidates if not preferred(hit)]
    downs: dict[str, set[str]] = defaultdict(set)
    if ups:
        with conn.cursor(row_factory=dict_row) as cur:
            for row in cur.execute(DOWNSTREAM_2, {"ids": ups}).fetchall():
                downs[row["up"]].add(row["down"])
    missing = list({d for ds in downs.values() for d in ds} - known.keys())
    if missing:
        extra = _rows(conn, SEMANTIC, params | {"ids": missing, "k": len(missing)})
        known |= {hit.unique_id: hit for hit in extra if preferred(hit)}

    scored = {
        uid: hit if preferred(hit) else replace(hit, score=hit.score * LAYER_PENALTY)
        for uid, hit in known.items()
        if uid in {c.unique_id for c in candidates} or preferred(hit)
    }
    for up, children in downs.items():
        source = known[up]
        targets = sorted(
            (known[d] for d in children if d in known and preferred(known[d])),
            key=lambda hit: (-hit.score, hit.unique_id),
        )[:MAX_LIFTS]
        for target in targets:
            if source.score > scored[target.unique_id].score:
                scored[target.unique_id] = replace(
                    target, score=source.score, via=source.display_name
                )
    ranked = sorted(scored.values(), key=lambda hit: (-hit.score, hit.unique_id))
    return tuple(ranked[:k])


def _rows(conn: psycopg.Connection, sql: str, params: dict) -> tuple[Hit, ...]:
    with conn.cursor(row_factory=dict_row) as cur:
        return tuple(Hit(**row) for row in cur.execute(sql, params).fetchall())
