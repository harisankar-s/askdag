"""Keep node_chunks in step with the loaded nodes, embedding only what changed."""

import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass

import psycopg
from psycopg.rows import dict_row, tuple_row

from asklineage.embed.base import EmbeddingError, EmbeddingProvider, index_name
from asklineage.embed.chunks import Chunk, chunk_node
from asklineage.ingest.manifest import Column, Node

BATCH = 64

NODE_FIELDS = (
    "unique_id, resource_type, name, source_name, package_name, database_name, schema_name, "
    "relation_name, materialization, description, file_path, tags, owner, owner_source, raw_code"
)

UPSERT = """
    INSERT INTO node_chunks
        (node_id, project_name, kind, ordinal, content, content_sha, model, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
    ON CONFLICT (node_id, kind, ordinal) DO UPDATE SET
        project_name = EXCLUDED.project_name, content = EXCLUDED.content,
        content_sha = EXCLUDED.content_sha, model = EXCLUDED.model,
        embedding = EXCLUDED.embedding, embedded_at = now()
"""


@dataclass(frozen=True)
class SyncReport:
    project: str
    chunks: Counter[str]
    embedded: int
    kept: int
    deleted: int
    seconds: float


def project_chunks(conn: psycopg.Connection, project: str) -> list[Chunk]:
    with conn.cursor(row_factory=dict_row) as cur:
        nodes = cur.execute(
            f"SELECT {NODE_FIELDS} FROM nodes WHERE project_name = %s ORDER BY unique_id",
            (project,),
        ).fetchall()
        rows = cur.execute(
            "SELECT c.node_id, c.name, c.description, c.data_type FROM node_columns c "
            "JOIN nodes n ON n.unique_id = c.node_id WHERE n.project_name = %s "
            "ORDER BY c.node_id, c.name",
            (project,),
        ).fetchall()
    columns: dict[str, list[Column]] = defaultdict(list)
    for row in rows:
        columns[row["node_id"]].append(Column(**row))
    chunks: list[Chunk] = []
    for row in nodes:
        node = Node(**{**row, "tags": tuple(row["tags"])})
        chunks += chunk_node(node, columns[node.unique_id])
    return chunks


def column_dimensions(conn: psycopg.Connection) -> int:
    with conn.cursor(row_factory=tuple_row) as cur:
        (dims,) = cur.execute(
            "SELECT atttypmod FROM pg_attribute "
            "WHERE attrelid = 'node_chunks'::regclass AND attname = 'embedding'"
        ).fetchone()
    return dims


def sync_chunks(
    conn: psycopg.Connection,
    project: str,
    embedder: EmbeddingProvider,
    on_batch: Callable[[int, int], None] | None = None,
) -> SyncReport:
    """Embed new or changed chunks, keep the rest, delete chunks of nodes that are gone.

    A chunk is re-embedded when its text, the model or the document prefix changed. On an
    autocommit connection each batch commits on its own, so an interrupted run keeps what it
    finished.
    """
    dims = column_dimensions(conn)
    if dims != embedder.dimensions:
        raise EmbeddingError(
            f"node_chunks.embedding has {dims} dimensions, but {embedder.model!r} gives "
            f"{embedder.dimensions}. Change the column type by hand, then embed again."
        )
    started = time.perf_counter()
    tag = index_name(embedder)
    wanted = project_chunks(conn, project)
    stored = _stored(conn, project)
    keys = {(c.node_id, c.kind, c.ordinal) for c in wanted}
    stale = [key for key in stored if key not in keys]
    todo = [c for c in wanted if stored.get((c.node_id, c.kind, c.ordinal)) != (c.sha, tag)]
    with conn.transaction():
        conn.cursor().executemany(
            "DELETE FROM node_chunks WHERE node_id = %s AND kind = %s AND ordinal = %s", stale
        )
    for start in range(0, len(todo), BATCH):
        batch = todo[start : start + BATCH]
        vectors = embedder.embed([c.content for c in batch], kind="document").vectors
        with conn.transaction():
            conn.cursor().executemany(
                UPSERT,
                [
                    (
                        c.node_id,
                        project,
                        c.kind,
                        c.ordinal,
                        c.content,
                        c.sha,
                        tag,
                        vector_literal(vector),
                    )
                    for c, vector in zip(batch, vectors, strict=True)
                ],
            )
        if on_batch:
            on_batch(start + len(batch), len(todo))
    return SyncReport(
        project,
        Counter(c.kind for c in wanted),
        embedded=len(todo),
        kept=len(wanted) - len(todo),
        deleted=len(stale),
        seconds=time.perf_counter() - started,
    )


def _stored(conn: psycopg.Connection, project: str) -> dict[tuple[str, str, int], tuple[str, str]]:
    with conn.cursor(row_factory=tuple_row) as cur:
        rows = cur.execute(
            "SELECT node_id, kind, ordinal, content_sha, model FROM node_chunks "
            "WHERE project_name = %s",
            (project,),
        ).fetchall()
    return {(r[0], r[1], r[2]): (r[3], r[4]) for r in rows}


def vector_literal(vector: list[float]) -> str:
    # pgvector's text form; avoids the pgvector package, which needs numpy.
    return "[" + ",".join(repr(x) for x in vector) + "]"
