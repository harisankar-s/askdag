"""Index a corpus's incidents: rows in documents, sections in doc_chunks, vectors if configured.

A full replace on each load, after embedding has succeeded, so a failed embed leaves the old
index in place. Forty incidents are about 200 chunks, a few seconds to embed; a ticketing
system would instead sync tickets changed since the last run, keyed on the issue key (ADR 0015).
"""

import time
from dataclasses import dataclass

import psycopg
from psycopg.rows import tuple_row

from asklineage.corpus.docs import Corpus
from asklineage.embed.base import EmbeddingError, EmbeddingProvider, index_name
from asklineage.embed.store import BATCH, column_dimensions, vector_literal
from asklineage.incidents.chunks import chunk_incident, detect_nodes, spellings

SOURCE = "markdown"

DISPLAY_NAMES = """
    SELECT CASE WHEN source_name IS NULL THEN name ELSE source_name || '.' || name END,
           unique_id
    FROM nodes WHERE project_name = %s
"""

ALL_DISPLAY_NAMES = DISPLAY_NAMES.replace("WHERE project_name = %s", "")

INSERT_DOCUMENT = """
    INSERT INTO documents
        (project_name, doc_id, source, url, doc_type, title, doc_date, severity, node_ids, body)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""

INSERT_CHUNK = """
    INSERT INTO doc_chunks (project_name, doc_id, ordinal, section, content, model, embedding)
    VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
"""


@dataclass(frozen=True)
class IndexReport:
    project: str
    documents: int
    chunks: int
    linked: int  # documents that name at least one node
    links: int
    model: str | None
    seconds: float


def index_incidents(
    conn: psycopg.Connection,
    corpus: Corpus,
    embedder: EmbeddingProvider | None,
) -> IndexReport:
    """Replace the project's indexed incidents. A document's url is its path as loaded."""
    started = time.perf_counter()
    project = corpus.project
    with conn.cursor(row_factory=tuple_row) as cur:
        known = spellings(cur.execute(DISPLAY_NAMES, (project,)).fetchall())
    if not known:
        raise LookupError(f"no nodes loaded for {project!r}; run load-manifest first")

    incidents = [doc for doc in corpus.documents if doc.type == "incident"]
    nodes = {doc.id: detect_nodes(f"{doc.title}\n{doc.body}", known) for doc in incidents}
    chunks = [chunk for doc in incidents for chunk in chunk_incident(doc)]

    tag = None
    vectors: list[str | None] = [None] * len(chunks)
    if embedder is not None:
        dims = column_dimensions(conn, "doc_chunks")
        if dims != embedder.dimensions:
            raise EmbeddingError(
                f"doc_chunks.embedding has {dims} dimensions, but {embedder.model!r} gives "
                f"{embedder.dimensions}. Change the column type by hand, then load again."
            )
        tag = index_name(embedder)
        vectors = []
        for start in range(0, len(chunks), BATCH):
            batch = [c.content for c in chunks[start : start + BATCH]]
            vectors += [vector_literal(v) for v in embedder.embed(batch, kind="document").vectors]

    with conn.transaction(), conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE project_name = %s", (project,))
        cur.executemany(
            INSERT_DOCUMENT,
            [
                (
                    project,
                    doc.id,
                    SOURCE,
                    doc.path.as_posix(),
                    doc.type,
                    doc.title,
                    doc.date,
                    doc.meta.get("severity"),
                    list(nodes[doc.id]),
                    doc.body,
                )
                for doc in incidents
            ],
        )
        cur.executemany(
            INSERT_CHUNK,
            [
                (project, c.doc_id, c.ordinal, c.section, c.content, tag if v else None, v)
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
    return IndexReport(
        project,
        documents=len(incidents),
        chunks=len(chunks),
        linked=sum(1 for ids in nodes.values() if ids),
        links=sum(len(ids) for ids in nodes.values()),
        model=tag,
        seconds=time.perf_counter() - started,
    )
