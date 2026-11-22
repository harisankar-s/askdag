"""Turn an incident into indexed chunks and detect which nodes it names. Pure: no database.

One chunk per section, so "how was it fixed?" can match the Fix section on its own. Each
chunk starts with a header (id, date, severity, title) so a bare "Fix" paragraph still says
which incident it belongs to.
"""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from asklineage.corpus.docs import Document, exact_names
from asklineage.embed.chunks import CHUNK_CHARS, pack

SECTION = re.compile(r"^## (.+?)\s*$", re.MULTILINE)
# Model names, and source names written as source_name.table.
TOKEN = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)?")


@dataclass(frozen=True)
class DocChunk:
    doc_id: str
    ordinal: int
    section: str
    content: str


def sections(body: str) -> list[tuple[str, str]]:
    """(heading, text) for each `## ` section; text before the first heading is dropped."""
    parts = SECTION.split(body)
    return [
        (heading, " ".join(text.split()))
        for heading, text in zip(parts[1::2], parts[2::2], strict=True)
        if text.strip()
    ]


def header(doc: Document) -> str:
    severity = doc.meta.get("severity")
    tag = f"{doc.type} {doc.id} · {doc.date.isoformat()}" + (f" · {severity}" if severity else "")
    return f"{tag} · {doc.title}"


def chunk_incident(doc: Document, limit: int = CHUNK_CHARS) -> list[DocChunk]:
    head = header(doc)
    chunks: list[DocChunk] = []
    for heading, text in sections(doc.body):
        lead = f"{head}\n{heading}: "
        for piece in pack(text.split(), limit - len(lead), " "):
            chunks.append(DocChunk(doc.id, len(chunks), heading, f"{lead}{piece}"))
    return chunks


def spellings(names: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Lower-cased spelling -> unique_id, from (display name, unique_id) pairs.

    A spelling shared by two nodes (two sources with the same table name) is dropped: naming
    it doesn't say which node is meant.
    """
    owners: dict[str, set[str]] = defaultdict(set)
    for name, unique_id in names:
        for spelling in exact_names(name):
            owners[spelling.lower()].add(unique_id)
    return {spelling: next(iter(ids)) for spelling, ids in owners.items() if len(ids) == 1}


def detect_nodes(text: str, known: dict[str, str]) -> tuple[str, ...]:
    """unique_ids of the nodes the text names exactly, in order of first mention."""
    found: dict[str, None] = {}
    for token in TOKEN.findall(text.lower()):
        for candidate in (token, *token.split(".")):
            if candidate in known:
                found.setdefault(known[candidate], None)
                break
    return tuple(found)
