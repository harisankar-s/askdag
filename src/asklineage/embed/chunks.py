"""Turn a node into the texts that get embedded. Pure: no database, no model.

Each node gives up to three kinds of chunk, so the search eval can compare what is worth
embedding by filtering on kind: summary (name and description, always present), columns and
sql. Every chunk starts with the node's name, so a piece of SQL still says whose SQL it is.
"""

import hashlib
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from asklineage.ingest.manifest import Column, Node

type ChunkKind = Literal["summary", "columns", "sql"]
CHUNK_KINDS: tuple[ChunkKind, ...] = ("summary", "columns", "sql")
# About 500 tokens: well inside the context Ollama may give an embedding model.
CHUNK_CHARS = 2000
# Embedding models read occupancy_status as one odd token; "occupancy status" as two words.
SNAKE = re.compile(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])")


@dataclass(frozen=True)
class Chunk:
    node_id: str
    kind: ChunkKind
    ordinal: int
    content: str

    @property
    def sha(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()


def header(node: Node) -> str:
    name = f"{node.source_name}.{node.name}" if node.source_name else node.name
    folder = str(PurePosixPath(node.file_path).parent) if node.file_path else None
    return (
        f"{name} ({node.resource_type}, {folder})" if folder else f"{name} ({node.resource_type})"
    )


def chunk_node(node: Node, columns: Sequence[Column], limit: int = CHUNK_CHARS) -> list[Chunk]:
    head = header(node)
    room = limit - len(head) - 1
    description = " ".join(node.description.split())
    pieces: dict[ChunkKind, list[str]] = {
        # An empty description still gives a summary chunk: the name alone can match.
        "summary": _pack(description.split(), room, " ") or [""],
        "columns": _pack(
            (" ".join(f"{_words(c.name)}: {c.description}".split()).rstrip(":") for c in columns),
            room,
            "\n",
        ),
        "sql": _pack(_words(node.raw_code or "").split(), room, " "),
    }
    return [
        Chunk(node.unique_id, kind, ordinal, f"{head}\n{piece}".rstrip())
        for kind in CHUNK_KINDS
        for ordinal, piece in enumerate(pieces[kind])
    ]


def _words(text: str) -> str:
    return SNAKE.sub(" ", text)


def _pack(items: Iterable[str], room: int, sep: str) -> list[str]:
    """Join items into pieces of at most room characters; an oversized item is cut."""
    pieces: list[str] = []
    current = ""
    for item in items:
        while len(item) > room:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(item[:room])
            item = item[room:]
        if current and len(current) + len(sep) + len(item) > room:
            pieces.append(current)
            current = item
        else:
            current = f"{current}{sep}{item}" if current else item
    if current:
        pieces.append(current)
    return pieces
