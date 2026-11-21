"""A deterministic embedder for tests: texts that share words get similar vectors.

It runs offline, so search queries can be tested against real pgvector in CI. It knows nothing
about meaning (synonyms score zero), so it must never be used to judge search quality.
"""

import hashlib
import math
import re
from collections.abc import Sequence

from asklineage.embed.base import Embeddings, Kind

# Splits on underscores too, so dim_schedule_feeds shares tokens with "schedule feeds".
TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dimensions: int = 768) -> None:
        self.model = f"hashing-{dimensions}"
        self.dimensions = dimensions
        self.document_prefix = ""

    def embed(self, texts: Sequence[str], *, kind: Kind) -> Embeddings:
        return Embeddings([self._vector(text) for text in texts], self.model, None, 0)

    def _vector(self, text: str) -> list[float]:
        tokens = TOKEN.findall(text.lower())
        if not tokens:
            raise ValueError(f"no words to embed in {text!r}")
        vector = [0.0] * self.dimensions
        for token in tokens:
            digest = int.from_bytes(hashlib.blake2b(token.encode(), digest_size=8).digest())
            # A signed bucket keeps unrelated tokens that collide from always adding up.
            sign = 1.0 if digest >> 63 else -1.0
            vector[digest % self.dimensions] += sign
        norm = math.sqrt(sum(x * x for x in vector))
        return [x / norm for x in vector]
