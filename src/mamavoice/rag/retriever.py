from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from mamavoice.config import settings


@dataclass
class Hit:
    text: str
    source: str
    score: float


class Retriever:
    def __init__(self) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        d = Path(settings.index_dir)
        self.index = faiss.read_index(str(d / "faiss.index"))
        self.chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        self.model = SentenceTransformer(settings.embed_model)

    def search(self, query_en: str, k: int | None = None) -> list[Hit]:
        q = self.model.encode([f"query: {query_en}"], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(q, k or settings.top_k)
        return [
            Hit(self.chunks[i]["text"], self.chunks[i]["source"], float(s))
            for s, i in zip(scores[0], ids[0]) if i >= 0
        ]
