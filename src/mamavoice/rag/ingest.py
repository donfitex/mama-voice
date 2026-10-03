"""Build a FAISS index from PDFs/TXT/MD files in data/guidelines/.

Download public guidelines yourself (WHO, Nigeria FMoH) and record the source URL + licence in
data/guidelines/SOURCES.md. Usage: python -m mamavoice.rag.ingest
"""
from __future__ import annotations

import json
from pathlib import Path

from mamavoice.config import settings


def read_file(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += size - overlap
    return [c for c in out if len(c) > 80]


def main() -> None:
    import faiss
    from sentence_transformers import SentenceTransformer

    docs = []
    for p in sorted(Path(settings.guidelines_dir).glob("*")):
        if p.suffix.lower() in {".pdf", ".txt", ".md"} and p.name != "SOURCES.md":
            for j, c in enumerate(chunk(read_file(p))):
                docs.append({"source": p.name, "chunk": j, "text": c})
    if not docs:
        raise SystemExit(f"No documents found in {settings.guidelines_dir}")

    model = SentenceTransformer(settings.embed_model)
    emb = model.encode([f"passage: {d['text']}" for d in docs], normalize_embeddings=True,
                       show_progress_bar=True)
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb.astype("float32"))

    out = Path(settings.index_dir)
    out.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out / "faiss.index"))
    (out / "chunks.json").write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
    print(f"Indexed {len(docs)} chunks from {len({d['source'] for d in docs})} files")


if __name__ == "__main__":
    main()
