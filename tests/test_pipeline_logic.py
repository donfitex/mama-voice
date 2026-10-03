from mamavoice.llm.generate import build_prompt
from mamavoice.rag.ingest import chunk
from mamavoice.rag.retriever import Hit


def test_chunk_overlap_and_min_len():
    text = "word " * 1000
    cs = chunk(text, size=300, overlap=50)
    assert len(cs) > 5 and all(len(c) > 80 for c in cs)


def test_prompt_contains_sources():
    p = build_prompt("q?", [Hit("some text", "who.pdf", 0.9)])
    assert "who.pdf" in p and "QUESTION: q?" in p
