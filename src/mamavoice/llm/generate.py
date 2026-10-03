from __future__ import annotations

import httpx

from mamavoice.config import settings
from mamavoice.rag.retriever import Hit

SYSTEM = """You are a maternal health information assistant for women in Nigeria.
Rules:
- Answer ONLY from the CONTEXT. If the context does not answer the question, say you are not sure
  and advise the person to ask a nurse, midwife or doctor.
- Use short, simple sentences (grade 6 reading level). No medical jargon.
- Never diagnose. Never give drug names or doses. Never contradict the context.
- If anything sounds urgent, tell the person to go to a clinic now.
- End with: "This is general information, not a replacement for a health worker."
"""

REFUSAL_EN = (
    "I am not sure about this. Please ask a nurse, midwife or doctor at your nearest clinic. "
    "This is general information, not a replacement for a health worker."
)


def build_prompt(question_en: str, hits: list[Hit]) -> str:
    ctx = "\n\n".join(f"[{i+1}] ({h.source}) {h.text}" for i, h in enumerate(hits))
    return f"CONTEXT:\n{ctx}\n\nQUESTION: {question_en}\n\nANSWER:"


def generate(question_en: str, hits: list[Hit]) -> str:
    r = httpx.post(
        f"{settings.ollama_url}/api/chat",
        json={
            "model": settings.llm_model,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(question_en, hits)},
            ],
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
