from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from mamavoice.config import settings
from mamavoice.safety.red_flags import ESCALATION_EN, check

log = logging.getLogger("mamavoice")


@dataclass
class Result:
    lang: str
    transcript: str
    question_en: str
    answer_en: str
    answer: str
    escalated: bool
    safety_categories: tuple[str, ...] = ()
    sources: list[str] = field(default_factory=list)
    latency_ms: dict[str, int] = field(default_factory=dict)


class Pipeline:
    """Lazy-loads heavy components so unit tests and the safety layer work without GPUs/models."""

    def __init__(self) -> None:
        self._retriever = None

    @property
    def retriever(self):
        if self._retriever is None:
            from mamavoice.rag.retriever import Retriever

            self._retriever = Retriever()
        return self._retriever

    def ask_text(self, text: str, lang: str) -> Result:
        from mamavoice.llm.generate import REFUSAL_EN, generate
        from mamavoice.translate.nllb import translate, translate_long

        t: dict[str, int] = {}

        def timed(name, fn, *a):
            s = time.perf_counter()
            r = fn(*a)
            t[name] = int((time.perf_counter() - s) * 1000)
            return r

        q_en = timed("translate_in", translate, text, lang, "en")
        safety = check(q_en, text, lang)
        if safety:
            ans_en = ESCALATION_EN
            hits = []
        else:
            hits = timed("retrieve", self.retriever.search, q_en)
            if not hits or hits[0].score < settings.min_retrieval_score:
                ans_en = REFUSAL_EN
            else:
                ans_en = timed("generate", generate, q_en, hits)
        ans = timed("translate_out", translate_long, ans_en, "en", lang)
        res = Result(lang, text, q_en, ans_en, ans, bool(safety), safety.categories,
                     sorted({h.source for h in hits}), t)
        # Log metadata only. Never log raw user text in production (privacy).
        log.info("ask lang=%s escalated=%s cats=%s latency=%s", lang, res.escalated,
                 res.safety_categories, t)
        return res

    def ask_audio(self, audio_path: str, lang: str, tts_out: str | None = None) -> Result:
        from mamavoice.speech.stt import transcribe
        from mamavoice.speech.tts import synthesize

        s = time.perf_counter()
        text = transcribe(audio_path, lang)
        res = self.ask_text(text, lang)
        res.latency_ms["stt"] = int((time.perf_counter() - s) * 1000)
        if tts_out:
            synthesize(res.answer, lang, tts_out)
        return res
