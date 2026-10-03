"""Deterministic red-flag detector for maternal danger signs.

Design rule: this layer NEVER depends on an LLM. If it fires, the pipeline returns a fixed
escalation message and skips generation. Patterns are matched on the English translation AND
on the original text (seed terms for local languages must be reviewed by native speakers).
Sources for the English list: WHO danger signs in pregnancy/postpartum (verify against the
current WHO/FMoH documents before any real-world use).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# (category, regex) - deliberately high-recall; false positives only cost a clinic visit.
EN_PATTERNS: list[tuple[str, str]] = [
    ("bleeding", r"\b(heavy|severe|a lot of|too much|soaking)\b.{0,30}\b(bleed|blood)"),
    ("bleeding", r"\bbleeding\b.{0,30}\b(pregnan|after (birth|delivery)|won'?t stop)"),
    ("bleeding", r"\bvaginal bleeding\b"),
    ("bleeding", r"\bbleeding\b.{0,15}\b(heavily|badly|a lot|profusely)"),
    ("headache_vision", r"\b(severe|bad|terrible|worst)\b.{0,20}\bheadache"),
    ("headache_vision", r"\b(blurr?y|blurred|double|loss of) vision\b|\bseeing spots\b"),
    ("seizure", r"\b(seizures?|convulsions?|convulsing|fits?|fitting|fainted|unconscious|passed out)\b"),
    ("fetal_movement", r"\b(baby|fetus)\b.{0,30}\b(not|stopped|less|reduced|no)\b.{0,15}\bmov"),
    ("fetal_movement", r"\b(reduced|decreased|no|less|stopped)\b.{0,15}\b(fetal )?movement"),
    ("fluid", r"\b(water|waters|fluid)\b.{0,15}\b(broke|breaking|leaking|gushing)"),
    ("fever_infection", r"\bhigh fever\b|\bfever\b.{0,25}\b(after (birth|delivery)|pregnan)"),
    ("fever_infection", r"\bfoul[- ]smelling\b"),
    ("breathing_pain", r"\b(difficulty|trouble|hard to) breath|\bshort(ness)? of breath\b|\bchest pain\b"),
    ("abdominal_pain", r"\b(severe|sharp|constant)\b.{0,15}\b(abdominal|stomach|belly) pain"),
    ("swelling", r"\bsudden(ly)?\b.{0,20}\bswell(ing|en)\b.{0,20}\b(face|hands|feet)"),
    ("swelling", r"\b(face|hands|feet)\b.{0,30}\bsudden(ly)?\b.{0,15}\bswoll?en\b"),
    ("swelling", r"\b(face|hands|feet)\b.{0,30}\bsuddenly\b.{0,15}\bswell"),
    ("labour", r"\b(labou?r|contractions)\b.{0,30}\b(before (37|38)|too early|preterm|premature)"),
    ("self_harm", r"\b(kill myself|want to die|end my life|suicid)"),
]

# Seed terms in local languages. MUST be extended and validated with native speakers.
LOCAL_SEED: dict[str, list[tuple[str, str]]] = {
    "yo": [("bleeding", r"\bẹ̀jẹ̀\b|\bèjè\b")],
    "ha": [("bleeding", r"\bzubar da jini\b|\bzubar jini\b")],
    "ig": [("bleeding", r"\bọbara\b.{0,20}\bmgbali\b")],
}

_COMPILED_EN = [(c, re.compile(p, re.I | re.S)) for c, p in EN_PATTERNS]
_COMPILED_LOCAL = {
    lang: [(c, re.compile(p, re.I)) for c, p in pats] for lang, pats in LOCAL_SEED.items()
}

ESCALATION_EN = (
    "This could be a danger sign. Please go to the nearest clinic or hospital now, "
    "or ask someone to take you immediately. If you can, call your local emergency number. "
    "Do not wait for this app to answer further."
)


@dataclass(frozen=True)
class SafetyResult:
    escalate: bool
    categories: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.escalate


def check(text_en: str = "", text_original: str = "", lang: str = "en") -> SafetyResult:
    cats: set[str] = set()
    for cat, rx in _COMPILED_EN:
        if rx.search(text_en or "") or (lang == "en" and rx.search(text_original or "")):
            cats.add(cat)
    for cat, rx in _COMPILED_LOCAL.get(lang, []):
        if rx.search(text_original or ""):
            cats.add(cat)
    return SafetyResult(bool(cats), tuple(sorted(cats)))
