from __future__ import annotations

import re
from functools import lru_cache

from mamavoice.config import LANGS, settings


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(settings.nllb_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(settings.nllb_model)
    if settings.nllb_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, settings.nllb_adapter).merge_and_unload()
    return tok, model.eval()


def translate(text: str, src: str, tgt: str, max_new_tokens: int = 256) -> str:
    if src == tgt or not text.strip():
        return text
    import torch

    tok, model = _load()
    tok.src_lang = LANGS[src]["nllb"]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(LANGS[tgt]["nllb"]),
            max_new_tokens=max_new_tokens,
            num_beams=4,
        )
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def translate_long(text: str, src: str, tgt: str) -> str:
    """Sentence-wise translation: NLLB degrades on long inputs."""
    parts = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    return " ".join(translate(p, src, tgt) for p in parts)
