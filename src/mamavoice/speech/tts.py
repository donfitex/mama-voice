from __future__ import annotations

from functools import lru_cache

from mamavoice.config import LANGS


@lru_cache(maxsize=4)
def _load(lang: str):
    from transformers import AutoTokenizer, VitsModel

    name = LANGS[lang]["mms_tts"]
    return AutoTokenizer.from_pretrained(name), VitsModel.from_pretrained(name).eval()


def synthesize(text: str, lang: str, out_path: str) -> str:
    """Writes a mono WAV. For low bandwidth, convert to Opus afterwards (scripts/to_opus.sh)."""
    import soundfile as sf
    import torch

    tok, model = _load(lang)
    with torch.inference_mode():
        wav = model(**tok(text, return_tensors="pt")).waveform[0].numpy()
    sf.write(out_path, wav, model.config.sampling_rate)
    return out_path
