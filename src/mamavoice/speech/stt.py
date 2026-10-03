"""STT: faster-whisper for en/yo/ha; Meta MMS (adapter) for Igbo, which Whisper does not cover."""
from __future__ import annotations

from functools import lru_cache

from mamavoice.config import LANGS, settings


@lru_cache(maxsize=1)
def _whisper():
    from faster_whisper import WhisperModel

    return WhisperModel(settings.whisper_size, device="cpu", compute_type="int8")


@lru_cache(maxsize=1)
def _mms():
    from transformers import AutoProcessor, Wav2Vec2ForCTC

    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all")
    return proc, model


def transcribe(audio_path: str, lang: str) -> str:
    code = LANGS[lang]["whisper"]
    if code:
        segs, _ = _whisper().transcribe(audio_path, language=code, vad_filter=True, beam_size=5)
        return " ".join(s.text.strip() for s in segs).strip()
    import librosa  # pip install librosa for the MMS path
    import torch

    proc, model = _mms()
    proc.tokenizer.set_target_lang(LANGS[lang]["mms_asr"])
    model.load_adapter(LANGS[lang]["mms_asr"])
    wav, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = proc(wav, sampling_rate=16000, return_tensors="pt")
    with torch.inference_mode():
        ids = torch.argmax(model(**inputs).logits, dim=-1)[0]
    return proc.decode(ids)
