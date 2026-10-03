import time
from mamavoice.speech.tts import synthesize
from mamavoice.speech.stt import transcribe
from mamavoice.translate.nllb import translate

en = "I have a headache and I feel tired."
for lang in ["en", "yo", "ha"]:
    text = translate(en, "en", lang)
    wav = f"loop_{lang}.wav"
    t = time.time(); synthesize(text, lang, wav); tts_s = time.time() - t
    t = time.time(); heard = transcribe(wav, lang); stt_s = time.time() - t
    print(f"[{lang}] said : {text}\n     heard: {heard}\n     tts {tts_s:.1f}s, stt {stt_s:.1f}s\n")