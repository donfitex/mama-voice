import time
from mamavoice.translate.nllb import translate

text = "Pregnant women should visit the clinic regularly."
for lang in ["yo", "ha", "ig"]:
    t = time.time()
    out = translate(text, "en", lang)
    back = translate(out, lang, "en")
    print(f"[{lang}] {out}\n     back to English: {back}   ({time.time() - t:.1f}s)\n")