import json
from mamavoice.safety.red_flags import check
from mamavoice.translate.nllb import translate

cases = [json.loads(l) for l in open("eval/data/safety_cases.jsonl", encoding="utf-8")]
danger = [c["text"] for c in cases if c["escalate"]]
danger = danger[:8]   # first run: only 8 sentences. Remove this line to test all 20.

for lang in ["yo", "ha", "ig"]:
    caught, missed = 0, []
    for en in danger:
        local = translate(en, "en", lang)
        back = translate(local, lang, "en")
        if check(back, local, lang).escalate:
            caught += 1
        else:
            missed.append((en, local, back))
    print(f"\n{lang}: caught {caught}/{len(danger)}")
    for m in missed:
        print("   MISSED:", m)