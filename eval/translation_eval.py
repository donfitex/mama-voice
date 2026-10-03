"""Compare base vs fine-tuned NLLB on FLORES-200 devtest (or any TSV of src\ttgt).
    python eval/translation_eval.py --lang yo --direction xx-en [--adapter outputs/nllb-yo-lora]
"""
import argparse
import json

import sacrebleu
from datasets import load_dataset

from mamavoice.config import settings
from mamavoice.translate import nllb

CODE = {"yo": "yor_Latn", "ha": "hau_Latn", "ig": "ibo_Latn"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=CODE)
    ap.add_argument("--direction", default="xx-en", choices=["xx-en", "en-xx"])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    settings.nllb_adapter = a.adapter

    ds = load_dataset("facebook/flores", "all", split="devtest", trust_remote_code=True)
    src_col, tgt_col = (f"sentence_{CODE[a.lang]}", "sentence_eng_Latn")
    if a.direction == "en-xx":
        src_col, tgt_col = tgt_col, src_col
    s, t = (a.lang, "en") if a.direction == "xx-en" else ("en", a.lang)
    rows = ds.select(range(min(a.n, len(ds))))
    hyp = [nllb.translate(r[src_col], s, t) for r in rows]
    ref = [r[tgt_col] for r in rows]
    out = {"lang": a.lang, "direction": a.direction, "adapter": a.adapter, "n": len(hyp),
           "chrF++": round(sacrebleu.corpus_chrf(hyp, [ref], word_order=2).score, 2),
           "spBLEU": round(sacrebleu.corpus_bleu(hyp, [ref], tokenize="flores200").score, 2)}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
