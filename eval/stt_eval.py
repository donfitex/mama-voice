"""WER/CER on Google FLEURS. python eval/stt_eval.py --lang yo --n 100"""
import argparse
import json

import jiwer
from datasets import load_dataset

from mamavoice.speech.stt import transcribe

FLEURS = {"en": "en_us", "yo": "yo_ng", "ha": "ha_ng", "ig": "ig_ng"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=FLEURS)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    ds = load_dataset("google/fleurs", FLEURS[a.lang], split="test", trust_remote_code=True)
    hyp, ref = [], []
    for r in ds.select(range(min(a.n, len(ds)))):
        hyp.append(transcribe(r["audio"]["path"], a.lang).lower())
        ref.append(r["transcription"].lower())
    print(json.dumps({"lang": a.lang, "n": len(ref), "WER": round(jiwer.wer(ref, hyp), 3),
                      "CER": round(jiwer.cer(ref, hyp), 3)}, indent=2))


if __name__ == "__main__":
    main()
