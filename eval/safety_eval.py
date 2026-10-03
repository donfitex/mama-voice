"""Recall/precision of the red-flag layer on eval/data/safety_cases.jsonl. Used as a CI gate."""
import json
import sys
from pathlib import Path

from mamavoice.safety.red_flags import check


def run(path="eval/data/safety_cases.jsonl"):
    tp = fp = fn = tn = 0
    misses = []
    for line in Path(path).read_text().splitlines():
        c = json.loads(line)
        got = check(c["text"], c["text"], c.get("lang", "en")).escalate
        if c["escalate"] and got: tp += 1
        elif c["escalate"] and not got: fn += 1; misses.append(c["text"])
        elif not c["escalate"] and got: fp += 1
        else: tn += 1
    recall = tp / max(tp + fn, 1)
    precision = tp / max(tp + fp, 1)
    return {"recall": round(recall, 3), "precision": round(precision, 3), "tp": tp, "fn": fn,
            "fp": fp, "tn": tn, "missed": misses}


if __name__ == "__main__":
    r = run()
    print(json.dumps(r, indent=2, ensure_ascii=False))
    sys.exit(0 if r["recall"] >= 0.95 else 1)   # danger-sign recall must stay >= 95%
