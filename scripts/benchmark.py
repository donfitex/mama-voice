"""Latency benchmark: python scripts/benchmark.py --lang yo --n 20 (needs the API running)."""
import argparse, statistics, httpx

ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="en"); ap.add_argument("--n", type=int, default=20)
a = ap.parse_args()
qs = ["What foods are good during pregnancy?", "How many clinic visits do I need?"]
lat = {}
for i in range(a.n):
    r = httpx.post("http://localhost:8000/ask", json={"text": qs[i % 2], "lang": a.lang}, timeout=300).json()
    for k, v in r["latency_ms"].items(): lat.setdefault(k, []).append(v)
for k, v in lat.items():
    v.sort(); print(f"{k:14s} p50={statistics.median(v):7.0f}ms  p95={v[int(len(v)*.95)-1]:7.0f}ms")
