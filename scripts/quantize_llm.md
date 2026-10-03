# Running small and offline
- Ollama pulls 4-bit GGUF builds: `ollama pull qwen2.5:3b-instruct` (~2 GB, CPU-friendly).
- Export NLLB to CTranslate2 int8 for ~3x faster CPU inference:
  `ct2-transformers-converter --model facebook/nllb-200-distilled-600M --output_dir nllb-ct2 --quantization int8`
- Record model size, RAM, and p50/p95 latency per stage in docs/BENCHMARKS.md.
