# Architecture
```
audio -> STT (Whisper | MMS for ig) -> NLLB to EN -> [red-flag check] -+-> fixed escalation
                                                                       |
                                          RAG (FAISS + e5) -> LLM (Ollama, grounded) -> NLLB to user lang -> MMS-TTS -> audio
```
Design choices: English pivot lets one guideline index serve all languages; safety is rule-based and
model-free; every heavy component lazy-loads; /safety/check works with no models for very weak
servers; Telegram/SMS front-ends keep payloads tiny (text first, 16 kbps Opus audio second).
Cloud: containerised API deployable to GCP Cloud Run / AWS App Runner / Azure Container Apps free tiers;
or Hugging Face Spaces (Docker) for the public demo.
