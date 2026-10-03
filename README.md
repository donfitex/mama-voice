# MamaVoice: voice-based maternal health information in Nigerian languages

Research prototype: speech-in, speech-out maternal health Q&A for Yoruba, Hausa, Igbo and English, built
on free/open tooling only. Covers LLM + RAG, NLP, STT/TTS, translation fine-tuning, MLOps, and
responsible-AI practice.

## Pipeline
`voice -> STT -> translate to EN -> red-flag safety gate -> RAG + LLM -> translate back -> TTS -> voice`
See `docs/ARCHITECTURE.md`.

## Quick start
```bash
pip install -e ".[ml,dev,train]" librosa
ollama pull qwen2.5:3b-instruct
# 1. add public WHO / FMoH guideline PDFs to data/guidelines/ and record them in SOURCES.md
python -m mamavoice.rag.ingest
uvicorn mamavoice.api.main:app --reload
curl -X POST localhost:8000/ask -H 'content-type: application/json' \
     -d '{"text":"How many antenatal visits do I need?","lang":"en"}'
```
Tests and safety gate (no GPU/models needed): `make test`

## Fine-tuning and evaluation (free GPUs)
| Task | Command |
|---|---|
| NLLB LoRA (Kaggle/Colab) | `python training/finetune_nllb.py --lang yo --out outputs/nllb-yo-lora` |
| Translation eval | `python eval/translation_eval.py --lang yo --adapter outputs/nllb-yo-lora` |
| STT eval (WER/CER) | `python eval/stt_eval.py --lang yo` |
| RAG eval (Ragas) | `python eval/rag_eval.py` |
| Safety eval (CI gate) | `make eval-safety` |
| Latency | `python scripts/benchmark.py --lang yo` |
Set `MV_NLLB_ADAPTER=outputs/nllb-yo-lora` to serve the fine-tuned model.

## Results (fill in with your measured numbers)
| Metric | Base | Fine-tuned |
|---|---|---|
| yo->en chrF++ | | |
| STT WER (yo / ha) | | |
| RAG faithfulness | | |
| Red-flag recall | 1.00 on 35-case seed set | |
| p95 latency (CPU) | | |

## Deployment
Docker (`docker compose up`), CI in `.github/workflows`, Telegram bot in `bot/`, Hugging Face Spaces demo.
Low-bandwidth: int8/GGUF models, 16 kbps Opus audio (`scripts/to_opus.sh`), text-first replies.

## Responsible AI
See `docs/RESPONSIBLE_AI.md` and `docs/MODEL_CARD.md`. Not a medical device; synthetic data only.
