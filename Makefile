.PHONY: install test lint index serve eval-safety
install: ; pip install -e ".[ml,dev,train]"
test: ; PYTHONPATH=src:. pytest -q
lint: ; ruff check src tests eval
index: ; python -m mamavoice.rag.ingest
serve: ; uvicorn mamavoice.api.main:app --reload
eval-safety: ; PYTHONPATH=src:. python eval/safety_eval.py
