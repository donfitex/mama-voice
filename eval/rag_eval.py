"""RAG faithfulness/relevance with Ragas, using a local Ollama judge (free).
Fill eval/data/rag_questions.jsonl with {"question": ..., "ground_truth": ...} written from your
guideline documents. pip install ragas langchain-community
"""
import json
from pathlib import Path

from datasets import Dataset

from mamavoice.llm.generate import generate
from mamavoice.rag.retriever import Retriever


def main():
    r = Retriever()
    rows = [json.loads(l) for l in Path("eval/data/rag_questions.jsonl").read_text().splitlines()]
    data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}
    for row in rows:
        hits = r.search(row["question"])
        data["question"].append(row["question"])
        data["answer"].append(generate(row["question"], hits))
        data["contexts"].append([h.text for h in hits])
        data["ground_truth"].append(row["ground_truth"])
    from langchain_community.chat_models import ChatOllama
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness
    res = evaluate(Dataset.from_dict(data), metrics=[faithfulness, answer_relevancy, context_precision],
                   llm=ChatOllama(model="qwen2.5:3b-instruct"))
    print(res)


if __name__ == "__main__":
    main()
