from mamavoice.rag.retriever import Retriever

r = Retriever()
for q in [
    "How many antenatal visits should I attend?",
    "What should I eat during pregnancy?",
    "Who won the football world cup?",
    "How do I fix my car engine?",
]:
    hits = r.search(q)
    print(f"{hits[0].score:.3f}  {q}")