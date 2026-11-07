"""A tiny deterministic regression set, not a comprehensive RAG benchmark."""
import json
from pathlib import Path
from bank_ml.rag import ProductKnowledge

knowledge = ProductKnowledge()
checks = json.loads(Path("examples/retrieval_checks.json").read_text())
passed = 0
for case in checks:
    sources = knowledge.retrieve(case["product_id"], case["question"], top_k=1)
    actual = sources[0]["id"] if sources else None
    if actual != case["expected"]:
        raise AssertionError((case, actual))
    passed += 1
print(json.dumps({"passed": passed, "total": len(checks), "knowledge_sha256": knowledge.digest,
                  "retriever": "TF-IDF + FAISS", "scope": "handwritten regression checks, not general semantic quality"}, indent=2))
