"""
Calibrate MIN_SCORE (config.py) for score-based routing (see router.py).

Runs the golden dataset (in-scope questions — should route
DOCUMENT_QUESTION) and 10 off-topic questions (should route OUT_OF_SCOPE)
through retrieval, then for each candidate threshold prints:
- true-positive rate:  fraction of golden questions scored >= threshold
- false-positive rate: fraction of off-topic questions scored >= threshold

Pick MIN_SCORE where TPR is high and FPR is low — the raw scores printed
at the end help see how much margin there is around that choice.

Usage: python -m eval.calibrate_threshold
"""

import json
from pathlib import Path

from rag.retrieve import search

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"

# Generic questions with no connection to the indexed documents.
OFF_TOPIC_QUESTIONS = [
    "What is the capital of Japan?",
    "How do I bake a chocolate cake?",
    "What is the boiling point of water?",
    "Who wrote Romeo and Juliet?",
    "What is the speed of light?",
    "How does photosynthesis work?",
    "What's the weather like today?",
    "Recommend a good sci-fi movie.",
    "What is the tallest mountain in the world?",
    "Explain quantum entanglement.",
]

THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(9)]  # 0.20, 0.25, ..., 0.60


def _top_score(question: str) -> float:
    chunks = search(question, top_k=1)
    return chunks[0].score if chunks else 0.0


def main() -> None:
    golden = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    golden_questions = [item["question"] for item in golden]

    print(f"Scoring {len(golden_questions)} golden (in-scope) questions...")
    in_scope_scores = [(q, _top_score(q)) for q in golden_questions]

    print(f"Scoring {len(OFF_TOPIC_QUESTIONS)} off-topic questions...")
    off_topic_scores = [(q, _top_score(q)) for q in OFF_TOPIC_QUESTIONS]

    print()
    print(f"{'MIN_SCORE':>9} | {'TPR':>6} | {'FPR':>6}")
    print("-" * 28)
    for t in THRESHOLDS:
        tp = sum(1 for _, s in in_scope_scores if s >= t)
        fp = sum(1 for _, s in off_topic_scores if s >= t)
        tpr = tp / len(in_scope_scores)
        fpr = fp / len(off_topic_scores)
        print(f"{t:>9.2f} | {tpr:>6.0%} | {fpr:>6.0%}")

    print()
    print("Raw scores — golden (in-scope):")
    for q, s in sorted(in_scope_scores, key=lambda x: x[1]):
        print(f"  {s:.3f}  {q}")
    print("Raw scores — off-topic:")
    for q, s in sorted(off_topic_scores, key=lambda x: x[1]):
        print(f"  {s:.3f}  {q}")


if __name__ == "__main__":
    main()
