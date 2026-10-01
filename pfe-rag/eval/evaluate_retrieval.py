"""
Retrieval quality on the golden dataset: does the search find the page
of the course where the answer is?

For each golden question, the top-k chunks are retrieved and compared
with the expected pages listed in golden_dataset.json:
- hit@1: the first chunk comes from an expected page
- hit@k: at least one of the k chunks comes from an expected page
- MRR:   mean of 1/rank of the first chunk from an expected page

No LLM call — this measures retrieval only, not the generated answer.

Usage: python -m eval.evaluate_retrieval [--top-k 5]
"""

import argparse
import json
from pathlib import Path

from rag import config
from rag.retrieve import search

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval hit rate on the golden dataset.")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    args = parser.parse_args()

    golden = json.loads(GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    hits_1 = hits_k = 0
    reciprocal_ranks = []

    for item in golden:
        chunks = search(item["question"], top_k=args.top_k)
        pages = [c.page_number for c in chunks]
        # rank (1-based) of the first chunk coming from an expected page
        rank = next((i for i, p in enumerate(pages, start=1) if p in item["pages"]), None)
        hits_1 += rank == 1
        hits_k += rank is not None
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        mark = f"rank {rank}" if rank else "MISS  "
        print(f"  {mark:<7} top={chunks[0].score:.3f}  pages={pages}  expected={item['pages']}  {item['question']}")

    n = len(golden)
    print()
    print(f"hit@1 = {hits_1}/{n} ({hits_1 / n:.0%})")
    print(f"hit@{args.top_k} = {hits_k}/{n} ({hits_k / n:.0%})")
    print(f"MRR   = {sum(reciprocal_ranks) / n:.2f}")


if __name__ == "__main__":
    main()
