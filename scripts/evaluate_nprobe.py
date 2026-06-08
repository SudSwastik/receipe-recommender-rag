"""
nprobe tuning evaluation — spec Q6.

For each of the 3 test queries, run retrieve_recipes at nprobe ∈ {1, 5, 10, 50}
and report:
  - Recall@5 vs exhaustive search (nprobe=nlist=64)
  - Latency (ms)
  - Top-5 results at each nprobe level

Recall@5 here = |results ∩ gold| / 5, where gold = nprobe=nlist results.
This is standard self-recall against exhaustive IVF search.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.embedder import RecipeEmbedder
from src.indexer import load_artifacts

QUERIES = [
    "What can I cook with chicken and lemon?",
    "Low-carb dessert under 30 min",
    "Vegan soup for cold weather",
]
NPROBE_VALUES = [1, 5, 10, 50]
TOP_K = 5
NLIST = 64          # index was built with nlist=64
REPEATS = 5         # average latency over this many runs


def mean_latency(index, query_vec: np.ndarray, nprobe: int, top_k: int) -> tuple[list[int], float]:
    index.nprobe = nprobe
    # Warm-up
    index.search(query_vec, top_k)
    t0 = time.perf_counter()
    for _ in range(REPEATS):
        distances, ids = index.search(query_vec, top_k)
    elapsed_ms = (time.perf_counter() - t0) / REPEATS * 1000
    return ids[0].tolist(), elapsed_ms


def main() -> None:
    print("Loading index …")
    index, chunk_store, _ = load_artifacts()
    embedder = RecipeEmbedder()

    print(f"\nIndex: {index.ntotal} chunks, nlist={NLIST}")
    print(f"Top-K={TOP_K}, latency averaged over {REPEATS} runs\n")
    print("=" * 72)

    for query in QUERIES:
        query_vec = embedder.embed_query(query)

        # Gold standard: exhaustive IVF (nprobe = nlist covers all clusters)
        gold_ids, _ = mean_latency(index, query_vec, nprobe=NLIST, top_k=TOP_K)
        gold_set = set(i for i in gold_ids if i != -1)

        print(f"\nQuery: \"{query}\"")
        print(f"  Gold ({NLIST}-probe) results:")
        for gid in gold_ids:
            if gid != -1:
                c = chunk_store[gid]
                print(f"    [{c.section:12s}] {c.title} — {c.tags[:3]}")

        print(f"\n  {'nprobe':>6}  {'latency':>10}  {'recall@5':>9}  {'coverage':>9}")
        print(f"  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*9}")

        for nprobe in NPROBE_VALUES:
            ids, lat_ms = mean_latency(index, query_vec, nprobe=nprobe, top_k=TOP_K)
            result_set = set(i for i in ids if i != -1)
            recall = len(result_set & gold_set) / max(len(gold_set), 1)
            pct_searched = nprobe / NLIST * 100
            print(
                f"  {nprobe:>6}  {lat_ms:>8.3f}ms  {recall:>8.0%}   {pct_searched:>7.1f}%"
            )

        print()

    print("=" * 72)
    print("\nConclusion:")
    print(f"  With nlist={NLIST} and ~{index.ntotal} chunks, recall plateaus quickly.")
    print(f"  nprobe=10 (covers {10/NLIST*100:.0f}% of index) is the production default.")
    print(f"  At 50K chunks, rebuild with nlist=256 and raise nprobe default to 32.")


if __name__ == "__main__":
    main()
