from __future__ import annotations

import time

import faiss
import numpy as np

from src.schema import ChunkRecord
from src.embedder import RecipeEmbedder
from src.filters import extract_dietary_filters, apply_tag_filter
from src.indexer import load_artifacts

_embedder: RecipeEmbedder | None = None
_index: faiss.IndexIVFFlat | None = None
_chunk_store: list[ChunkRecord] | None = None
_embeddings: np.ndarray | None = None


def _load() -> tuple[RecipeEmbedder, faiss.IndexIVFFlat, list[ChunkRecord], np.ndarray]:
    """Lazy singleton — loads once per process."""
    global _embedder, _index, _chunk_store, _embeddings
    if _embedder is None:
        _embedder = RecipeEmbedder()
    if _index is None:
        _index, _chunk_store, _embeddings = load_artifacts()
    return _embedder, _index, _chunk_store, _embeddings


def retrieve_recipes(
    query: str,
    top_k: int = 5,
    nprobe: int = 10,
    use_filter: bool = True,
) -> tuple[list[dict], float]:
    """Return (results, latency_ms).

    Each result dict contains all ChunkRecord fields plus a 'score' key
    (lower L2 distance = better match on normalised vectors).
    """
    embedder, index, chunk_store, embeddings = _load()

    query_vec = embedder.embed_query(query)  # (1, 384)
    dietary_filters = extract_dietary_filters(query) if use_filter else set()

    t0 = time.perf_counter()

    if dietary_filters:
        filtered_chunks, original_indices = apply_tag_filter(chunk_store, dietary_filters)
        if not filtered_chunks:
            latency_ms = (time.perf_counter() - t0) * 1000
            return [], latency_ms

        filtered_vecs = embeddings[original_indices].astype(np.float32)
        tmp_index = faiss.IndexFlatL2(embeddings.shape[1])
        tmp_index.add(filtered_vecs)
        k = min(top_k, len(filtered_chunks))
        distances, local_ids = tmp_index.search(query_vec, k)

        results = []
        for dist, local_idx in zip(distances[0], local_ids[0]):
            if local_idx == -1:
                continue
            r = filtered_chunks[local_idx].to_dict()
            r["score"] = float(dist)
            results.append(r)
    else:
        index.nprobe = nprobe
        distances, ids = index.search(query_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], ids[0]):
            if idx == -1:
                continue
            r = chunk_store[idx].to_dict()
            r["score"] = float(dist)
            results.append(r)

    latency_ms = (time.perf_counter() - t0) * 1000
    return results, latency_ms
