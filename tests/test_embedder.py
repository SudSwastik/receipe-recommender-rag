import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


def test_env_var_conflict_raises():
    os.environ["RECIPE_EMBEDDER_MODEL"] = "some-other-model"
    try:
        from src.embedder import RecipeEmbedder
        with pytest.raises(EnvironmentError, match="conflicts"):
            RecipeEmbedder("all-MiniLM-L6-v2")
    finally:
        del os.environ["RECIPE_EMBEDDER_MODEL"]


def test_env_var_match_passes(monkeypatch):
    monkeypatch.setenv("RECIPE_EMBEDDER_MODEL", "all-MiniLM-L6-v2")
    from src.embedder import RecipeEmbedder
    embedder = RecipeEmbedder()  # should not raise
    assert embedder.model_name == "all-MiniLM-L6-v2"


def test_embed_query_shape():
    from src.embedder import RecipeEmbedder
    embedder = RecipeEmbedder()
    vec = embedder.embed_query("chicken soup")
    assert vec.shape == (1, 384)
    assert vec.dtype == np.float32


def test_embeddings_are_normalised():
    from src.embedder import RecipeEmbedder
    embedder = RecipeEmbedder()
    vec = embedder.embed_query("test")
    norm = np.linalg.norm(vec[0])
    assert abs(norm - 1.0) < 1e-5


def test_batch_embed_shape():
    from src.embedder import RecipeEmbedder
    embedder = RecipeEmbedder()
    texts = ["chicken soup", "vegan salad", "pasta"]
    vecs = embedder.embed(texts, batch_size=2)
    assert vecs.shape == (3, 384)
