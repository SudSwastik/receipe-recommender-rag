import os
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class RecipeEmbedder:
    """
    Wraps SentenceTransformer with guards against the HashEmbedder trap:
    a random/hash-based embedding that produces meaningless cosine similarity.
    """

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        from sentence_transformers import SentenceTransformer

        self._check_env_consistency(model_name)
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

        if not isinstance(self.model, SentenceTransformer):
            raise TypeError(
                f"model must be SentenceTransformer, got {type(self.model).__name__}. "
                "Hash or stub embedders are not allowed."
            )

        self.dim = self.model.get_sentence_embedding_dimension()
        if self.dim != EMBEDDING_DIM:
            raise AssertionError(
                f"Embedding dim mismatch: expected {EMBEDDING_DIM}, got {self.dim}. "
                "Rebuild the index if you changed models."
            )

    @staticmethod
    def _check_env_consistency(model_name: str) -> None:
        """Cross-check with RECIPE_EMBEDDER_MODEL env var when set."""
        env_model = os.environ.get("RECIPE_EMBEDDER_MODEL")
        if env_model is not None and env_model != model_name:
            raise EnvironmentError(
                f"RECIPE_EMBEDDER_MODEL='{env_model}' conflicts with "
                f"requested model '{model_name}'. "
                "Ingestion and retrieval must use the same model."
            )

    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        """Return float32 array of shape (N, 384), L2-normalised."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Return float32 array of shape (1, 384), L2-normalised."""
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
