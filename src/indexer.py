import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import faiss
import numpy as np

from src.schema import ChunkRecord
from src.embedder import MODEL_NAME, EMBEDDING_DIM

INDEX_DIR = Path(__file__).parent.parent / "index"
INDEX_PATH = INDEX_DIR / "recipe.index"
STORE_PATH = INDEX_DIR / "chunk_store.pkl"
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
MANIFEST_PATH = INDEX_DIR / "manifest.json"


def build_index(embeddings: np.ndarray, nlist: int = 64) -> faiss.IndexIVFFlat:
    """Build an IVFFlat index over L2-normalised embeddings.
    L2 on normalised vectors = cosine similarity ranking."""
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(embeddings)
    index.add(embeddings)
    return index


def save_artifacts(
    index: faiss.IndexIVFFlat,
    chunk_store: list[ChunkRecord],
    embeddings: np.ndarray,
    nlist: int = 64,
) -> None:
    INDEX_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    with open(STORE_PATH, "wb") as f:
        pickle.dump(chunk_store, f, protocol=5)
    np.save(str(EMBEDDINGS_PATH), embeddings)
    manifest = {
        "model_name": MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "nlist": nlist,
        "chunk_count": len(chunk_store),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def load_artifacts() -> tuple[faiss.IndexIVFFlat, list[ChunkRecord], np.ndarray]:
    for path in (INDEX_PATH, STORE_PATH, EMBEDDINGS_PATH, MANIFEST_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Index artifact not found: {path}\n"
                "Run: python scripts/build_index.py"
            )

    manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest["model_name"] != MODEL_NAME:
        raise EnvironmentError(
            f"Index was built with model '{manifest['model_name']}' "
            f"but current embedder uses '{MODEL_NAME}'. "
            "Rebuild the index: python scripts/build_index.py"
        )

    index = faiss.read_index(str(INDEX_PATH))
    with open(STORE_PATH, "rb") as f:
        chunk_store = pickle.load(f)
    embeddings = np.load(str(EMBEDDINGS_PATH))
    return index, chunk_store, embeddings


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())
