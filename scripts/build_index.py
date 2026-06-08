"""
Build the recipe index from scratch.

    python scripts/build_index.py [--data-dir data] [--nlist 64]

Phases:
    1. Discover .txt / .docx / .pdf files
    2. Parse into RawRecipe objects
    3. Chunk into ChunkRecord objects (one per section)
    4. Embed all chunks with SentenceTransformer
    5. Build IVFFlat FAISS index
    6. Save index + chunk_store + embeddings + manifest
"""
import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent))

from tqdm import tqdm

from src.schema import RawRecipe, ChunkRecord
from src.parsers import TxtParser, DocxParser, PdfParser
from src.chunker import recipe_to_chunks
from src.embedder import RecipeEmbedder
from src.indexer import build_index, save_artifacts


_PARSERS = {
    ".txt": TxtParser(),
    ".docx": DocxParser(),
    ".pdf": PdfParser(),
}


def discover_files(data_dir: Path) -> list[Path]:
    files = []
    for ext in _PARSERS:
        files.extend(data_dir.glob(f"*{ext}"))
    return sorted(files)


def parse_all(files: list[Path]) -> tuple[list[RawRecipe], list[str]]:
    recipes, errors = [], []
    for f in tqdm(files, desc="Parsing"):
        parser = _PARSERS[f.suffix]
        try:
            recipes.append(parser.parse(str(f)))
        except Exception as exc:
            errors.append(f"{f.name}: {exc}")
    return recipes, errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Build recipe FAISS index")
    parser.add_argument("--data-dir", default="data", help="Recipe data directory")
    parser.add_argument(
        "--nlist",
        type=int,
        default=64,
        help="IVFFlat cluster count (default 64; use 256 at 50K chunks)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        sys.exit(f"Data directory not found: {data_dir}")

    # Phase 1: Discover
    files = discover_files(data_dir)
    print(f"\n[1/5] Found {len(files)} recipe files")

    # Phase 2: Parse
    recipes, errors = parse_all(files)
    print(f"[2/5] Parsed {len(recipes)} recipes ({len(errors)} errors)")
    if errors:
        for e in errors[:10]:
            print(f"      WARN {e}")
        if len(errors) > 10:
            print(f"      ... and {len(errors) - 10} more")

    if not recipes:
        sys.exit("No recipes parsed — aborting.")

    # Phase 3: Chunk
    chunk_store: list[ChunkRecord] = []
    for recipe in tqdm(recipes, desc="Chunking"):
        chunk_store.extend(recipe_to_chunks(recipe))

    section_counts = {}
    for c in chunk_store:
        section_counts[c.section] = section_counts.get(c.section, 0) + 1
    print(
        f"[3/5] {len(chunk_store)} chunks  "
        f"(ingredients={section_counts.get('ingredients', 0)}, "
        f"method={section_counts.get('method', 0)}, "
        f"tips={section_counts.get('tips', 0)})"
    )

    # Phase 4: Embed
    print(f"[4/5] Embedding {len(chunk_store)} chunks …")
    embedder = RecipeEmbedder()
    texts = [c.text for c in chunk_store]
    embeddings = embedder.embed(texts, batch_size=128)
    print(f"      Embedding shape: {embeddings.shape}")

    # Phase 5: Index + Save
    print(f"[5/5] Building IVFFlat index (nlist={args.nlist}) …")
    index = build_index(embeddings, nlist=args.nlist)
    save_artifacts(index, chunk_store, embeddings, nlist=args.nlist)
    print(
        f"\n✓ Done. Index saved to index/\n"
        f"  Chunks: {index.ntotal}  nlist: {args.nlist}  "
        f"Model: all-MiniLM-L6-v2\n"
        f"\nRun the app:  streamlit run app/streamlit_app.py"
    )


if __name__ == "__main__":
    main()
