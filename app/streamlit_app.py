"""Recipe Recommender — Streamlit UI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.retriever import retrieve_recipes
from src.filters import extract_dietary_filters, DIETARY_KEYWORDS
from src.indexer import load_artifacts, load_manifest

st.set_page_config(
    page_title="Recipe Recommender",
    page_icon="🍳",
    layout="wide",
)

# ── Load index once per process ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading index…")
def _load_index():
    """Load FAISS index, chunk store, embeddings, and embedder."""
    from src.embedder import RecipeEmbedder
    try:
        index, chunk_store, embeddings = load_artifacts()
        embedder = RecipeEmbedder()
        return index, chunk_store, embeddings, embedder, None
    except FileNotFoundError as e:
        return None, None, None, None, str(e)


index, chunk_store, embeddings, embedder, load_error = _load_index()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")

    top_k = st.slider("Results (top-K)", min_value=1, max_value=20, value=5)
    nprobe = st.slider(
        "nprobe",
        min_value=1, max_value=64, value=10,
        help="IVFFlat cells to visit. Higher = more accurate, slower.",
    )
    use_filter = st.toggle("Dietary pre-filter", value=True,
                           help="Auto-detect vegan/keto/gluten-free keywords and filter before vector search.")

    section_filter = st.multiselect(
        "Show sections",
        options=["ingredients", "method", "tips"],
        default=["ingredients", "method", "tips"],
    )

    compare_mode = st.checkbox(
        "Compare: filtered vs unfiltered",
        help="Run the query twice and show results side by side.",
    )

    st.divider()
    st.subheader("Index info")
    manifest = load_manifest()
    if manifest:
        st.caption(f"Model: `{manifest.get('model_name', '?')}`")
        st.caption(f"Chunks: {manifest.get('chunk_count', '?'):,}")
        st.caption(f"nlist: {manifest.get('nlist', '?')}")
        built = manifest.get("built_at", "")
        if built:
            st.caption(f"Built: {built[:10]}")
    else:
        st.caption("Index not found — run `build_index.py`")

# ── Main area ─────────────────────────────────────────────────────────────────

st.title("🍳 Recipe Recommender")
st.caption("Ask anything — ingredients you have, dietary needs, cuisine, meal type.")

if load_error:
    st.error(f"Index not loaded: {load_error}")
    st.code("python scripts/build_index.py", language="bash")
    st.stop()

query = st.text_input(
    "What are you looking for?",
    placeholder='e.g. "low-carb chicken under 30 minutes" or "vegan soup for cold weather"',
)

col_search, col_clear = st.columns([1, 5])
with col_search:
    search_clicked = st.button("Search", type="primary", use_container_width=True)
with col_clear:
    if st.button("Clear"):
        st.session_state.pop("last_query", None)
        st.rerun()

if search_clicked and query.strip():
    st.session_state["last_query"] = query.strip()

active_query = st.session_state.get("last_query", "")

if not active_query:
    st.info("Enter a query above to find matching recipes.")
    st.stop()

# Show detected dietary filters
detected = extract_dietary_filters(active_query) if use_filter else set()
if detected:
    st.markdown(
        "**Dietary filters detected:** "
        + " ".join(f"`{t}`" for t in sorted(detected))
    )

# ── Run retrieval ─────────────────────────────────────────────────────────────

def _run(q: str, k: int, np_: int, filt: bool) -> tuple[list[dict], float]:
    return retrieve_recipes(q, top_k=k, nprobe=np_, use_filter=filt)


def _render_results(results: list[dict], latency_ms: float, label: str = "") -> None:
    visible = [r for r in results if r["section"] in section_filter]
    header = f"**{label}** — " if label else ""
    st.markdown(f"{header}{len(visible)} result(s) in **{latency_ms:.1f} ms**")

    if not visible:
        st.warning("No results. Try a broader query or disable the dietary filter.")
        return

    for i, r in enumerate(visible, 1):
        similarity = max(0.0, 1.0 - r["score"] / 2)  # cosine sim ≈ 1 - L2²/2 for unit vecs
        total_time = r["prep_time_mins"] + r["cook_time_mins"]
        title_line = (
            f"**#{i}  {r['title']}** — "
            f"`{r['section']}` · "
            f"{', '.join(r['tags'][:4])} · "
            f"Prep {r['prep_time_mins']}m / Cook {r['cook_time_mins']}m · "
            f"Match {similarity:.0%}"
        )
        with st.expander(title_line):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{r['section'].title()} text**")
                st.text(r["text"])
            with col2:
                st.markdown("**Metadata**")
                st.write({
                    "recipe_id": r["recipe_id"],
                    "section": r["section"],
                    "tags": r["tags"],
                    "prep_time_mins": r["prep_time_mins"],
                    "cook_time_mins": r["cook_time_mins"],
                    "total_mins": total_time,
                    "serves": r["serves"],
                    "score (L2)": f"{r['score']:.4f}",
                })


if compare_mode and use_filter and detected:
    col_filtered, col_unfiltered = st.columns(2)
    with col_filtered:
        st.subheader("With dietary filter")
        results_f, lat_f = _run(active_query, top_k, nprobe, True)
        _render_results(results_f, lat_f, "Filtered")
    with col_unfiltered:
        st.subheader("Without dietary filter")
        results_u, lat_u = _run(active_query, top_k, nprobe, False)
        _render_results(results_u, lat_u, "Unfiltered")
else:
    results, latency_ms = _run(active_query, top_k, nprobe, use_filter)
    _render_results(results, latency_ms)
