from src.schema import ChunkRecord

DIETARY_KEYWORDS: dict[str, list[str]] = {
    "vegan": ["vegan"],
    "vegetarian": ["vegetarian", "vegan"],
    "gluten-free": ["gluten-free"],
    "gluten free": ["gluten-free"],
    "keto": ["keto", "low-carb"],
    "low-carb": ["low-carb", "keto"],
    "low carb": ["low-carb", "keto"],
    "dairy-free": ["dairy-free"],
    "dairy free": ["dairy-free"],
    "quick": ["quick"],
    "healthy": ["healthy"],
}


def extract_dietary_filters(query: str) -> set[str]:
    query_lower = query.lower()
    filters: set[str] = set()
    for keyword, tags in DIETARY_KEYWORDS.items():
        if keyword in query_lower:
            filters.update(tags)
    return filters


def apply_tag_filter(
    chunk_store: list[ChunkRecord],
    required_tags: set[str],
) -> tuple[list[ChunkRecord], list[int]]:
    """Return (filtered_chunks, original_indices) where original_indices maps
    filtered position back to chunk_store position."""
    if not required_tags:
        return chunk_store, list(range(len(chunk_store)))
    filtered, indices = [], []
    for i, chunk in enumerate(chunk_store):
        if required_tags.intersection(chunk.tags):
            filtered.append(chunk)
            indices.append(i)
    return filtered, indices
