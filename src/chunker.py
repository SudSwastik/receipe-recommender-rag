from src.schema import RawRecipe, ChunkRecord, SectionType

_SECTIONS: list[tuple[SectionType, str]] = [
    ("ingredients", "ingredients_text"),
    ("method", "method_text"),
    ("tips", "tips_text"),
]


def recipe_to_chunks(recipe: RawRecipe) -> list[ChunkRecord]:
    """Produce one ChunkRecord per non-empty section. Never merges sections."""
    chunks = []
    for section, attr in _SECTIONS:
        text = getattr(recipe, attr).strip()
        if not text:
            continue
        chunks.append(ChunkRecord(
            chunk_id=f"{recipe.recipe_id}_{section}",
            recipe_id=recipe.recipe_id,
            title=recipe.title,
            section=section,
            text=text,
            tags=list(recipe.tags),
            prep_time_mins=recipe.prep_time_mins,
            cook_time_mins=recipe.cook_time_mins,
            serves=recipe.serves,
            source_file=recipe.source_file,
        ))
    return chunks
