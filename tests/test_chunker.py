import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schema import RawRecipe
from src.chunker import recipe_to_chunks


def _make_recipe(**kwargs) -> RawRecipe:
    defaults = dict(
        recipe_id="test_recipe",
        source_file="test.txt",
        title="Test Recipe",
        tags=["vegan", "quick"],
        prep_time_mins=10,
        cook_time_mins=20,
        serves="4",
        ingredients_text="200g flour\n2 eggs",
        method_text="1. Mix ingredients.\n2. Bake.",
        tips_text="Use fresh eggs.",
    )
    defaults.update(kwargs)
    return RawRecipe(**defaults)


def test_three_chunks_for_complete_recipe():
    chunks = recipe_to_chunks(_make_recipe())
    assert len(chunks) == 3


def test_section_types():
    chunks = recipe_to_chunks(_make_recipe())
    sections = {c.section for c in chunks}
    assert sections == {"ingredients", "method", "tips"}


def test_chunk_ids():
    chunks = recipe_to_chunks(_make_recipe())
    for c in chunks:
        assert c.chunk_id == f"test_recipe_{c.section}"


def test_no_section_headers_in_text():
    chunks = recipe_to_chunks(_make_recipe())
    for c in chunks:
        assert "## " not in c.text


def test_empty_tips_gives_two_chunks():
    chunks = recipe_to_chunks(_make_recipe(tips_text=""))
    assert len(chunks) == 2
    sections = {c.section for c in chunks}
    assert "tips" not in sections


def test_metadata_propagated():
    chunks = recipe_to_chunks(_make_recipe())
    for c in chunks:
        assert c.tags == ["vegan", "quick"]
        assert c.prep_time_mins == 10
        assert c.cook_time_mins == 20
        assert c.serves == "4"
        assert c.title == "Test Recipe"
