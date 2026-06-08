import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.filters import extract_dietary_filters, apply_tag_filter
from src.schema import RawRecipe, ChunkRecord


def _make_chunk(tags: list[str]) -> ChunkRecord:
    return ChunkRecord(
        chunk_id="r1_ingredients",
        recipe_id="r1",
        title="Test",
        section="ingredients",
        text="some text",
        tags=tags,
        prep_time_mins=10,
        cook_time_mins=20,
        serves="2",
        source_file="test.txt",
    )


def test_extract_vegan():
    assert extract_dietary_filters("vegan soup") == {"vegan"}


def test_extract_low_carb_with_space():
    filters = extract_dietary_filters("low carb dinner")
    assert "low-carb" in filters


def test_extract_keto():
    filters = extract_dietary_filters("keto breakfast ideas")
    assert "keto" in filters
    assert "low-carb" in filters


def test_no_keywords():
    assert extract_dietary_filters("chicken soup") == set()


def test_apply_tag_filter_matching():
    chunks = [
        _make_chunk(["vegan", "quick"]),
        _make_chunk(["chicken", "dinner"]),
        _make_chunk(["vegan", "gluten-free"]),
    ]
    filtered, indices = apply_tag_filter(chunks, {"vegan"})
    assert len(filtered) == 2
    assert indices == [0, 2]


def test_apply_tag_filter_empty_required():
    chunks = [_make_chunk(["vegan"]), _make_chunk(["chicken"])]
    filtered, indices = apply_tag_filter(chunks, set())
    assert len(filtered) == 2
    assert indices == [0, 1]


def test_apply_tag_filter_no_match():
    chunks = [_make_chunk(["chicken"]), _make_chunk(["beef"])]
    filtered, indices = apply_tag_filter(chunks, {"vegan"})
    assert filtered == []
    assert indices == []
