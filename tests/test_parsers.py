"""Parser tests against the first few real data files."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from src.parsers import TxtParser, DocxParser, PdfParser

DATA = Path(__file__).parent.parent / "data"


def _first_file(ext: str) -> Path | None:
    files = sorted(DATA.glob(f"*{ext}"))
    return files[0] if files else None


class TestTxtParser:
    def setup_method(self):
        self.parser = TxtParser()

    def test_parses_title(self):
        f = _first_file(".txt")
        if f is None:
            pytest.skip("No .txt files found")
        recipe = self.parser.parse(str(f))
        assert recipe.title

    def test_parses_tags(self):
        f = _first_file(".txt")
        if f is None:
            pytest.skip("No .txt files found")
        recipe = self.parser.parse(str(f))
        assert isinstance(recipe.tags, list)
        assert len(recipe.tags) > 0

    def test_parses_times(self):
        f = _first_file(".txt")
        if f is None:
            pytest.skip("No .txt files found")
        recipe = self.parser.parse(str(f))
        assert recipe.prep_time_mins >= 0
        assert recipe.cook_time_mins >= 0

    def test_parses_sections(self):
        f = _first_file(".txt")
        if f is None:
            pytest.skip("No .txt files found")
        recipe = self.parser.parse(str(f))
        assert recipe.ingredients_text
        assert recipe.method_text
        assert recipe.tips_text

    def test_recipe_id_from_stem(self):
        f = _first_file(".txt")
        if f is None:
            pytest.skip("No .txt files found")
        recipe = self.parser.parse(str(f))
        assert recipe.recipe_id == f.stem

    def test_sections_are_separate(self):
        f = _first_file(".txt")
        if f is None:
            pytest.skip("No .txt files found")
        recipe = self.parser.parse(str(f))
        # Section headers should not appear in content
        assert "## Ingredients" not in recipe.ingredients_text
        assert "## Method" not in recipe.method_text
        assert "## Chef Tips" not in recipe.tips_text


class TestDocxParser:
    def setup_method(self):
        self.parser = DocxParser()

    def test_parses_docx(self):
        f = _first_file(".docx")
        if f is None:
            pytest.skip("No .docx files found")
        recipe = self.parser.parse(str(f))
        assert recipe.title
        assert recipe.tags
        assert recipe.ingredients_text
        assert recipe.method_text


class TestPdfParser:
    def setup_method(self):
        self.parser = PdfParser()

    def test_parses_pdf(self):
        f = _first_file(".pdf")
        if f is None:
            pytest.skip("No .pdf files found")
        recipe = self.parser.parse(str(f))
        assert recipe.title
        assert recipe.ingredients_text
        assert recipe.method_text
