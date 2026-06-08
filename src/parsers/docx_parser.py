from __future__ import annotations

import re
from pathlib import Path
from src.schema import RawRecipe
from src.parsers.base import BaseParser

_META_RE = re.compile(
    r"Tags:\s*(.+?)\s*\|+\s*Prep:\s*(\d+)\s*min\s*\|+\s*Cook:\s*(\d+)\s*min\s*\|+\s*Serves:\s*(.+)",
    re.IGNORECASE,
)
_SECTION_NAMES = {"ingredients", "method", "chef tips"}


class DocxParser(BaseParser):
    def parse(self, file_path: str) -> RawRecipe:
        from docx import Document

        doc = Document(str(file_path))

        title = ""
        tags: list[str] = []
        prep_time_mins = 0
        cook_time_mins = 0
        serves = "1"
        current_section: str | None = None
        section_lines: dict[str, list[str]] = {
            "ingredients": [],
            "method": [],
            "tips": [],
        }

        for para in doc.paragraphs:
            style = para.style.name or ""
            text = para.text.strip()
            if not text:
                continue

            if style.startswith("Heading 1"):
                title = text
                continue

            if style.startswith("Heading 2"):
                lower = text.lower()
                if lower == "ingredients":
                    current_section = "ingredients"
                elif lower == "method":
                    current_section = "method"
                elif "tip" in lower:
                    current_section = "tips"
                else:
                    current_section = None
                continue

            # Metadata line
            if not tags:
                m = _META_RE.search(text)
                if m:
                    tags = [t.strip() for t in m.group(1).split(",") if t.strip()]
                    prep_time_mins = int(m.group(2))
                    cook_time_mins = int(m.group(3))
                    serves = m.group(4).strip()
                    continue

            if current_section:
                section_lines[current_section].append(text)

        if not title:
            raise ValueError(f"No title found in {file_path}")

        recipe_id = Path(file_path).stem
        return RawRecipe(
            recipe_id=recipe_id,
            source_file=str(file_path),
            title=title,
            tags=tags,
            prep_time_mins=prep_time_mins,
            cook_time_mins=cook_time_mins,
            serves=serves,
            ingredients_text="\n".join(section_lines["ingredients"]),
            method_text="\n".join(section_lines["method"]),
            tips_text="\n".join(section_lines["tips"]),
        )
