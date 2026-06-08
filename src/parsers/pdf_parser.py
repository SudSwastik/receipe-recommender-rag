from __future__ import annotations

import re
from pathlib import Path
from src.schema import RawRecipe
from src.parsers.base import BaseParser

_META_RE = re.compile(
    r"Tags:\s*(.+?)\s*\|+\s*Prep:\s*(\d+)\s*min\s*\|+\s*Cook:\s*(\d+)\s*min\s*\|+\s*Serves:\s*(.+)",
    re.IGNORECASE,
)
_SECTION_HEADINGS = {"ingredients", "method", "chef tips"}
# ReportLab renders the bullet as unicode bullet and arrow chars
_BULLET_PREFIX = re.compile(r"^[•→\-\*]\s*")
_NUMBERED_PREFIX = re.compile(r"^\d+\.\s*")


class PdfParser(BaseParser):
    def parse(self, file_path: str) -> RawRecipe:
        import pdfplumber

        with pdfplumber.open(str(file_path)) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )

        lines = full_text.splitlines()

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

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            lower = stripped.lower()

            # Section headings appear as standalone lines
            if lower in _SECTION_HEADINGS:
                if lower == "ingredients":
                    current_section = "ingredients"
                elif lower == "method":
                    current_section = "method"
                elif "tip" in lower:
                    current_section = "tips"
                continue

            # First non-empty non-section line before metadata = title
            if not title:
                # Skip if it looks like a meta line
                if not _META_RE.search(stripped):
                    title = stripped
                    continue

            # Metadata line
            if not tags:
                m = _META_RE.search(stripped)
                if m:
                    raw_tags = m.group(1)
                    # Strip HTML italic tags that pdfplumber may leave
                    raw_tags = re.sub(r"<[^>]+>", "", raw_tags)
                    tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                    prep_time_mins = int(m.group(2))
                    cook_time_mins = int(m.group(3))
                    serves = m.group(4).strip()
                    continue

            if current_section:
                # Strip decorative bullet/arrow/numbering prefixes
                clean = _BULLET_PREFIX.sub("", stripped)
                clean = _NUMBERED_PREFIX.sub("", clean).strip()
                if clean:
                    section_lines[current_section].append(clean)

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
