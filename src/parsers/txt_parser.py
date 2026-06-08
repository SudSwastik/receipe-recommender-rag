from __future__ import annotations

import re
from pathlib import Path
from src.schema import RawRecipe
from src.parsers.base import BaseParser

_PREP_COOK_RE = re.compile(
    r"Prep:\s*(\d+)\s*min\s*\|\s*Cook:\s*(\d+)\s*min\s*\|\s*Serves:\s*(.+)",
    re.IGNORECASE,
)
_SECTION_RE = re.compile(r"^##\s+(Ingredients|Method|Chef Tips)\s*$", re.IGNORECASE)


class TxtParser(BaseParser):
    def parse(self, file_path: str) -> RawRecipe:
        text = Path(file_path).read_text(encoding="utf-8")
        lines = text.splitlines()

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

            if not title and stripped.startswith("Title:"):
                title = stripped[len("Title:"):].strip()
                continue

            if not tags and stripped.startswith("Tags:"):
                raw_tags = stripped[len("Tags:"):].strip()
                tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
                continue

            m = _PREP_COOK_RE.match(stripped)
            if m:
                prep_time_mins = int(m.group(1))
                cook_time_mins = int(m.group(2))
                serves = m.group(3).strip()
                continue

            section_m = _SECTION_RE.match(stripped)
            if section_m:
                heading = section_m.group(1).lower()
                current_section = "tips" if "tip" in heading else heading
                continue

            if current_section and stripped:
                section_lines[current_section].append(stripped)

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
