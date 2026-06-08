from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal

SectionType = Literal["ingredients", "method", "tips"]


@dataclass
class RawRecipe:
    recipe_id: str
    source_file: str
    title: str
    tags: list[str]
    prep_time_mins: int
    cook_time_mins: int
    serves: str
    ingredients_text: str
    method_text: str
    tips_text: str


@dataclass
class ChunkRecord:
    chunk_id: str       # "{recipe_id}_{section}"
    recipe_id: str
    title: str
    section: SectionType
    text: str
    tags: list[str]
    prep_time_mins: int
    cook_time_mins: int
    serves: str
    source_file: str

    def to_dict(self) -> dict:
        return asdict(self)
