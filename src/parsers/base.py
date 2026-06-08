from abc import ABC, abstractmethod
from src.schema import RawRecipe


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str) -> RawRecipe:
        """Parse a single recipe file. Raises ValueError on malformed input."""
        ...
