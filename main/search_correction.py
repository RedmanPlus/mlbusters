from dataclasses import dataclass, field

from autocorrect import Speller


@dataclass
class SpellCorrection:

    _spell: Speller = field(default_factory=lambda: Speller("ru", fast=True))

    def __call__(self, search_query: str) -> str:
        return self._spell(search_query)
