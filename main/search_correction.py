from dataclasses import dataclass, field

from autocorrect import Speller, load_from_tar


@dataclass
class SpellCorrection:

    _spell: Speller = field(
        default_factory=lambda: Speller(
            "ru",
            fast=True,
            nlp_data=load_from_tar("https://ipfs.io/ipfs/QmbRSZvfJV6zN12zzWhecphcvE9ZBeQdAJGQ9c9ttJXzcg/ru.tar.gz")
        )
    )

    def __call__(self, search_query: str) -> str:
        return self._spell(search_query)
