"""Style package init with venue registry."""

from academicguard.style.base import BaseStyleChecker
from academicguard.style.ieee import IEEEStyleChecker
from academicguard.style.elsevier import ElsevierStyleChecker
from academicguard.style.acm import ACMStyleChecker
from academicguard.style.iet import IETStyleChecker
from academicguard.style.bcs import BCSStyleChecker

VENUE_REGISTRY: dict[str, type[BaseStyleChecker]] = {
    "ieee": IEEEStyleChecker,
    "elsevier": ElsevierStyleChecker,
    "acm": ACMStyleChecker,
    "iet": IETStyleChecker,
    "bcs": BCSStyleChecker,
}

# Default LanguageTool language per venue -- IET and BCS/Oxford require
# British English; the others follow American English convention.
VENUE_LANGUAGE: dict[str, str] = {
    "ieee": "en-US",
    "elsevier": "en-US",
    "acm": "en-US",
    "iet": "en-GB",
    "bcs": "en-GB",
}


def get_venue_language(venue: str) -> str:
    """Return the default grammar-check language for a venue key ('en-US'/'en-GB')."""
    key = venue.lower().strip()
    key = VENUE_ALIASES.get(key, key)
    return VENUE_LANGUAGE.get(key, "en-US")

VENUE_ALIASES: dict[str, str] = {
    "ieee transactions": "ieee",
    "ieee access": "ieee",
    "ieee conference": "ieee",
    "ithenticate": "elsevier",
    "sciencedirect": "elsevier",
    "springer": "elsevier",  # similar guidelines
    "acm dl": "acm",
    "acm sigcomm": "acm",
    "acm ccs": "acm",
    "iet comms": "iet",
    "iet cyber": "iet",
    "computer journal": "bcs",
    "the computer journal": "bcs",
}


def get_style_checker(venue: str) -> BaseStyleChecker:
    """Return the appropriate style checker for the given venue name."""
    key = venue.lower().strip()
    if key in VENUE_ALIASES:
        key = VENUE_ALIASES[key]
    if key not in VENUE_REGISTRY:
        available = ", ".join(sorted(VENUE_REGISTRY.keys()))
        raise ValueError(f"Unknown venue '{venue}'. Available: {available}")
    return VENUE_REGISTRY[key]()


__all__ = [
    "BaseStyleChecker",
    "IEEEStyleChecker",
    "ElsevierStyleChecker",
    "ACMStyleChecker",
    "IETStyleChecker",
    "BCSStyleChecker",
    "VENUE_REGISTRY",
    "VENUE_LANGUAGE",
    "get_style_checker",
    "get_venue_language",
]
