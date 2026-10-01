"""
Query routing: decide GREETING / DOCUMENT_QUESTION / OUT_OF_SCOPE.

No LLM call anywhere in this module — a small-model classifier turned out
too unreliable (see eval notes). Routing is score-based instead:
- A short, explicit greeting list catches small talk for free.
- Everything else is routed from the top retrieval similarity score:
  retrieval already has to run for a real document question, so it's
  run first and its own top score decides whether the question is
  in scope, instead of asking the LLM a separate question about itself.

MIN_SCORE (config.py) is the threshold; calibrate it with
eval/calibrate_threshold.py rather than guessing.
"""

from enum import Enum

from . import config


class Route(str, Enum):
    GREETING = "GREETING"
    DOCUMENT_QUESTION = "DOCUMENT_QUESTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"


# Explicit greeting list only — exact/near match (whole message, not a
# substring), so a real question that happens to start with "hi" is never
# caught here. Covers the project's languages (French, English, Arabic).
_GREETINGS = {
    "hi", "hello", "hey", "yo",
    "thanks", "thank you", "thankyou",
    "salut", "bonjour", "bonsoir", "coucou", "merci",
    "salam", "salam alaykoum", "assalamu alaykum", "marhaba", "shukran",
}


def is_greeting(question: str) -> bool:
    """True if the whole message, normalized, is a known greeting —
    stripped of case and trailing punctuation, not fuzzy-matched."""
    normalized = question.strip().lower().rstrip("!.?,")
    return normalized in _GREETINGS


def classify_by_score(top_score: float) -> Route:
    """Route a non-greeting question from its top retrieval score."""
    return Route.DOCUMENT_QUESTION if top_score >= config.MIN_SCORE else Route.OUT_OF_SCOPE
