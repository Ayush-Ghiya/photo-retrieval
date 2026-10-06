"""Keyword matching of a search query against a photo's own metadata (title, description, tags).

CLIP's text-to-text similarity is high and nearly uniform for any short phrase, so it cannot tell
"goa trip" apart from "tags: ship". User metadata is matched by words instead; CLIP handles the
visual side of the query.
"""
import re

STOPWORDS = frozenset(
    "a an and at by for from in into is it my of on or our the to with "
    "photo photos picture pictures image images pic pics".split()
)


def _fold(word: str) -> str:
    """Cheap plural folding: cars -> car, but glass stays glass."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(text: str) -> list[str]:
    return [_fold(w) for w in re.findall(r"[a-z0-9]+", text.lower())]


def query_terms(q: str) -> list[str]:
    """Distinct meaningful words of the query, in order."""
    terms: list[str] = []
    for w in _words(q):
        if w not in STOPWORDS and w not in terms:
            terms.append(w)
    return terms


def metadata_terms(title: str | None, description: str | None, tags: list[str]) -> set[str]:
    """Words a photo's metadata can be found by; tags like 'goa-trip' contribute 'goa' and 'trip'."""
    return set(_words(" ".join([title or "", description or "", *tags])))


def coverage(terms: list[str], doc_terms: set[str]) -> float:
    """Fraction of query terms present in the photo's metadata (0..1)."""
    if not terms:
        return 0.0
    return sum(t in doc_terms for t in terms) / len(terms)
