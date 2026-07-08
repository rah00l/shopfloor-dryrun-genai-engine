"""
Cache — simple in-memory key-value cache for repeat queries (CAG).

Design decisions:
    - Plain Python dict, in-process, no external service
    - Normalize queries before caching (lowercase, strip, collapse whitespace)
      so near-identical phrasing still hits the cache
    - No expiry needed for a demo session — cache lives for the process lifetime
"""

import re

_cache = {}


def normalize(query: str) -> str:
    """Lowercase, strip, and collapse whitespace so minor phrasing differences still match."""
    q = query.lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\s]", "", q)
    return q


def get(query: str):
    """Return the cached result for this query, or None if not cached."""
    key = normalize(query)
    return _cache.get(key)


def set(query: str, result: dict):
    """Store a result under this query's normalized key."""
    key = normalize(query)
    _cache[key] = result


def clear():
    """Clear the entire cache — useful for testing."""
    _cache.clear()


def size() -> int:
    """Number of entries currently cached."""
    return len(_cache)
