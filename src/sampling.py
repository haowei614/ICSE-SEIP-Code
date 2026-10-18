"""Deterministic (seeded) sampling for the pilot.

- `stratified_sample` keeps class balance where a stratum key is available
  (e.g. the gold label for classification), so a 30-50 item pilot still covers
  minority classes.
- Determinism: same (items, n, seed) -> same selection and same order. This is a
  correctness property the tests assert.

Never invents items; only selects from what the loader returned.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Optional, Sequence, TypeVar

T = TypeVar("T")


def _stable_key(item_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{item_id}".encode("utf-8")).hexdigest()


def deterministic_sample(items: Sequence[T], n: Optional[int], seed: int,
                         id_of: Callable[[T], str]) -> list[T]:
    """Deterministically pick up to n items, ordered by a seeded hash of the id."""
    ordered = sorted(items, key=lambda it: _stable_key(id_of(it), seed))
    return list(ordered) if n is None else ordered[:n]


def stratified_sample(items: Sequence[T], n: Optional[int], seed: int,
                      id_of: Callable[[T], str],
                      stratum_of: Optional[Callable[[T], Optional[str]]] = None
                      ) -> list[T]:
    """Deterministic stratified sample.

    If `stratum_of` is None or returns None for all items, falls back to a flat
    deterministic sample. Otherwise allocates n across strata proportionally
    (largest-remainder), then deterministically selects within each stratum.
    The final list is re-ordered by the same seeded hash for stable iteration.
    """
    if n is None:
        return deterministic_sample(items, None, seed, id_of)
    if stratum_of is None:
        return deterministic_sample(items, n, seed, id_of)

    buckets: dict[str, list[T]] = {}
    for it in items:
        key = stratum_of(it)
        buckets.setdefault(key if key is not None else "__none__", []).append(it)
    if set(buckets) == {"__none__"}:
        return deterministic_sample(items, n, seed, id_of)

    total = sum(len(v) for v in buckets.values())
    # proportional quotas with largest-remainder rounding
    raw = {k: (len(v) / total) * n for k, v in buckets.items()}
    floors = {k: int(v) for k, v in raw.items()}
    allocated = sum(floors.values())
    remainder = n - allocated
    # distribute leftover to the largest fractional parts (deterministic tie-break by key)
    order = sorted(buckets.keys(), key=lambda k: (-(raw[k] - floors[k]), k))
    for k in order[:max(0, remainder)]:
        floors[k] += 1

    picked: list[T] = []
    for k, group in buckets.items():
        take = min(floors.get(k, 0), len(group))
        picked.extend(deterministic_sample(group, take, seed, id_of))
    # global stable ordering
    return deterministic_sample(picked, None, seed, id_of)
