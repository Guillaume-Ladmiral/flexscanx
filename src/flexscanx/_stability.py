from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass
class StabilityThresholds:
    mean_jaccard: float = 0.90
    min_jaccard: float = 0.70
    coverage: float = 0.90
    count_tol: int = 1
    coverage_floor: float = 0.50

    def __post_init__(self) -> None:
        for name in ("mean_jaccard", "min_jaccard", "coverage", "coverage_floor"):
            raw_value = getattr(self, name)
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must be finite and in [0,1]") from exc
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1], got {raw_value!r}")
            setattr(self, name, value)
        if isinstance(self.count_tol, (bool, np.bool_)) or not isinstance(
            self.count_tol, (int, np.integer)
        ):
            raise ValueError(f"count_tol must be a non-negative integer, got {self.count_tol!r}")
        if int(self.count_tol) < 0:
            raise ValueError(f"count_tol must be a non-negative integer, got {self.count_tol!r}")
        self.count_tol = int(self.count_tol)


@dataclass
class StabilityMetrics:
    mean_jaccard: float
    min_jaccard: float
    coverage: float
    count_diff: int
    nA: int
    nB: int
    matched: list[tuple[int, int, float]]


def _normalize_cluster(cluster: Iterable[int]) -> frozenset[int]:
    return frozenset(int(x) for x in cluster)


def jaccard_similarity(a: Iterable[int], b: Iterable[int]) -> float:
    """Return the Jaccard similarity between two cluster memberships."""
    set_a = _normalize_cluster(a)
    set_b = _normalize_cluster(b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return float(inter) / float(union) if union > 0 else 1.0


def match_clusters_jaccard(
    sets_a: Sequence[Iterable[int]],
    sets_b: Sequence[Iterable[int]],
    *,
    coverage_floor: float = 0.5,
) -> StabilityMetrics:
    """Greedy maximum-Jaccard matching and stability metrics between two cluster pools."""
    try:
        coverage_floor_f = float(coverage_floor)
    except (TypeError, ValueError) as exc:
        raise ValueError("coverage_floor must be finite and in [0,1]") from exc
    if not np.isfinite(coverage_floor_f) or not 0.0 <= coverage_floor_f <= 1.0:
        raise ValueError(f"coverage_floor must be finite and in [0,1], got {coverage_floor!r}")
    pool_a = [_normalize_cluster(cluster) for cluster in sets_a]
    pool_b = [_normalize_cluster(cluster) for cluster in sets_b]
    n_a = len(pool_a)
    n_b = len(pool_b)

    if n_a == 0 and n_b == 0:
        return StabilityMetrics(
            mean_jaccard=1.0,
            min_jaccard=1.0,
            coverage=1.0,
            count_diff=0,
            nA=0,
            nB=0,
            matched=[],
        )

    pairs: list[tuple[int, int, float]] = []
    for i, cluster_a in enumerate(pool_a):
        for j, cluster_b in enumerate(pool_b):
            jac = jaccard_similarity(cluster_a, cluster_b)
            if jac > 0.0:
                pairs.append((i, j, jac))
    pairs.sort(key=lambda item: item[2], reverse=True)

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for i, j, jac in pairs:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matched.append((i, j, jac))

    if not matched:
        all_union: set[int] = set().union(*pool_a) | set().union(*pool_b)
        return StabilityMetrics(
            mean_jaccard=0.0,
            min_jaccard=0.0,
            coverage=(0.0 if all_union else 1.0),
            count_diff=abs(n_a - n_b),
            nA=n_a,
            nB=n_b,
            matched=[],
        )

    # A cluster without a counterpart is maximally unstable, so include its
    # zero-similarity score in the public summary.  Keep ``matched`` sparse and
    # backward-compatible: it continues to contain only positive pairings.
    jvals = [item[2] for item in matched]
    jvals.extend(0.0 for _ in range(max(n_a, n_b) - len(matched)))
    mean_j = float(np.mean(jvals))
    min_j = float(min(jvals))

    all_union = set().union(*pool_a) | set().union(*pool_b)
    covered: set[int] = set()
    for i, j, jac in matched:
        if jac < coverage_floor_f:
            continue
        covered |= set(pool_a[i])
        covered |= set(pool_b[j])
    coverage = (len(covered) / len(all_union)) if all_union else 1.0

    return StabilityMetrics(
        mean_jaccard=mean_j,
        min_jaccard=min_j,
        coverage=float(coverage),
        count_diff=abs(n_a - n_b),
        nA=n_a,
        nB=n_b,
        matched=matched,
    )


def is_stable(metrics: StabilityMetrics, thresholds: StabilityThresholds) -> bool:
    thresholds.__post_init__()
    return (
        (metrics.mean_jaccard >= float(thresholds.mean_jaccard))
        and (metrics.min_jaccard >= float(thresholds.min_jaccard))
        and (metrics.coverage >= float(thresholds.coverage))
        and (metrics.count_diff <= int(thresholds.count_tol))
    )
