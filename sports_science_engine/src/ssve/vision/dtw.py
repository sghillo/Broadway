"""Dynamic Time Warping for multivariate movement sequences."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class DTWResult:
    distance: float           # cumulative cost along the optimal path
    normalised: float         # distance / path length (comparable across clip lengths)
    path: np.ndarray          # (L, 2) aligned index pairs (i in x, j in y)
    cost_matrix: np.ndarray   # local cost matrix (N, M)


def dtw(x: np.ndarray, y: np.ndarray, metric: str = "euclidean",
        window: int | None = None) -> DTWResult:
    """DTW between x (N, D) and y (M, D) with optional Sakoe-Chiba band of half-width `window`."""
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    y = np.atleast_2d(y.T).T if y.ndim == 1 else y
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        raise ValueError("DTW requires non-empty sequences")
    cost = cdist(x, y, metric=metric)
    w = max(window, abs(n - m)) if window is not None else max(n, m)

    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - w), min(m, i + w)
        for j in range(lo, hi + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])

    # Backtrack.
    i, j = n, m
    path = [(i - 1, j - 1)]
    while (i, j) != (1, 1):
        steps = ((i - 1, j - 1), (i - 1, j), (i, j - 1))
        i, j = min((s for s in steps if s[0] >= 1 and s[1] >= 1), key=lambda s: acc[s])
        path.append((i - 1, j - 1))
    path_arr = np.asarray(path[::-1])
    total = float(acc[n, m])
    return DTWResult(total, total / len(path_arr), path_arr, cost)


def warp(y: np.ndarray, path: np.ndarray, length: int) -> np.ndarray:
    """Resample y onto x's timeline (length N) by averaging y-frames matched to each x-frame."""
    out = np.zeros((length,) + y.shape[1:])
    counts = np.zeros(length)
    for i, j in path:
        out[i] += y[j]
        counts[i] += 1
    return out / counts.reshape((-1,) + (1,) * (y.ndim - 1))
