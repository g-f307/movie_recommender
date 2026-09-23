"""Estatística pareada definida pelo protocolo experimental v1."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon


def describe(values: Sequence[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    if data.size == 0 or not np.isfinite(data).all():
        raise ValueError("Resumo exige valores finitos.")
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    return {
        "n": int(data.size),
        "mean": float(data.mean()),
        "median": float(median),
        "standard_deviation": float(data.std(ddof=1)) if data.size > 1 else 0.0,
        "q1": float(q1),
        "q3": float(q3),
        "interquartile_range": float(q3 - q1),
    }


def bootstrap_mean_ci(
    differences: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 42,
    confidence: float = 0.95,
) -> tuple[float, float]:
    data = np.asarray(differences, dtype=float)
    if data.size == 0 or resamples < 1 or not np.isfinite(data).all():
        raise ValueError("Bootstrap exige diferenças finitas e ao menos uma reamostragem.")
    rng = np.random.default_rng(seed)
    means = np.empty(resamples, dtype=float)
    for start in range(0, resamples, 1_000):
        size = min(1_000, resamples - start)
        indexes = rng.integers(0, data.size, (size, data.size))
        means[start:start + size] = data[indexes].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    return float(low), float(high)


def paired_rank_biserial(differences: Sequence[float]) -> float:
    data = np.asarray(differences, dtype=float)
    nonzero = data[data != 0.0]
    if nonzero.size == 0:
        return 0.0
    ranks = rankdata(np.abs(nonzero), method="average")
    positive = float(ranks[nonzero > 0].sum())
    negative = float(ranks[nonzero < 0].sum())
    return (positive - negative) / (positive + negative)


def paired_inference(
    b4: Sequence[float],
    b5: Sequence[float],
    *,
    resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, float | int | list[float]]:
    if len(b4) != len(b5) or not b4:
        raise ValueError("Comparação exige vetores pareados não vazios.")
    left, right = np.asarray(b4, dtype=float), np.asarray(b5, dtype=float)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("Comparação pareada recebeu valor não finito.")
    differences = right - left
    if np.all(differences == 0.0):
        statistic, p_value = 0.0, 1.0
    else:
        test = wilcoxon(
            right, left, alternative="two-sided", zero_method="wilcox", method="auto"
        )
        statistic, p_value = float(test.statistic), float(test.pvalue)
    low, high = bootstrap_mean_ci(differences, resamples=resamples, seed=seed)
    b4_mean = float(left.mean())
    mean_difference = float(differences.mean())
    return {
        "valid_pairs": int(differences.size),
        "positive": int((differences > 0).sum()),
        "negative": int((differences < 0).sum()),
        "ties": int((differences == 0).sum()),
        "mean_difference": mean_difference,
        "relative_mean_difference": (
            mean_difference / b4_mean if not math.isclose(b4_mean, 0.0) else math.nan
        ),
        "confidence_interval_95": [low, high],
        "wilcoxon_statistic": statistic,
        "p_value_two_sided": p_value,
        "rank_biserial_correlation": paired_rank_biserial(differences),
    }
