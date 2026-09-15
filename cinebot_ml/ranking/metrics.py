"""Métricas Top-K puras, validadas e independentes dos recomendadores."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from cinebot_ml.ranking.contracts import MovieId


class RankingMetricError(ValueError):
    """Ranking ou julgamentos incompatíveis com uma métrica Top-K."""


def _validate(ranking: Sequence[MovieId], relevance: Mapping[MovieId, float], k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise RankingMetricError("K deve ser um inteiro maior que zero.")
    if len(ranking) != len(set(ranking)):
        raise RankingMetricError("Ranking contém movie_id duplicado.")
    for movie_id, value in relevance.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RankingMetricError(f"Relevância de {movie_id} deve ser numérica.")
        if not math.isfinite(float(value)) or value < 0:
            raise RankingMetricError(f"Relevância de {movie_id} deve ser finita e não negativa.")


def _top(ranking: Sequence[MovieId], k: int) -> list[MovieId]:
    return list(ranking[:k])


def precision_at_k(ranking: Sequence[MovieId], relevance: Mapping[MovieId, float], k: int) -> float:
    _validate(ranking, relevance, k)
    top = _top(ranking, k)
    return 0.0 if not top else sum(relevance.get(item, 0.0) > 0 for item in top) / len(top)


def recall_at_k(
    ranking: Sequence[MovieId], relevance: Mapping[MovieId, float], k: int
) -> float | None:
    _validate(ranking, relevance, k)
    total = sum(value > 0 for value in relevance.values())
    if total == 0:
        return None
    return sum(relevance.get(item, 0.0) > 0 for item in _top(ranking, k)) / total


def f1_at_k(ranking: Sequence[MovieId], relevance: Mapping[MovieId, float], k: int) -> float | None:
    precision = precision_at_k(ranking, relevance, k)
    recall = recall_at_k(ranking, relevance, k)
    if recall is None:
        return None
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def dcg_at_k(ranking: Sequence[MovieId], relevance: Mapping[MovieId, float], k: int) -> float:
    _validate(ranking, relevance, k)
    return sum(
        (2.0 ** float(relevance.get(movie_id, 0.0)) - 1.0) / math.log2(rank + 1)
        for rank, movie_id in enumerate(_top(ranking, k), start=1)
    )


def ndcg_at_k(
    ranking: Sequence[MovieId], relevance: Mapping[MovieId, float], k: int
) -> float | None:
    actual = dcg_at_k(ranking, relevance, k)
    ideal_values = sorted((float(value) for value in relevance.values()), reverse=True)[:k]
    ideal = sum((2.0**value - 1.0) / math.log2(rank + 1) for rank, value in enumerate(ideal_values, 1))
    return None if ideal == 0 else actual / ideal


def average_precision_at_k(
    ranking: Sequence[MovieId], relevance: Mapping[MovieId, float], k: int
) -> float | None:
    _validate(ranking, relevance, k)
    total = sum(value > 0 for value in relevance.values())
    if total == 0:
        return None
    hits = 0
    accumulated = 0.0
    for rank, movie_id in enumerate(_top(ranking, k), start=1):
        if relevance.get(movie_id, 0.0) > 0:
            hits += 1
            accumulated += hits / rank
    return accumulated / min(total, k)


def reciprocal_rank_at_k(
    ranking: Sequence[MovieId], relevance: Mapping[MovieId, float], k: int
) -> float:
    _validate(ranking, relevance, k)
    for rank, movie_id in enumerate(_top(ranking, k), start=1):
        if relevance.get(movie_id, 0.0) > 0:
            return 1.0 / rank
    return 0.0


def hit_rate_at_k(ranking: Sequence[MovieId], relevance: Mapping[MovieId, float], k: int) -> float:
    return float(reciprocal_rank_at_k(ranking, relevance, k) > 0)


def catalog_coverage_at_k(
    rankings: Sequence[Sequence[MovieId]], candidate_movie_ids: Sequence[MovieId], k: int
) -> float | None:
    if len(candidate_movie_ids) != len(set(candidate_movie_ids)):
        raise RankingMetricError("Catálogo candidato contém movie_id duplicado.")
    if not candidate_movie_ids:
        return None
    recommended: set[MovieId] = set()
    candidates = set(candidate_movie_ids)
    for ranking in rankings:
        _validate(ranking, {}, k)
        outside = set(ranking) - candidates
        if outside:
            raise RankingMetricError(f"Ranking contém item fora do catálogo: {next(iter(outside))}")
        recommended.update(ranking[:k])
    return len(recommended) / len(candidates)


def calculate_top_k_metrics(
    ranking: Sequence[MovieId], relevance: Mapping[MovieId, float], k: int
) -> dict[str, float | None]:
    """Retorna o contrato comum, mantendo métricas indefinidas como nulas."""

    return {
        "precision_at_k": precision_at_k(ranking, relevance, k),
        "recall_at_k": recall_at_k(ranking, relevance, k),
        "f1_at_k": f1_at_k(ranking, relevance, k),
        "ndcg_at_k": ndcg_at_k(ranking, relevance, k),
        "map_at_k": average_precision_at_k(ranking, relevance, k),
        "mrr_at_k": reciprocal_rank_at_k(ranking, relevance, k),
        "hit_rate_at_k": hit_rate_at_k(ranking, relevance, k),
        "diversity_at_k": None,
        "novelty_at_k": None,
    }
