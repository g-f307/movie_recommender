"""Métricas puras de descoberta, popularidade e estabilidade de rankings."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from itertools import combinations
from statistics import mean
from typing import Any, Mapping, Sequence

from cinebot_ml.ranking.contracts import MovieId
from cinebot_ml.ranking.metrics import RankingMetricError


ALLOWED_POPULARITY_PARTITIONS = {"train", "catalog_metadata"}


def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class PopularityReference:
    counts: Mapping[MovieId, int]
    partition: str
    version: str

    def __post_init__(self) -> None:
        if self.partition not in ALLOWED_POPULARITY_PARTITIONS:
            raise RankingMetricError("Popularidade exige partição train ou metadado congelado do catálogo.")
        if not self.version.strip() or not self.counts:
            raise RankingMetricError("Distribuição de popularidade deve possuir versão e itens.")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.counts.values()):
            raise RankingMetricError("Contagens de popularidade devem ser inteiros não negativos.")
        if sum(self.counts.values()) <= 0:
            raise RankingMetricError("Distribuição de popularidade não pode ter soma zero.")

    @property
    def distribution_id(self) -> str:
        return _hash({
            "partition": self.partition,
            "version": self.version,
            "counts": sorted((str(key), value) for key, value in self.counts.items()),
        })


def _validate_ranking(ranking: Sequence[MovieId], k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise RankingMetricError("K deve ser um inteiro maior que zero.")
    if len(ranking) != len(set(ranking)):
        raise RankingMetricError("Ranking contém movie_id duplicado.")


def _tokens(movie: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    for feature, keys in {
        "genre": ("generos", "genres", "genero", "genre", "generos_secundarios"),
        "director": ("diretores", "directors", "diretor", "director"),
        "keyword": ("palavras_chave", "keywords"),
    }.items():
        for key in keys:
            raw = movie.get(key)
            if isinstance(raw, str):
                raw = raw.replace("|", ",").split(",")
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                values.update(
                    f"{feature}:{str(item).strip().lower()}" for item in raw if str(item).strip()
                )
    year = movie.get("ano", movie.get("year"))
    try:
        values.add(f"decade:{int(year) // 10 * 10}")
    except (TypeError, ValueError):
        pass
    return values


def intra_list_diversity_at_k(
    ranking: Sequence[MovieId], catalog: Mapping[MovieId, Mapping[str, Any]], k: int
) -> float | None:
    _validate_ranking(ranking, k)
    top = list(ranking[:k])
    if not top:
        return None
    if len(top) == 1:
        return 0.0 if _tokens(catalog.get(top[0], {})) else None
    distances = []
    for left, right in combinations(top, 2):
        left_tokens, right_tokens = _tokens(catalog.get(left, {})), _tokens(catalog.get(right, {}))
        if not left_tokens or not right_tokens:
            continue
        union = left_tokens | right_tokens
        if union:
            distances.append(1.0 - len(left_tokens & right_tokens) / len(union))
    return mean(distances) if distances else None


def _popularity_probabilities(reference: PopularityReference) -> dict[MovieId, float]:
    total = sum(reference.counts.values()) + len(reference.counts)
    return {movie_id: (count + 1) / total for movie_id, count in reference.counts.items()}


def novelty_at_k(
    ranking: Sequence[MovieId], reference: PopularityReference | None, k: int
) -> float | None:
    _validate_ranking(ranking, k)
    top = list(ranking[:k])
    if not top or reference is None or any(item not in reference.counts for item in top):
        return None
    probabilities = _popularity_probabilities(reference)
    max_information = -math.log2(min(probabilities.values()))
    if max_information == 0:
        return 0.0
    return mean(-math.log2(probabilities[item]) / max_information for item in top)


def popularity_exposure_at_k(
    ranking: Sequence[MovieId], reference: PopularityReference | None, k: int
) -> float | None:
    _validate_ranking(ranking, k)
    top = list(ranking[:k])
    if not top or reference is None or any(item not in reference.counts for item in top):
        return None
    ordered = sorted(set(reference.counts.values()))
    if len(ordered) == 1:
        percentiles = {ordered[0]: 0.5}
    else:
        percentiles = {value: index / (len(ordered) - 1) for index, value in enumerate(ordered)}
    return mean(percentiles[reference.counts[item]] for item in top)


def popularity_bias_at_k(
    ranking: Sequence[MovieId], reference: PopularityReference | None, k: int
) -> float | None:
    exposure = popularity_exposure_at_k(ranking, reference, k)
    if exposure is None or reference is None:
        return None
    baseline = popularity_exposure_at_k(tuple(reference.counts), reference, len(reference.counts))
    return exposure - float(baseline)


def ranking_overlap_at_k(previous: Sequence[MovieId], current: Sequence[MovieId], k: int) -> float | None:
    _validate_ranking(previous, k)
    _validate_ranking(current, k)
    left, right = set(previous[:k]), set(current[:k])
    if not left and not right:
        return None
    return len(left & right) / len(left | right)


def ranking_repetition_at_k(previous: Sequence[MovieId], current: Sequence[MovieId], k: int) -> float | None:
    _validate_ranking(previous, k)
    _validate_ranking(current, k)
    left, right = list(previous[:k]), list(current[:k])
    if not left and not right:
        return None
    denominator = max(len(left), len(right))
    return sum(a == b for a, b in zip(left, right)) / denominator


def rank_position_variation_at_k(
    previous: Sequence[MovieId], current: Sequence[MovieId], k: int
) -> float | None:
    _validate_ranking(previous, k)
    _validate_ranking(current, k)
    left, right = list(previous[:k]), list(current[:k])
    shared = set(left) & set(right)
    if not shared:
        return None
    if k == 1:
        return 0.0
    left_positions = {item: index for index, item in enumerate(left)}
    right_positions = {item: index for index, item in enumerate(right)}
    return mean(abs(left_positions[item] - right_positions[item]) / (k - 1) for item in shared)


def calculate_discovery_metrics(
    ranking: Sequence[MovieId],
    catalog: Mapping[MovieId, Mapping[str, Any]],
    k: int,
    popularity: PopularityReference | None = None,
) -> dict[str, float | str | None]:
    return {
        "diversity_at_k": intra_list_diversity_at_k(ranking, catalog, k),
        "novelty_at_k": novelty_at_k(ranking, popularity, k),
        "popularity_exposure_at_k": popularity_exposure_at_k(ranking, popularity, k),
        "popularity_bias_at_k": popularity_bias_at_k(ranking, popularity, k),
        "popularity_distribution_id": popularity.distribution_id if popularity else None,
    }
