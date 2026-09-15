"""Baseline B0 de popularidade com média bayesiana versionada."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.ranking.candidates import CandidateSet
from cinebot_ml.ranking.contracts import (
    ContractValidationError,
    RankingResult,
    RecommendationRequest,
    rank_scored_candidates,
)


DEFAULT_B0_CONFIG_PATH = PROJECT_ROOT / "configs" / "methods" / "b0_popularity_v1.yaml"
DEFAULT_B0_SCHEMA_PATH = PROJECT_ROOT / "configs" / "methods" / "b0_popularity_v1.schema.json"


class PopularityConfigError(ValueError):
    """Configuração ou entrada incompatível com B0."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class PopularityConfig:
    version: str
    rating_minimum: float
    rating_maximum: float
    vote_minimum: int
    prior_strength_quantile: float
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]


def load_popularity_config(path: Path = DEFAULT_B0_CONFIG_PATH) -> PopularityConfig:
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_B0_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise PopularityConfigError(f"Não foi possível carregar a configuração B0: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PopularityConfigError("A configuração B0 deve ser um objeto.")
    Draft202012Validator.check_schema(schema)
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        location = ".".join(map(str, error.absolute_path)) or "config"
        raise PopularityConfigError(f"Configuração B0 inválida em {location}: {error.message}")
    lock_path = path.with_suffix(".lock.json")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PopularityConfigError(f"Lock B0 ausente ou inválido: {lock_path}") from exc
    digest = _sha256(path)
    if lock.get("sha256") != digest:
        raise PopularityConfigError("A configuração B0 congelada foi alterada sem nova versão.")
    rating = payload["rating"]
    votes = payload["votes"]
    if float(rating["maximum"]) <= float(rating["minimum"]):
        raise PopularityConfigError("rating.maximum deve ser maior que rating.minimum.")
    return PopularityConfig(
        version=str(payload["version"]),
        rating_minimum=float(rating["minimum"]),
        rating_maximum=float(rating["maximum"]),
        vote_minimum=int(votes["minimum"]),
        prior_strength_quantile=float(votes["prior_strength_quantile"]),
        source_path=path,
        source_sha256=digest,
        snapshot=dict(payload),
    )


def _rating(movie: Mapping[str, Any], config: PopularityConfig) -> float | None:
    try:
        value = float(movie.get("nota"))
    except (TypeError, ValueError):
        return None
    if value < config.rating_minimum or value > config.rating_maximum:
        return None
    return (value - config.rating_minimum) / (config.rating_maximum - config.rating_minimum)


def _votes(movie: Mapping[str, Any], config: PopularityConfig) -> int:
    try:
        value = int(movie.get("votos"))
    except (TypeError, ValueError):
        return config.vote_minimum
    return max(value, config.vote_minimum)


def _quantile(values: Sequence[int], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def popularity_parameters(
    candidates: CandidateSet,
    config: PopularityConfig,
) -> tuple[float, float]:
    ratings = [rating for movie in candidates.movies if (rating := _rating(movie, config)) is not None]
    catalog_mean = sum(ratings) / len(ratings) if ratings else 0.5
    prior_strength = _quantile(
        [_votes(movie, config) for movie in candidates.movies],
        config.prior_strength_quantile,
    )
    return catalog_mean, prior_strength


def bayesian_popularity_score(
    movie: Mapping[str, Any],
    *,
    catalog_mean: float,
    prior_strength: float,
    config: PopularityConfig,
) -> tuple[float, dict[str, float | int | str]]:
    rating = _rating(movie, config)
    votes = _votes(movie, config)
    effective_rating = catalog_mean if rating is None else rating
    denominator = votes + prior_strength
    score = catalog_mean if denominator == 0 else (
        (votes / denominator) * effective_rating
        + (prior_strength / denominator) * catalog_mean
    )
    return score, {
        "formula": "bayesian_weighted_rating",
        "rating_normalized": effective_rating,
        "rating_source": "catalog_mean" if rating is None else "item",
        "votes": votes,
        "catalog_mean": catalog_mean,
        "prior_strength": prior_strength,
    }


class PopularityRecommender:
    """B0 não personalizado, inicializado com um CandidateSet comum."""

    method = "B0"

    def __init__(
        self,
        candidates: CandidateSet,
        config_path: Path = DEFAULT_B0_CONFIG_PATH,
    ) -> None:
        self.candidates = candidates
        self.config = load_popularity_config(config_path)
        self.method_version = self.config.version
        self._catalog_mean, self._prior_strength = popularity_parameters(candidates, self.config)

    def recommend(self, request: RecommendationRequest) -> RankingResult:
        started = time.perf_counter()
        if request.method != self.method:
            raise ContractValidationError("PopularityRecommender aceita somente method=B0.")
        if request.method_version != self.method_version:
            raise ContractValidationError("A versão B0 da requisição diverge da configuração.")
        if request.candidate_movie_ids != self.candidates.movie_ids:
            raise ContractValidationError("A requisição diverge do CandidateSet de B0.")
        scored = []
        for movie in self.candidates.movies:
            score, components = bayesian_popularity_score(
                movie,
                catalog_mean=self._catalog_mean,
                prior_strength=self._prior_strength,
                config=self.config,
            )
            metadata: dict[str, Any] = {"score_components": components}
            title = str(movie.get("titulo") or "").strip()
            if title:
                metadata["title"] = title
            scored.append((movie["id"], score, metadata))
        ranked_items = rank_scored_candidates(scored, request.k)
        result = RankingResult(
            experiment_id=request.experiment_id,
            method=self.method,
            method_version=self.method_version,
            condition=request.condition,
            profile=request.profile,
            seed=request.seed,
            logical_timestamp=request.logical_timestamp,
            status="completed",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            ranked_items=ranked_items,
            metadata={
                "candidate_set_id": self.candidates.candidate_set_id,
                "model_id": f"b0-popularity-v{self.method_version}",
            },
        )
        result.validate_against(request)
        return result

    def method_manifest(self) -> dict[str, Any]:
        try:
            config_path = self.config.source_path.relative_to(PROJECT_ROOT)
        except ValueError:
            config_path = Path(self.config.source_path.name)
        return {
            "method": self.method,
            "method_version": self.method_version,
            "config_path": config_path.as_posix(),
            "config_sha256": self.config.source_sha256,
            "config_snapshot": dict(self.config.snapshot),
            "derived_parameters": {
                "catalog_mean": self._catalog_mean,
                "prior_strength": self._prior_strength,
            },
            "candidate_set_id": self.candidates.candidate_set_id,
        }
