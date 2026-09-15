"""Baseline B1 baseado em similaridade de conteúdo estruturado."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.dataset import (
    build_catalog_context,
    canonicalize_decade_preference,
    canonicalize_genre,
    canonicalize_popularity_preference,
    movie_genres,
    movie_popularity_bucket,
    movie_release_period,
    slugify,
)
from cinebot_ml.ranking.candidates import CandidateSet
from cinebot_ml.ranking.contracts import (
    ContractValidationError,
    RankingResult,
    RecommendationRequest,
    rank_scored_candidates,
)


DEFAULT_B1_CONFIG_PATH = PROJECT_ROOT / "configs" / "methods" / "b1_content_v1.yaml"
DEFAULT_B1_SCHEMA_PATH = PROJECT_ROOT / "configs" / "methods" / "b1_content_v1.schema.json"
PROFILE_GENRE_LIMIT = {"P0": 0, "P1": 1, "P2": 3, "P3": 3, "P4": 3, "P5": 3}


class ContentConfigError(ValueError):
    """Configuração ou entrada incompatível com B1."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ContentConfig:
    version: str
    feature_weights: Mapping[str, float]
    genre_rank_weights: tuple[float, float, float]
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]


@dataclass(frozen=True)
class ContentProfile:
    genres: tuple[str, ...] = ()
    decade: str | None = None
    popularity: str | None = None
    directors: frozenset[str] = frozenset()
    keywords: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ItemContent:
    genres: frozenset[str]
    decade: str | None
    popularity: str | None
    directors: frozenset[str]
    keywords: frozenset[str]


def load_content_config(path: Path = DEFAULT_B1_CONFIG_PATH) -> ContentConfig:
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_B1_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ContentConfigError(f"Não foi possível carregar a configuração B1: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ContentConfigError("A configuração B1 deve ser um objeto.")
    Draft202012Validator.check_schema(schema)
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        location = ".".join(map(str, error.absolute_path)) or "config"
        raise ContentConfigError(f"Configuração B1 inválida em {location}: {error.message}")
    lock_path = path.with_suffix(".lock.json")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContentConfigError(f"Lock B1 ausente ou inválido: {lock_path}") from exc
    digest = _sha256(path)
    if lock.get("sha256") != digest:
        raise ContentConfigError("A configuração B1 congelada foi alterada sem nova versão.")
    features = payload["features"]
    rank_weights = tuple(float(value) for value in features["genres"]["preference_rank_weights"])
    return ContentConfig(
        version=str(payload["version"]),
        feature_weights={name: float(value["weight"]) for name, value in features.items()},
        genre_rank_weights=rank_weights,  # type: ignore[arg-type]
        source_path=path,
        source_sha256=digest,
        snapshot=dict(payload),
    )


def _normalized_values(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = value
    else:
        return frozenset()
    return frozenset(normalized for raw in values if (normalized := slugify(str(raw))))


def _popularity_metadata_is_valid(movie: Mapping[str, Any]) -> bool:
    try:
        rating = float(movie.get("nota"))
        votes = int(movie.get("votos"))
    except (TypeError, ValueError):
        return False
    return 0.0 <= rating <= 10.0 and votes >= 0


def build_content_profile(request: RecommendationRequest) -> ContentProfile:
    """Extrai apenas os campos autorizados pelo nível de cold start."""

    data = request.profile_data
    genre_limit = PROFILE_GENRE_LIMIT[request.profile]
    raw_genres = data.get("ranked_genres", [])
    if not isinstance(raw_genres, (list, tuple)):
        raw_genres = []
    genres = tuple(
        genre
        for raw in raw_genres[:genre_limit]
        if (genre := canonicalize_genre(str(raw)))
    )
    decade = None
    if request.profile in {"P3", "P4", "P5"}:
        decade = canonicalize_decade_preference(data.get("decade_preference"))
    popularity = None
    if request.profile in {"P4", "P5"}:
        popularity = canonicalize_popularity_preference(data.get("popularity_preference"))

    # P5 pode transportar preferências estáticas enriquecidas. O histórico nunca
    # participa de B1 e, portanto, não é usado para reconstruir estes campos.
    directors = _normalized_values(data.get("preferred_directors")) if request.profile == "P5" else frozenset()
    keywords = _normalized_values(data.get("preferred_keywords")) if request.profile == "P5" else frozenset()
    return ContentProfile(genres, decade, popularity, directors, keywords)


def build_item_content(movie: Mapping[str, Any], catalog_context: Mapping[str, Any]) -> ItemContent:
    director = _normalized_values(movie.get("diretor"))
    keywords = _normalized_values(movie.get("palavras_chave"))
    return ItemContent(
        genres=frozenset(movie_genres(dict(movie))),
        decade=movie_release_period(dict(movie)) if movie.get("ano") not in (None, "") else None,
        popularity=(
            movie_popularity_bucket(dict(movie), dict(catalog_context))
            if _popularity_metadata_is_valid(movie)
            else None
        ),
        directors=director,
        keywords=keywords,
    )


def _set_similarity(preference: frozenset[str], item: frozenset[str]) -> float:
    if not preference:
        return 0.0
    return len(preference & item) / len(preference | item) if item else 0.0


def content_similarity(
    profile: ContentProfile,
    item: ItemContent,
    config: ContentConfig,
) -> tuple[float, dict[str, Any]]:
    """Calcula média ponderada somente sobre blocos declarados pelo perfil."""

    components: dict[str, float] = {}
    active_weights: dict[str, float] = {}
    if profile.genres:
        maximum = sum(config.genre_rank_weights[: len(profile.genres)])
        matched = sum(
            weight
            for genre, weight in zip(profile.genres, config.genre_rank_weights)
            if genre in item.genres
        )
        components["genres"] = matched / maximum
        active_weights["genres"] = config.feature_weights["genres"]
    if profile.decade:
        components["decade"] = float(item.decade == profile.decade)
        active_weights["decade"] = config.feature_weights["decade"]
    if profile.popularity:
        components["popularity"] = float(item.popularity == profile.popularity)
        active_weights["popularity"] = config.feature_weights["popularity"]
    if profile.directors:
        components["directors"] = _set_similarity(profile.directors, item.directors)
        active_weights["directors"] = config.feature_weights["directors"]
    if profile.keywords:
        components["keywords"] = _set_similarity(profile.keywords, item.keywords)
        active_weights["keywords"] = config.feature_weights["keywords"]
    denominator = sum(active_weights.values())
    score = 0.0 if denominator == 0 else sum(
        components[name] * weight for name, weight in active_weights.items()
    ) / denominator
    return score, {
        "formula": "weighted_content_similarity",
        "components": components,
        "active_weights": active_weights,
    }


class ContentRecommender:
    """B1 estático, sem uso de histórico ou feedback incremental."""

    method = "B1"

    def __init__(
        self,
        candidates: CandidateSet,
        config_path: Path = DEFAULT_B1_CONFIG_PATH,
    ) -> None:
        self.candidates = candidates
        self.config = load_content_config(config_path)
        self.method_version = self.config.version
        context_catalog = []
        for movie in candidates.movies:
            normalized = dict(movie)
            if not _popularity_metadata_is_valid(movie):
                normalized["votos"] = 0
            context_catalog.append(normalized)
        self._catalog_context = build_catalog_context(context_catalog)
        self._items = tuple(
            build_item_content(movie, self._catalog_context) for movie in candidates.movies
        )

    def recommend(self, request: RecommendationRequest) -> RankingResult:
        started = time.perf_counter()
        if request.method != self.method:
            raise ContractValidationError("ContentRecommender aceita somente method=B1.")
        if request.method_version != self.method_version:
            raise ContractValidationError("A versão B1 da requisição diverge da configuração.")
        if request.candidate_movie_ids != self.candidates.movie_ids:
            raise ContractValidationError("A requisição diverge do CandidateSet de B1.")
        profile = build_content_profile(request)
        scored = []
        for movie, item in zip(self.candidates.movies, self._items):
            score, components = content_similarity(profile, item, self.config)
            metadata: dict[str, Any] = {"score_components": components}
            title = str(movie.get("titulo") or "").strip()
            if title:
                metadata["title"] = title
            scored.append((movie["id"], score, metadata))
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
            ranked_items=rank_scored_candidates(scored, request.k),
            metadata={
                "candidate_set_id": self.candidates.candidate_set_id,
                "model_id": f"b1-content-v{self.method_version}",
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
            "transform_fit_partition": "none",
            "holdout_used_for_selection": False,
            "candidate_set_id": self.candidates.candidate_set_id,
        }
