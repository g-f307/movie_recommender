"""Baseline B4 personalizado pelo perfil inicial e estático no tempo."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.dataset import build_catalog_context
from cinebot_ml.personalization import UserState
from cinebot_ml.ranking.candidates import CandidateSet
from cinebot_ml.ranking.content import (
    ContentConfig,
    build_content_profile,
    build_item_content,
    content_similarity,
)
from cinebot_ml.ranking.contracts import (
    ContractValidationError,
    RankingResult,
    RecommendationRequest,
    rank_scored_candidates,
)


DEFAULT_B4_CONFIG_PATH = PROJECT_ROOT / "configs" / "methods" / "b4_static_v1.yaml"
DEFAULT_B4_SCHEMA_PATH = PROJECT_ROOT / "configs" / "methods" / "b4_static_v1.schema.json"


class StaticPersonalizedConfigError(ValueError):
    """Configuração ou estado incompatível com B4."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class StaticPersonalizedConfig:
    version: str
    feature_weights: Mapping[str, float]
    genre_rank_weights: tuple[float, float, float]
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]

    def as_content_config(self) -> ContentConfig:
        """Reutiliza apenas a função matemática estruturada, não o estado de B1."""
        return ContentConfig(
            version=self.version,
            feature_weights=self.feature_weights,
            genre_rank_weights=self.genre_rank_weights,
            source_path=self.source_path,
            source_sha256=self.source_sha256,
            snapshot=self.snapshot,
        )


def load_static_personalized_config(
    path: Path = DEFAULT_B4_CONFIG_PATH,
) -> StaticPersonalizedConfig:
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_B4_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise StaticPersonalizedConfigError(f"Não foi possível carregar a configuração B4: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise StaticPersonalizedConfigError("A configuração B4 deve ser um objeto.")
    Draft202012Validator.check_schema(schema)
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        location = ".".join(map(str, error.absolute_path)) or "config"
        raise StaticPersonalizedConfigError(f"Configuração B4 inválida em {location}: {error.message}")
    lock_path = path.with_suffix(".lock.json")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StaticPersonalizedConfigError(f"Lock B4 ausente ou inválido: {lock_path}") from exc
    digest = _sha256(path)
    if lock.get("sha256") != digest:
        raise StaticPersonalizedConfigError("A configuração B4 congelada foi alterada sem nova versão.")
    features = payload["features"]
    return StaticPersonalizedConfig(
        version=str(payload["version"]),
        feature_weights={name: float(value["weight"]) for name, value in features.items()},
        genre_rank_weights=tuple(float(value) for value in features["genres"]["preference_rank_weights"]),  # type: ignore[arg-type]
        source_path=path,
        source_sha256=digest,
        snapshot=dict(payload),
    )


class StaticPersonalizedRecommender:
    """B4 vinculado ao perfil inicial imutável de uma unidade experimental."""

    method = "B4"

    def __init__(
        self,
        candidates: CandidateSet,
        initial_state: UserState,
        config_path: Path = DEFAULT_B4_CONFIG_PATH,
    ) -> None:
        self.candidates = candidates
        self.initial_state = initial_state
        self.config = load_static_personalized_config(config_path)
        self.method_version = self.config.version
        context_catalog = []
        for movie in candidates.movies:
            normalized = dict(movie)
            try:
                float(movie.get("nota"))
                int(movie.get("votos"))
            except (TypeError, ValueError):
                normalized["votos"] = 0
            context_catalog.append(normalized)
        catalog_context = build_catalog_context(context_catalog)
        self._items = tuple(build_item_content(movie, catalog_context) for movie in candidates.movies)

    def _static_request(self, request: RecommendationRequest) -> RecommendationRequest:
        if request.unit_id != self.initial_state.subject_id:
            raise ContractValidationError("unit_id da requisição diverge do estado inicial de B4.")
        return replace(
            request,
            profile_data=self.initial_state.to_dict()["initial_profile"],
            history=(),
        )

    def recommend(self, request: RecommendationRequest) -> RankingResult:
        started = time.perf_counter()
        if request.method != self.method:
            raise ContractValidationError("StaticPersonalizedRecommender aceita somente method=B4.")
        if request.method_version != self.method_version:
            raise ContractValidationError("A versão B4 da requisição diverge da configuração.")
        if request.candidate_movie_ids != self.candidates.movie_ids:
            raise ContractValidationError("A requisição diverge do CandidateSet de B4.")
        static_request = self._static_request(request)
        profile = build_content_profile(static_request)
        content_config = self.config.as_content_config()
        scored = []
        for movie, item in zip(self.candidates.movies, self._items):
            score, components = content_similarity(profile, item, content_config)
            components["formula"] = "weighted_static_profile_similarity"
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
                "model_id": f"b4-static-v{self.method_version}",
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
            "state_source": "immutable_initial_profile",
            "initial_state_id": self.initial_state.state_id,
            "initial_state_version": 0,
            "observed_state_version": self.initial_state.version,
            "transform_fit_partition": "none",
            "holdout_used_for_selection": False,
            "candidate_set_id": self.candidates.candidate_set_id,
        }
