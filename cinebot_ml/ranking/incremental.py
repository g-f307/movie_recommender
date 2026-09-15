"""Baseline B5: perfil inicial acrescido de feedback incremental observado."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.dataset import build_catalog_context, movie_genres, slugify
from cinebot_ml.personalization import StateSnapshot
from cinebot_ml.ranking.candidates import CandidateSet
from cinebot_ml.ranking.content import build_content_profile, build_item_content, content_similarity
from cinebot_ml.ranking.contracts import (
    ContractValidationError,
    RankingResult,
    RecommendationRequest,
    rank_scored_candidates,
)
from cinebot_ml.ranking.static_personalized import load_static_personalized_config

DEFAULT_B5_CONFIG_PATH = PROJECT_ROOT / "configs/methods/b5_incremental_v1.yaml"
DEFAULT_B5_SCHEMA_PATH = PROJECT_ROOT / "configs/methods/b5_incremental_v1.schema.json"

class IncrementalConfigError(ValueError):
    """Configuração ou estado incompatível com B5."""

@dataclass(frozen=True)
class IncrementalConfig:
    version: str
    adjustment_weight: float
    feature_weights: Mapping[str, float]
    initial_profile_config: Path
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]

def load_incremental_config(path: Path = DEFAULT_B5_CONFIG_PATH) -> IncrementalConfig:
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_B5_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise IncrementalConfigError(str(exc)) from exc
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        raise IncrementalConfigError(f"Configuração B5 inválida: {error.message}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise IncrementalConfigError("Lock B5 ausente ou inválido.") from exc
    if lock.get("sha256") != digest:
        raise IncrementalConfigError("A configuração B5 congelada foi alterada sem nova versão.")
    numbers = [payload["feedback_adjustment_weight"], *payload["feature_weights"].values()]
    if any(not math.isfinite(float(value)) for value in numbers):
        raise IncrementalConfigError("Pesos B5 devem ser finitos.")
    initial_profile_config = (PROJECT_ROOT / payload["initial_profile_config"]).resolve()
    return IncrementalConfig(
        version=str(payload["version"]),
        adjustment_weight=float(payload["feedback_adjustment_weight"]),
        feature_weights={key: float(value) for key, value in payload["feature_weights"].items()},
        initial_profile_config=initial_profile_config,
        source_path=path,
        source_sha256=digest,
        snapshot=dict(payload),
    )

def _utc(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractValidationError("Timestamp de estado inválido.") from exc
    if parsed.tzinfo is None:
        raise ContractValidationError("Timestamp de estado deve incluir fuso.")
    return parsed.astimezone(timezone.utc)

def _values(movie: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    def normalized(value: Any) -> tuple[str, ...]:
        raw = [value] if isinstance(value, str) else (value or [])
        return tuple(sorted({slugify(str(v)) for v in raw if slugify(str(v))}))
    synopsis = slugify(str(movie.get("sinopse") or ""))
    return {
        "genres": tuple(sorted(set(movie_genres(dict(movie))))),
        "directors": normalized(movie.get("diretor")),
        "keywords": normalized(movie.get("palavras_chave")),
        "synopsis_terms": tuple(sorted(set(re.findall(r"[a-z0-9]+", synopsis)))),
    }

class IncrementalRecommender:
    method = "B5"
    def __init__(self, candidates: CandidateSet, state: StateSnapshot, config_path: Path = DEFAULT_B5_CONFIG_PATH):
        self.candidates, self.state, self.config = candidates, state, load_incremental_config(config_path)
        self.method_version = self.config.version
        b4 = load_static_personalized_config(self.config.initial_profile_config)
        self._content_config = b4.as_content_config()
        catalog = []
        for movie in candidates.movies:
            item = dict(movie)
            try: int(item.get("votos", 0) or 0)
            except (TypeError, ValueError): item["votos"] = 0
            catalog.append(item)
        context = build_catalog_context(catalog)
        self._items = tuple(build_item_content(movie, context) for movie in candidates.movies)

    def recommend(self, request: RecommendationRequest) -> RankingResult:
        started = time.perf_counter(); state = self.state.state
        if request.method != "B5" or request.method_version != self.method_version: raise ContractValidationError("IncrementalRecommender exige method=B5 e versão compatível.")
        if request.candidate_movie_ids != self.candidates.movie_ids: raise ContractValidationError("A requisição diverge do CandidateSet de B5.")
        if request.unit_id != state.subject_id: raise ContractValidationError("unit_id diverge do estado B5.")
        if _utc(state.logical_timestamp) > _utc(request.logical_timestamp): raise ContractValidationError("Estado futuro não pode influenciar o ranking.")
        consumed = set(state.consumed_movie_ids)
        if consumed.intersection(request.candidate_movie_ids): raise ContractValidationError("CandidateSet B5 contém item consumido.")
        static_request = replace(request, profile_data=state.to_dict()["initial_profile"], history=())
        profile = build_content_profile(static_request)
        learned_preferences = state.to_dict()["learned_preferences"]
        learned = learned_preferences.get("affinities", {})
        if not isinstance(learned, Mapping):
            raise ContractValidationError("Afinidades aprendidas de B5 devem ser um objeto.")
        scored = []
        for movie, item in zip(self.candidates.movies, self._items):
            initial, details = content_similarity(profile, item, self._content_config)
            weighted, total = 0.0, 0.0
            for feature, values in _values(movie).items():
                affinities = learned.get(feature, {})
                if not isinstance(affinities, Mapping):
                    raise ContractValidationError(f"Afinidades de {feature} devem ser um objeto.")
                try:
                    matches = [float(affinities[value]) for value in values if value in affinities]
                except (TypeError, ValueError) as exc:
                    raise ContractValidationError(f"Afinidade não numérica em {feature}.") from exc
                if any(not math.isfinite(value) or not -1.0 <= value <= 1.0 for value in matches):
                    raise ContractValidationError(f"Afinidade inválida em {feature}.")
                if matches:
                    weight = self.config.feature_weights[feature]; weighted += weight * sum(matches) / len(matches); total += weight
            adjustment = weighted / total if total else 0.0
            score = min(1.0, max(0.0, initial + self.config.adjustment_weight * adjustment))
            if not math.isfinite(score): raise ContractValidationError("Score B5 deve ser finito.")
            scored.append((movie["id"], score, {"score_components": {"initial_profile": initial, "feedback_adjustment": adjustment, "state_version": state.version, "initial_details": details}}))
        result = RankingResult(request.experiment_id, "B5", self.method_version, request.condition, request.profile, request.seed, request.logical_timestamp, "completed", (time.perf_counter()-started)*1000, rank_scored_candidates(scored, request.k), {"candidate_set_id": self.candidates.candidate_set_id, "model_id": f"b5-incremental-v{self.method_version}"})
        result.validate_against(request); return result

    def method_manifest(self) -> dict[str, Any]:
        return {"method":"B5","method_version":self.method_version,"config_path":self.config.source_path.relative_to(PROJECT_ROOT).as_posix(),"config_sha256":self.config.source_sha256,"config_snapshot":dict(self.config.snapshot),"state_id":self.state.state.state_id,"state_version":self.state.state.version,"candidate_set_id":self.candidates.candidate_set_id,"holdout_used_for_selection":False}
