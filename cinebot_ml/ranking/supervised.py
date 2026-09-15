"""Adaptador B3 para ranking por probabilidade do modelo supervisionado."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.dataset import build_inference_frame
from cinebot_ml.schema import FEATURE_COLUMNS
from cinebot_ml.ranking.candidates import CandidateSet
from cinebot_ml.ranking.contracts import (
    ContractValidationError,
    RankingResult,
    RecommendationRequest,
    rank_scored_candidates,
)


DEFAULT_B3_CONFIG_PATH = PROJECT_ROOT / "configs" / "methods" / "b3_supervised_v1.yaml"
DEFAULT_B3_SCHEMA_PATH = PROJECT_ROOT / "configs" / "methods" / "b3_supervised_v1.schema.json"
PROFILE_GENRE_LIMIT = {"P0": 0, "P1": 1, "P2": 3, "P3": 3, "P4": 3, "P5": 3}
REQUIRED_ARTIFACT_METADATA = {
    "method_version",
    "winner_model",
    "winner_params",
    "winner_threshold",
    "feature_columns",
    "positive_class",
    "label_source",
    "fit_partition",
    "selection_partitions",
    "frozen_before_holdout",
}


class SupervisedConfigError(ValueError):
    """Configuração, modelo ou metadado incompatível com B3."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class SupervisedConfig:
    version: str
    positive_class: int
    label_source: str
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]


@dataclass(frozen=True)
class SupervisedArtifact:
    model: Any
    model_path: Path
    model_sha256: str
    metadata_path: Path
    metadata_sha256: str
    metadata: Mapping[str, Any]

    @classmethod
    def load(
        cls,
        model_path: Path,
        metadata_path: Path,
        config_path: Path = DEFAULT_B3_CONFIG_PATH,
    ) -> "SupervisedArtifact":
        config = load_supervised_config(config_path)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SupervisedConfigError(f"Não foi possível carregar metadados B3: {exc}") from exc
        if not isinstance(metadata, Mapping):
            raise SupervisedConfigError("Metadados B3 devem ser um objeto.")
        validate_supervised_metadata(metadata, config)
        try:
            model = joblib.load(model_path)
        except (OSError, ValueError, TypeError) as exc:
            raise SupervisedConfigError(f"Não foi possível carregar modelo B3: {exc}") from exc
        if not hasattr(model, "predict_proba"):
            raise SupervisedConfigError("Modelo B3 deve implementar predict_proba.")
        return cls(
            model=model,
            model_path=model_path.resolve(),
            model_sha256=_sha256(model_path),
            metadata_path=metadata_path.resolve(),
            metadata_sha256=_sha256(metadata_path),
            metadata=dict(metadata),
        )


def load_supervised_config(path: Path = DEFAULT_B3_CONFIG_PATH) -> SupervisedConfig:
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_B3_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise SupervisedConfigError(f"Não foi possível carregar a configuração B3: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SupervisedConfigError("A configuração B3 deve ser um objeto.")
    Draft202012Validator.check_schema(schema)
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        location = ".".join(map(str, error.absolute_path)) or "config"
        raise SupervisedConfigError(f"Configuração B3 inválida em {location}: {error.message}")
    lock_path = path.with_suffix(".lock.json")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupervisedConfigError(f"Lock B3 ausente ou inválido: {lock_path}") from exc
    digest = _sha256(path)
    if lock.get("sha256") != digest:
        raise SupervisedConfigError("A configuração B3 congelada foi alterada sem nova versão.")
    return SupervisedConfig(
        version=str(payload["version"]),
        positive_class=int(payload["positive_class"]),
        label_source=str(payload["label_source"]),
        source_path=path,
        source_sha256=digest,
        snapshot=dict(payload),
    )


def validate_supervised_metadata(
    metadata: Mapping[str, Any],
    config: SupervisedConfig,
) -> None:
    missing = sorted(REQUIRED_ARTIFACT_METADATA - set(metadata))
    if missing:
        raise SupervisedConfigError(f"Metadado obrigatório ausente no artefato B3: {missing[0]}")
    if metadata["method_version"] != config.version:
        raise SupervisedConfigError("Versão do modelo diverge da configuração B3.")
    if list(metadata["feature_columns"]) != FEATURE_COLUMNS:
        raise SupervisedConfigError("Schema de features do modelo B3 é incompatível.")
    if metadata["positive_class"] != config.positive_class:
        raise SupervisedConfigError("Classe positiva do modelo B3 é incompatível.")
    if metadata["label_source"] != config.label_source:
        raise SupervisedConfigError("Origem dos rótulos do modelo B3 não foi declarada corretamente.")
    if metadata["fit_partition"] != "train":
        raise SupervisedConfigError("Modelo B3 deve ser ajustado somente em treino.")
    if list(metadata["selection_partitions"]) != ["train", "validation"]:
        raise SupervisedConfigError("Seleção B3 deve usar somente treino e validação.")
    if metadata["frozen_before_holdout"] is not True:
        raise SupervisedConfigError("Modelo B3 precisa estar congelado antes do holdout.")
    threshold = metadata["winner_threshold"]
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
        raise SupervisedConfigError("Threshold diagnóstico B3 deve pertencer a [0,1].")


def assert_supervised_fit_partition(partition: str) -> None:
    if partition.strip().lower() != "train":
        raise SupervisedConfigError("Modelo B3 só pode ser ajustado com partition=train.")


def assert_supervised_selection_partition(partition: str) -> None:
    if partition.strip().lower() not in {"train", "validation"}:
        raise SupervisedConfigError("Seleção B3 só pode usar train ou validation; holdout é proibido.")


def _safe_number(value: Any, converter: type[int] | type[float]) -> int | float:
    try:
        number = converter(value)
    except (TypeError, ValueError, OverflowError):
        return converter(0)
    return number if math.isfinite(float(number)) else converter(0)


def _sanitized_catalog(candidates: CandidateSet) -> list[dict[str, Any]]:
    result = []
    for source in candidates.movies:
        movie = dict(source)
        movie["ano"] = _safe_number(movie.get("ano"), int)
        movie["nota"] = _safe_number(movie.get("nota"), float)
        movie["votos"] = max(0, int(_safe_number(movie.get("votos"), int)))
        movie["duracao"] = max(0, int(_safe_number(movie.get("duracao"), int)))
        for field in ("titulo", "sinopse", "diretor", "perfil", "genero"):
            movie[field] = str(movie.get(field) or "")
        for field in ("generos_secundarios", "palavras_chave", "streaming"):
            if not isinstance(movie.get(field), (list, tuple)):
                movie[field] = []
        result.append(movie)
    return result


def build_supervised_frame(
    candidates: CandidateSet,
    request: RecommendationRequest,
) -> pd.DataFrame:
    data = request.profile_data
    raw_genres = data.get("ranked_genres", [])
    if not isinstance(raw_genres, (list, tuple)):
        raw_genres = []
    genres = [str(value) for value in raw_genres[: PROFILE_GENRE_LIMIT[request.profile]]]
    decade = data.get("decade_preference") if request.profile in {"P3", "P4", "P5"} else None
    popularity = data.get("popularity_preference") if request.profile in {"P4", "P5"} else None
    frame = build_inference_frame(
        _sanitized_catalog(candidates),
        genres,
        decade,
        popularity,
        allow_missing_preferences=True,
    )
    missing = [column for column in FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise SupervisedConfigError(f"Feature obrigatória ausente na inferência B3: {missing[0]}")
    return frame


def _positive_class_index(model: Any, expected: int) -> int:
    classes = list(getattr(model, "classes_", []))
    if expected not in classes:
        raise SupervisedConfigError("Modelo B3 não contém a classe positiva esperada.")
    return classes.index(expected)


class SupervisedRecommender:
    """B3 ordena todos os candidatos pela probabilidade da classe positiva."""

    method = "B3"

    def __init__(
        self,
        candidates: CandidateSet,
        artifact: SupervisedArtifact,
        config_path: Path = DEFAULT_B3_CONFIG_PATH,
    ) -> None:
        self.candidates = candidates
        self.artifact = artifact
        self.config = load_supervised_config(config_path)
        validate_supervised_metadata(artifact.metadata, self.config)
        self.method_version = self.config.version
        self._positive_index = _positive_class_index(artifact.model, self.config.positive_class)

    def recommend(self, request: RecommendationRequest) -> RankingResult:
        started = time.perf_counter()
        if request.method != self.method:
            raise ContractValidationError("SupervisedRecommender aceita somente method=B3.")
        if request.method_version != self.method_version:
            raise ContractValidationError("A versão B3 da requisição diverge da configuração.")
        if request.candidate_movie_ids != self.candidates.movie_ids:
            raise ContractValidationError("A requisição diverge do CandidateSet de B3.")
        if self.candidates.movies:
            frame = build_supervised_frame(self.candidates, request)
            probabilities = np.asarray(
                self.artifact.model.predict_proba(frame[FEATURE_COLUMNS]),
                dtype=float,
            )
            if probabilities.shape != (len(frame), len(self.artifact.model.classes_)):
                raise SupervisedConfigError("predict_proba retornou shape incompatível no B3.")
            scores = probabilities[:, self._positive_index]
        else:
            scores = np.asarray([], dtype=float)
        if any(not math.isfinite(float(value)) or not 0 <= float(value) <= 1 for value in scores):
            raise SupervisedConfigError("Modelo B3 produziu score fora do intervalo [0,1].")
        threshold = float(self.artifact.metadata["winner_threshold"])
        scored = []
        for movie, score in zip(self.candidates.movies, scores):
            value = float(score)
            metadata: dict[str, Any] = {
                "score_components": {
                    "formula": "positive_class_probability",
                    "probability": value,
                    "diagnostic_threshold": threshold,
                    "label_source": self.config.label_source,
                }
            }
            title = str(movie.get("titulo") or "").strip()
            if title:
                metadata["title"] = title
            scored.append((movie["id"], value, metadata))
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
                "model_id": self.artifact.model_sha256[:20],
                "warnings": ["Scores aprendidos majoritariamente de rótulos proxy heurísticos."],
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
            "model_sha256": self.artifact.model_sha256,
            "metadata_sha256": self.artifact.metadata_sha256,
            "model_family": self.artifact.metadata["winner_model"],
            "model_params": dict(self.artifact.metadata["winner_params"]),
            "diagnostic_threshold": float(self.artifact.metadata["winner_threshold"]),
            "feature_columns": list(FEATURE_COLUMNS),
            "label_source": self.config.label_source,
            "fit_partition": "train",
            "selection_partitions": ["train", "validation"],
            "frozen_before_holdout": True,
            "candidate_set_id": self.candidates.candidate_set_id,
        }
