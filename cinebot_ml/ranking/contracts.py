"""Contrato comum, portátil e validável para recomendadores B0--B6."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Protocol, Sequence, runtime_checkable


MovieId = int | str
METHODS = {f"B{index}" for index in range(7)}
CONDITIONS = {f"C{index}" for index in range(6)}
PROFILES = {f"P{index}" for index in range(6)}
RESULT_STATUSES = {"completed", "failed"}
ALLOWED_ITEM_METADATA = {"title", "reason", "score_components", "explanation_tags"}
ALLOWED_RESULT_METADATA = {"candidate_set_id", "model_id", "warnings"}
FORBIDDEN_KEY_MARKERS = (
    "api_key",
    "apikey",
    "credential",
    "email",
    "name",
    "password",
    "phone",
    "secret",
    "token",
)


class ContractValidationError(ValueError):
    """Um pedido ou resultado viola o contrato experimental de ranking."""


def _movie_id_key(movie_id: MovieId) -> tuple[int, Any]:
    _validate_movie_id(movie_id)
    if isinstance(movie_id, int):
        return (0, movie_id)
    return (1, movie_id)


def _validate_movie_id(movie_id: Any) -> None:
    if isinstance(movie_id, bool) or not isinstance(movie_id, (int, str)):
        raise ContractValidationError("movie_id deve ser inteiro ou texto.")
    if isinstance(movie_id, int) and movie_id < 0:
        raise ContractValidationError("movie_id inteiro não pode ser negativo.")
    if isinstance(movie_id, str) and not movie_id.strip():
        raise ContractValidationError("movie_id textual não pode ser vazio.")


def _parse_timestamp(value: str, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{field_name} deve ser um timestamp ISO 8601.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractValidationError(f"{field_name} possui timestamp inválido: {value}") from exc
    if parsed.tzinfo is None:
        raise ContractValidationError(f"{field_name} deve incluir fuso horário.")
    return parsed.astimezone(timezone.utc)


def _assert_json_value(value: Any, field_name: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ContractValidationError(f"{field_name} deve conter somente valores JSON válidos.") from exc


def _find_forbidden_key(value: Any, path: str) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower()
            current = f"{path}.{key}"
            if any(marker in normalized for marker in FORBIDDEN_KEY_MARKERS):
                return current
            found = _find_forbidden_key(child, current)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _find_forbidden_key(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _validate_metadata(metadata: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    if not isinstance(metadata, Mapping):
        raise ContractValidationError(f"{field_name} deve ser um objeto.")
    unknown = sorted(set(metadata) - allowed)
    if unknown:
        raise ContractValidationError(f"Metadado não permitido em {field_name}: {unknown[0]}")
    forbidden = _find_forbidden_key(metadata, field_name)
    if forbidden:
        raise ContractValidationError(f"Metadado pessoal ou secreto proibido: {forbidden}")
    _assert_json_value(metadata, field_name)


@dataclass(frozen=True)
class RecommendationRequest:
    """Contexto completo disponível igualmente aos métodos comparados."""

    experiment_id: str
    protocol_version: str
    method: str
    method_version: str
    condition: str
    profile: str
    seed: int
    logical_timestamp: str
    candidate_movie_ids: tuple[MovieId, ...]
    k: int
    unit_id: str | None = None
    session_id: str | None = None
    profile_data: Mapping[str, Any] = field(default_factory=dict)
    history: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        for name in ("experiment_id", "protocol_version", "method_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ContractValidationError(f"{name} deve ser um texto não vazio.")
        if self.method not in METHODS:
            raise ContractValidationError(f"Método desconhecido: {self.method}")
        if self.condition not in CONDITIONS:
            raise ContractValidationError(f"Condição desconhecida: {self.condition}")
        if self.profile not in PROFILES:
            raise ContractValidationError(f"Perfil desconhecido: {self.profile}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractValidationError("seed deve ser um inteiro.")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k <= 0:
            raise ContractValidationError("k deve ser um inteiro maior que zero.")
        logical_time = _parse_timestamp(self.logical_timestamp, "logical_timestamp")
        if not isinstance(self.candidate_movie_ids, tuple):
            raise ContractValidationError("candidate_movie_ids deve ser uma tupla imutável.")
        for movie_id in self.candidate_movie_ids:
            _validate_movie_id(movie_id)
        if len(self.candidate_movie_ids) != len(set(self.candidate_movie_ids)):
            raise ContractValidationError("candidate_movie_ids não pode conter duplicidades.")
        for identifier in ("unit_id", "session_id"):
            value = getattr(self, identifier)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ContractValidationError(f"{identifier} deve ser nulo ou texto não vazio.")
        if not isinstance(self.profile_data, Mapping):
            raise ContractValidationError("profile_data deve ser um objeto.")
        _assert_json_value(self.profile_data, "profile_data")
        forbidden = _find_forbidden_key(self.profile_data, "profile_data")
        if forbidden:
            raise ContractValidationError(f"Dado pessoal ou secreto proibido: {forbidden}")
        if not isinstance(self.history, tuple):
            raise ContractValidationError("history deve ser uma tupla imutável.")
        for index, event in enumerate(self.history):
            if not isinstance(event, Mapping):
                raise ContractValidationError(f"history[{index}] deve ser um objeto.")
            if "timestamp" not in event:
                raise ContractValidationError(f"history[{index}].timestamp é obrigatório.")
            event_time = _parse_timestamp(event["timestamp"], f"history[{index}].timestamp")
            if event_time >= logical_time:
                raise ContractValidationError(
                    f"history[{index}] não é anterior ao timestamp lógico da recomendação."
                )
            forbidden = _find_forbidden_key(event, f"history[{index}]")
            if forbidden:
                raise ContractValidationError(f"Dado pessoal ou secreto proibido: {forbidden}")
            _assert_json_value(event, f"history[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "protocol_version": self.protocol_version,
            "method": self.method,
            "method_version": self.method_version,
            "condition": self.condition,
            "profile": self.profile,
            "seed": self.seed,
            "logical_timestamp": self.logical_timestamp,
            "candidate_movie_ids": list(self.candidate_movie_ids),
            "k": self.k,
            "unit_id": self.unit_id,
            "session_id": self.session_id,
            "profile_data": dict(self.profile_data),
            "history": [dict(event) for event in self.history],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecommendationRequest":
        payload = dict(value)
        payload["candidate_movie_ids"] = tuple(payload.get("candidate_movie_ids", ()))
        payload["history"] = tuple(payload.get("history", ()))
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> "RecommendationRequest":
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractValidationError("JSON de RecommendationRequest inválido.") from exc
        if not isinstance(payload, Mapping):
            raise ContractValidationError("JSON de RecommendationRequest deve representar um objeto.")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class RecommendationItem:
    """Item ordenado; score bruto e posição são campos independentes."""

    movie_id: MovieId
    rank: int
    score: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_movie_id(self.movie_id)
        if isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank <= 0:
            raise ContractValidationError("rank deve ser um inteiro maior que zero.")
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)):
            raise ContractValidationError("score deve ser numérico.")
        if not math.isfinite(float(self.score)):
            raise ContractValidationError("score deve ser finito.")
        _validate_metadata(self.metadata, ALLOWED_ITEM_METADATA, "item.metadata")

    def to_dict(self) -> dict[str, Any]:
        return {
            "movie_id": self.movie_id,
            "rank": self.rank,
            "score": float(self.score),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecommendationItem":
        return cls(**dict(value))


@dataclass(frozen=True)
class RankingResult:
    """Resultado validado de uma chamada ao recomendador."""

    experiment_id: str
    method: str
    method_version: str
    condition: str
    profile: str
    seed: int
    logical_timestamp: str
    status: str
    latency_ms: float
    ranked_items: tuple[RecommendationItem, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        for name in ("experiment_id", "method_version"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ContractValidationError(f"{name} deve ser um texto não vazio.")
        if self.method not in METHODS:
            raise ContractValidationError(f"Método desconhecido: {self.method}")
        if self.condition not in CONDITIONS or self.profile not in PROFILES:
            raise ContractValidationError("Condição ou perfil desconhecido no resultado.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ContractValidationError("seed deve ser um inteiro.")
        if self.status not in RESULT_STATUSES:
            raise ContractValidationError(f"Status inválido: {self.status}")
        if isinstance(self.latency_ms, bool) or not isinstance(self.latency_ms, (int, float)):
            raise ContractValidationError("latency_ms deve ser numérico.")
        if not math.isfinite(float(self.latency_ms)) or self.latency_ms < 0:
            raise ContractValidationError("latency_ms deve ser finito e não negativo.")
        if not isinstance(self.ranked_items, tuple):
            raise ContractValidationError("ranked_items deve ser uma tupla imutável.")
        if any(not isinstance(item, RecommendationItem) for item in self.ranked_items):
            raise ContractValidationError("ranked_items deve conter somente RecommendationItem.")
        if self.status == "failed" and self.ranked_items:
            raise ContractValidationError("Resultado com falha não pode conter ranking.")
        if self.status == "failed" and not self.error_code:
            raise ContractValidationError("Resultado com falha deve informar error_code.")
        if self.error_code is not None and (
            not isinstance(self.error_code, str) or not self.error_code.strip()
        ):
            raise ContractValidationError("error_code deve ser nulo ou texto não vazio.")
        if self.error_message is not None and not isinstance(self.error_message, str):
            raise ContractValidationError("error_message deve ser nulo ou texto.")
        if self.status == "completed" and (self.error_code or self.error_message):
            raise ContractValidationError("Resultado concluído não pode conter erro.")
        _parse_timestamp(self.logical_timestamp, "logical_timestamp")
        _validate_metadata(self.metadata, ALLOWED_RESULT_METADATA, "result.metadata")

    def validate_against(self, request: RecommendationRequest) -> None:
        identity = (
            "experiment_id", "method", "method_version", "condition", "profile", "seed",
            "logical_timestamp",
        )
        for field_name in identity:
            if getattr(self, field_name) != getattr(request, field_name):
                raise ContractValidationError(f"Resultado diverge da requisição em {field_name}.")
        if len(self.ranked_items) > request.k:
            raise ContractValidationError("Ranking contém mais itens que K.")
        if self.status == "completed" and request.candidate_movie_ids and not self.ranked_items:
            raise ContractValidationError("Ranking vazio com candidatos deve usar status de falha.")
        movie_ids = [item.movie_id for item in self.ranked_items]
        if len(movie_ids) != len(set(movie_ids)):
            raise ContractValidationError("Ranking contém movie_id duplicado.")
        candidates = set(request.candidate_movie_ids)
        outside = [movie_id for movie_id in movie_ids if movie_id not in candidates]
        if outside:
            raise ContractValidationError(f"Item fora do conjunto candidato: {outside[0]}")
        expected_ranks = list(range(1, len(self.ranked_items) + 1))
        if [item.rank for item in self.ranked_items] != expected_ranks:
            raise ContractValidationError("Ranks devem começar em 1 e ser contínuos.")
        expected_order = sorted(
            self.ranked_items,
            key=lambda item: (-float(item.score), _movie_id_key(item.movie_id)),
        )
        if list(self.ranked_items) != expected_order:
            raise ContractValidationError("Ranking viola a ordem por score e movie_id.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "method": self.method,
            "method_version": self.method_version,
            "condition": self.condition,
            "profile": self.profile,
            "seed": self.seed,
            "logical_timestamp": self.logical_timestamp,
            "status": self.status,
            "latency_ms": float(self.latency_ms),
            "ranked_items": [item.to_dict() for item in self.ranked_items],
            "metadata": dict(self.metadata),
            "error_code": self.error_code,
            "error_message": self.error_message,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RankingResult":
        payload = dict(value)
        payload["ranked_items"] = tuple(
            RecommendationItem.from_dict(item) for item in payload.get("ranked_items", ())
        )
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> "RankingResult":
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ContractValidationError("JSON de RankingResult inválido.") from exc
        if not isinstance(payload, Mapping):
            raise ContractValidationError("JSON de RankingResult deve representar um objeto.")
        return cls.from_dict(payload)


def rank_scored_candidates(
    scored_candidates: Iterable[tuple[MovieId, float, Mapping[str, Any]]],
    k: int,
) -> tuple[RecommendationItem, ...]:
    """Ordena score descendente e usa movie_id ascendente como desempate."""
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ContractValidationError("k deve ser um inteiro maior que zero.")
    collected: list[tuple[MovieId, float, Mapping[str, Any]]] = []
    seen: set[MovieId] = set()
    for movie_id, score, metadata in scored_candidates:
        _validate_movie_id(movie_id)
        if movie_id in seen:
            raise ContractValidationError(f"Score duplicado para movie_id: {movie_id}")
        seen.add(movie_id)
        item = RecommendationItem(movie_id=movie_id, rank=1, score=score, metadata=metadata)
        collected.append((item.movie_id, float(item.score), item.metadata))
    ordered = sorted(collected, key=lambda item: (-item[1], _movie_id_key(item[0])))[:k]
    return tuple(
        RecommendationItem(movie_id=movie_id, rank=index, score=score, metadata=metadata)
        for index, (movie_id, score, metadata) in enumerate(ordered, start=1)
    )


@runtime_checkable
class Recommender(Protocol):
    """Interface estrutural que deverá ser implementada por B0--B6."""

    method: str
    method_version: str

    def recommend(self, request: RecommendationRequest) -> RankingResult:
        """Produz e valida um ranking para a requisição fornecida."""
        ...
