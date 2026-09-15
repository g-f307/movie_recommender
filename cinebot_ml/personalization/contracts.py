"""Estado imutável e versionado usado pelos experimentos incrementais."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any, Mapping

from cinebot_ml.ranking.contracts import MovieId


SCHEMA_VERSION = "1.0"
FEEDBACK_VALUES = {"like", "dislike"}
IDENTITY_KINDS = {"synthetic", "pseudonymous"}
FEEDBACK_SOURCES = {"real_feedback", "synthetic_user", "human_judgment"}
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


class PersonalizationContractError(ValueError):
    """Um estado ou evento viola o contrato de personalização."""


def _parse_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise PersonalizationContractError(f"{field_name} deve ser um timestamp ISO 8601.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PersonalizationContractError(f"{field_name} possui timestamp inválido: {value}") from exc
    if parsed.tzinfo is None:
        raise PersonalizationContractError(f"{field_name} deve incluir fuso horário.")
    return parsed.astimezone(timezone.utc)


def _validate_text(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise PersonalizationContractError(f"{field_name} deve ser um texto não vazio.")


def _validate_movie_id(movie_id: Any) -> None:
    if isinstance(movie_id, bool) or not isinstance(movie_id, (int, str)):
        raise PersonalizationContractError("movie_id deve ser inteiro ou texto.")
    if isinstance(movie_id, int) and movie_id < 0:
        raise PersonalizationContractError("movie_id inteiro não pode ser negativo.")
    if isinstance(movie_id, str) and not movie_id.strip():
        raise PersonalizationContractError("movie_id textual não pode ser vazio.")


def _find_forbidden_key(value: Any, path: str) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            current = f"{path}.{key}"
            if any(marker in str(key).lower() for marker in FORBIDDEN_KEY_MARKERS):
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


def _freeze_json(value: Any, field_name: str) -> Any:
    """Valida JSON finito e o converte recursivamente para estruturas imutáveis."""
    if isinstance(value, Mapping):
        frozen = {str(key): _freeze_json(child, f"{field_name}.{key}") for key, child in value.items()}
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(child, f"{field_name}[]") for child in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise PersonalizationContractError(f"{field_name} deve conter somente valores JSON finitos.")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(child) for child in value]
    return value


def _validate_private_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PersonalizationContractError(f"{field_name} deve ser um objeto.")
    forbidden = _find_forbidden_key(value, field_name)
    if forbidden:
        raise PersonalizationContractError(f"Dado pessoal ou secreto proibido: {forbidden}")
    return _freeze_json(value, field_name)


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _event_order(event: "FeedbackEvent") -> tuple[datetime, int, str]:
    return (_parse_timestamp(event.timestamp, "event.timestamp"), event.sequence, event.event_id)


@dataclass(frozen=True)
class FeedbackEvent:
    """Feedback explícito observado em um instante da sequência experimental."""

    event_id: str
    movie_id: MovieId
    feedback: str
    timestamp: str
    sequence: int
    source: str
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise PersonalizationContractError(f"schema_version de evento não suportada: {self.schema_version}")
        _validate_text(self.event_id, "event_id")
        _validate_movie_id(self.movie_id)
        if self.feedback not in FEEDBACK_VALUES:
            raise PersonalizationContractError(f"feedback inválido: {self.feedback}")
        _parse_timestamp(self.timestamp, "timestamp")
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int) or self.sequence <= 0:
            raise PersonalizationContractError("sequence deve ser um inteiro maior que zero.")
        if self.source not in FEEDBACK_SOURCES:
            raise PersonalizationContractError(f"source de feedback inválida: {self.source}")
        object.__setattr__(self, "metadata", _validate_private_mapping(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "movie_id": self.movie_id,
            "feedback": self.feedback,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            "source": self.source,
            "metadata": _thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FeedbackEvent":
        if not isinstance(value, Mapping):
            raise PersonalizationContractError("FeedbackEvent deve ser reconstruído de um objeto.")
        return cls(**dict(value))


@dataclass(frozen=True)
class UserState:
    """Estado completo disponível em um timestamp lógico, profundamente imutável."""

    subject_id: str
    identity_kind: str
    logical_timestamp: str
    version: int = 0
    initial_profile: Mapping[str, Any] = field(default_factory=dict)
    learned_preferences: Mapping[str, Any] = field(default_factory=dict)
    presented_movie_ids: tuple[MovieId, ...] = field(default_factory=tuple)
    consumed_movie_ids: tuple[MovieId, ...] = field(default_factory=tuple)
    history: tuple[FeedbackEvent, ...] = field(default_factory=tuple)
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise PersonalizationContractError(f"schema_version de estado não suportada: {self.schema_version}")
        _validate_text(self.subject_id, "subject_id")
        if self.identity_kind not in IDENTITY_KINDS:
            raise PersonalizationContractError(f"identity_kind inválido: {self.identity_kind}")
        state_time = _parse_timestamp(self.logical_timestamp, "logical_timestamp")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version < 0:
            raise PersonalizationContractError("version deve ser um inteiro não negativo.")
        if not isinstance(self.history, tuple) or any(not isinstance(event, FeedbackEvent) for event in self.history):
            raise PersonalizationContractError("history deve ser uma tupla de FeedbackEvent.")
        if self.version != len(self.history):
            raise PersonalizationContractError("version deve corresponder à quantidade de eventos do histórico.")
        if len({event.event_id for event in self.history}) != len(self.history):
            raise PersonalizationContractError("history não pode conter event_id duplicado.")
        if len({event.sequence for event in self.history}) != len(self.history):
            raise PersonalizationContractError("history não pode conter sequence duplicada.")
        if tuple(sorted(self.history, key=_event_order)) != self.history:
            raise PersonalizationContractError("history deve estar em ordem cronológica determinística.")
        if tuple(event.sequence for event in self.history) != tuple(range(1, len(self.history) + 1)):
            raise PersonalizationContractError("sequence do histórico deve começar em 1 e ser contínua.")
        if any(_parse_timestamp(event.timestamp, "event.timestamp") > state_time for event in self.history):
            raise PersonalizationContractError("Evento futuro não pode fazer parte do estado.")
        object.__setattr__(self, "initial_profile", _validate_private_mapping(self.initial_profile, "initial_profile"))
        object.__setattr__(
            self,
            "learned_preferences",
            _validate_private_mapping(self.learned_preferences, "learned_preferences"),
        )
        for field_name in ("presented_movie_ids", "consumed_movie_ids"):
            values = getattr(self, field_name)
            if not isinstance(values, tuple):
                raise PersonalizationContractError(f"{field_name} deve ser uma tupla imutável.")
            for movie_id in values:
                _validate_movie_id(movie_id)
            if len(values) != len(set(values)):
                raise PersonalizationContractError(f"{field_name} não pode conter duplicidades.")
        evaluated = {event.movie_id for event in self.history}
        if not evaluated.issubset(set(self.presented_movie_ids)):
            raise PersonalizationContractError("Todo item avaliado deve constar entre os itens apresentados.")
        if not set(self.consumed_movie_ids).issubset(set(self.presented_movie_ids)):
            raise PersonalizationContractError("Todo item consumido deve constar entre os itens apresentados.")

    @property
    def evaluated_movie_ids(self) -> tuple[MovieId, ...]:
        return tuple(dict.fromkeys(event.movie_id for event in self.history))

    @property
    def state_id(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "subject_id": self.subject_id,
            "identity_kind": self.identity_kind,
            "logical_timestamp": self.logical_timestamp,
            "version": self.version,
            "initial_profile": _thaw_json(self.initial_profile),
            "learned_preferences": _thaw_json(self.learned_preferences),
            "presented_movie_ids": list(self.presented_movie_ids),
            "consumed_movie_ids": list(self.consumed_movie_ids),
            "history": [event.to_dict() for event in self.history],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UserState":
        if not isinstance(value, Mapping):
            raise PersonalizationContractError("UserState deve ser reconstruído de um objeto.")
        payload = dict(value)
        payload["presented_movie_ids"] = tuple(payload.get("presented_movie_ids", ()))
        payload["consumed_movie_ids"] = tuple(payload.get("consumed_movie_ids", ()))
        payload["history"] = tuple(FeedbackEvent.from_dict(event) for event in payload.get("history", ()))
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> "UserState":
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PersonalizationContractError("JSON de UserState inválido.") from exc
        return cls.from_dict(payload)


@dataclass(frozen=True)
class StateSnapshot:
    """Envelope encadeado que identifica um estado antes ou depois de uma interação."""

    state: UserState
    previous_state_id: str | None = None
    transition_event_id: str | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise PersonalizationContractError(f"schema_version de snapshot não suportada: {self.schema_version}")
        if not isinstance(self.state, UserState):
            raise PersonalizationContractError("state deve ser um UserState.")
        if self.state.version == 0:
            if self.previous_state_id is not None or self.transition_event_id is not None:
                raise PersonalizationContractError("Snapshot inicial não pode possuir transição anterior.")
        else:
            _validate_text(self.previous_state_id, "previous_state_id")
            _validate_text(self.transition_event_id, "transition_event_id")
            if self.transition_event_id != self.state.history[-1].event_id:
                raise PersonalizationContractError("transition_event_id deve identificar o último evento do estado.")

    @property
    def snapshot_id(self) -> str:
        return _canonical_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state": self.state.to_dict(),
            "previous_state_id": self.previous_state_id,
            "transition_event_id": self.transition_event_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateSnapshot":
        if not isinstance(value, Mapping):
            raise PersonalizationContractError("StateSnapshot deve ser reconstruído de um objeto.")
        payload = dict(value)
        payload["state"] = UserState.from_dict(payload["state"])
        return cls(**payload)

    @classmethod
    def from_json(cls, value: str) -> "StateSnapshot":
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise PersonalizationContractError("JSON de StateSnapshot inválido.") from exc
        return cls.from_dict(payload)
