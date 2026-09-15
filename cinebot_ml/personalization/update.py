"""Política determinística e auditável de atualização incremental do perfil."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.dataset import movie_genres, slugify
from cinebot_ml.personalization.contracts import FeedbackEvent, StateSnapshot, UserState


DEFAULT_PROFILE_UPDATE_CONFIG_PATH = PROJECT_ROOT / "configs" / "profile_update_v1.yaml"
DEFAULT_PROFILE_UPDATE_SCHEMA_PATH = PROJECT_ROOT / "configs" / "profile_update_v1.schema.json"
AFFINITY_FEATURES = ("genres", "directors", "keywords", "synopsis_terms")


class ProfileUpdateError(ValueError):
    """Configuração, estado, evento ou item incompatível com a atualização."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_timestamp(value: str, field_name: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ProfileUpdateError(f"{field_name} deve ser um timestamp ISO 8601.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProfileUpdateError(f"{field_name} possui timestamp inválido: {value}") from exc
    if parsed.tzinfo is None:
        raise ProfileUpdateError(f"{field_name} deve incluir fuso horário.")
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class ProfileUpdateConfig:
    version: str
    learning_rate: float
    decay: float
    minimum_affinity: float
    maximum_affinity: float
    like_direction: float
    dislike_direction: float
    marks_consumed: bool
    feature_weights: Mapping[str, float]
    enabled_features: frozenset[str]
    maximum_terms_per_item: int
    minimum_token_length: int
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]


def load_profile_update_config(
    path: Path = DEFAULT_PROFILE_UPDATE_CONFIG_PATH,
) -> ProfileUpdateConfig:
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_PROFILE_UPDATE_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ProfileUpdateError(f"Não foi possível carregar a política de atualização: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ProfileUpdateError("A política de atualização deve ser um objeto.")
    Draft202012Validator.check_schema(schema)
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        location = ".".join(map(str, error.absolute_path)) or "config"
        raise ProfileUpdateError(f"Política de atualização inválida em {location}: {error.message}")
    numeric_parameters = [
        payload["feedback"]["like_direction"],
        payload["feedback"]["dislike_direction"],
        payload["update"]["learning_rate"],
        payload["update"]["decay"],
        payload["update"]["minimum_affinity"],
        payload["update"]["maximum_affinity"],
        *(value["weight"] for value in payload["features"].values()),
    ]
    if any(not math.isfinite(float(value)) for value in numeric_parameters):
        raise ProfileUpdateError("Parâmetros numéricos da política devem ser finitos.")
    minimum = float(payload["update"]["minimum_affinity"])
    maximum = float(payload["update"]["maximum_affinity"])
    if minimum >= maximum:
        raise ProfileUpdateError("minimum_affinity deve ser menor que maximum_affinity.")
    lock_path = path.with_suffix(".lock.json")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileUpdateError(f"Lock da política ausente ou inválido: {lock_path}") from exc
    digest = _sha256(path)
    if lock.get("sha256") != digest:
        raise ProfileUpdateError("A política congelada foi alterada sem nova versão.")
    features = payload["features"]
    return ProfileUpdateConfig(
        version=str(payload["version"]),
        learning_rate=float(payload["update"]["learning_rate"]),
        decay=float(payload["update"]["decay"]),
        minimum_affinity=minimum,
        maximum_affinity=maximum,
        like_direction=float(payload["feedback"]["like_direction"]),
        dislike_direction=float(payload["feedback"]["dislike_direction"]),
        marks_consumed=bool(payload["feedback"]["marks_consumed"]),
        feature_weights={name: float(value["weight"]) for name, value in features.items()},
        enabled_features=frozenset(name for name, value in features.items() if value["enabled"]),
        maximum_terms_per_item=int(features["synopsis_terms"]["maximum_terms_per_item"]),
        minimum_token_length=int(payload["text"]["minimum_token_length"]),
        source_path=path,
        source_sha256=digest,
        snapshot=dict(payload),
    )


def _normalized_values(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[Any] = (value,)
    elif isinstance(value, (list, tuple, set, frozenset)):
        values = tuple(value)
    else:
        return ()
    return tuple(sorted({normalized for raw in values if (normalized := slugify(str(raw))) }))


def _item_features(movie: Mapping[str, Any], config: ProfileUpdateConfig) -> dict[str, tuple[str, ...]]:
    features: dict[str, tuple[str, ...]] = {name: () for name in AFFINITY_FEATURES}
    if "genres" in config.enabled_features:
        features["genres"] = tuple(sorted(set(movie_genres(dict(movie)))))
    if "directors" in config.enabled_features:
        features["directors"] = _normalized_values(movie.get("diretor"))
    if "keywords" in config.enabled_features:
        features["keywords"] = _normalized_values(movie.get("palavras_chave"))
    if "synopsis_terms" in config.enabled_features:
        normalized = slugify(str(movie.get("sinopse") or ""))
        terms = {
            token
            for token in re.findall(r"[a-z0-9]+", normalized)
            if len(token) >= config.minimum_token_length
        }
        features["synopsis_terms"] = tuple(sorted(terms)[: config.maximum_terms_per_item])
    return features


def _existing_preferences(state: UserState, config: ProfileUpdateConfig) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    payload = state.to_dict()["learned_preferences"]
    unknown = sorted(set(payload) - {"affinities", "event_contributions"})
    if unknown:
        raise ProfileUpdateError(f"Campo desconhecido em learned_preferences: {unknown[0]}")
    raw_affinities = payload.get("affinities", {})
    raw_contributions = payload.get("event_contributions", {})
    if not isinstance(raw_affinities, Mapping) or not isinstance(raw_contributions, Mapping):
        raise ProfileUpdateError("Afinidades e contribuições devem ser objetos.")
    affinities: dict[str, dict[str, float]] = {}
    for feature in AFFINITY_FEATURES:
        raw_values = raw_affinities.get(feature, {})
        if not isinstance(raw_values, Mapping):
            raise ProfileUpdateError(f"Afinidades de {feature} devem ser um objeto.")
        values: dict[str, float] = {}
        for key, raw_score in raw_values.items():
            if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
                raise ProfileUpdateError(f"Afinidade de {feature}.{key} deve ser numérica.")
            score = float(raw_score)
            if not math.isfinite(score) or not config.minimum_affinity <= score <= config.maximum_affinity:
                raise ProfileUpdateError(f"Afinidade de {feature}.{key} está fora dos limites.")
            values[str(key)] = score
        affinities[feature] = values
    return affinities, dict(raw_contributions)


def _append_unique(values: tuple[Any, ...], item: Any) -> tuple[Any, ...]:
    return values if item in values else values + (item,)


class ProfileUpdater:
    """Aplica feedback a um snapshot sem acoplar a atualização ao recomendador."""

    def __init__(self, config_path: Path = DEFAULT_PROFILE_UPDATE_CONFIG_PATH) -> None:
        self.config = load_profile_update_config(config_path)

    def update(
        self,
        snapshot: StateSnapshot,
        event: FeedbackEvent | None,
        movie: Mapping[str, Any] | None,
        *,
        logical_timestamp: str,
    ) -> StateSnapshot:
        if event is None:
            if movie is not None:
                raise ProfileUpdateError("Item não pode ser informado sem feedback.")
            return snapshot
        if movie is None:
            raise ProfileUpdateError("O item avaliado é obrigatório.")
        state = snapshot.state
        cutoff = _parse_timestamp(logical_timestamp, "logical_timestamp")
        event_time = _parse_timestamp(event.timestamp, "event.timestamp")
        state_time = _parse_timestamp(state.logical_timestamp, "state.logical_timestamp")
        if event_time > cutoff:
            raise ProfileUpdateError("Evento futuro não pode atualizar o estado.")
        if event_time < state_time:
            raise ProfileUpdateError("Evento anterior ao estado atual viola a ordem temporal.")
        if event.event_id in {item.event_id for item in state.history}:
            raise ProfileUpdateError(f"Evento duplicado: {event.event_id}")
        if event.sequence != state.version + 1:
            raise ProfileUpdateError("A sequência do evento deve continuar a versão do estado.")
        movie_id = movie.get("id")
        if movie_id != event.movie_id:
            raise ProfileUpdateError("movie_id do evento diverge do item avaliado.")

        affinities, contributions = _existing_preferences(state, self.config)
        for feature_values in affinities.values():
            for key, score in tuple(feature_values.items()):
                feature_values[key] = round(score * self.config.decay, 12)

        direction = self.config.like_direction if event.feedback == "like" else self.config.dislike_direction
        feature_contributions: dict[str, dict[str, float]] = {}
        for feature, values in _item_features(movie, self.config).items():
            delta = direction * self.config.learning_rate * self.config.feature_weights[feature]
            if not math.isfinite(delta):
                raise ProfileUpdateError("A contribuição calculada deve ser finita.")
            applied: dict[str, float] = {}
            for value in values:
                previous = affinities[feature].get(value, 0.0)
                updated = min(self.config.maximum_affinity, max(self.config.minimum_affinity, previous + delta))
                affinities[feature][value] = round(updated, 12)
                applied[value] = round(affinities[feature][value] - previous, 12)
            if applied:
                feature_contributions[feature] = applied
        contributions[event.event_id] = {
            "feedback": event.feedback,
            "direction": direction,
            "features": feature_contributions,
        }

        presented = _append_unique(state.presented_movie_ids, event.movie_id)
        consumed = state.consumed_movie_ids
        if self.config.marks_consumed:
            consumed = _append_unique(consumed, event.movie_id)
        new_state = UserState(
            subject_id=state.subject_id,
            identity_kind=state.identity_kind,
            logical_timestamp=logical_timestamp,
            version=state.version + 1,
            initial_profile=state.to_dict()["initial_profile"],
            learned_preferences={
                "affinities": affinities,
                "event_contributions": contributions,
            },
            presented_movie_ids=presented,
            consumed_movie_ids=consumed,
            history=state.history + (event,),
        )
        return StateSnapshot(
            state=new_state,
            previous_state_id=state.state_id,
            transition_event_id=event.event_id,
        )

    def method_manifest(self) -> dict[str, Any]:
        try:
            config_path = self.config.source_path.relative_to(PROJECT_ROOT)
        except ValueError:
            config_path = Path(self.config.source_path.name)
        return {
            "component": "profile_update",
            "policy": "bounded_additive_decay",
            "version": self.config.version,
            "config_path": config_path.as_posix(),
            "config_sha256": self.config.source_sha256,
            "config_snapshot": dict(self.config.snapshot),
            "fit_partition": "none",
            "holdout_used_for_selection": False,
        }
