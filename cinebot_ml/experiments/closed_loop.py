"""Contrato e auditoria da exposição em ciclo fechado da execução v2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.simulation.temporal import SimulationConfig, SimulationResult


DEFAULT_CLOSED_LOOP_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/closed_loop_v2.yaml"
DEFAULT_CLOSED_LOOP_SCHEMA_PATH = PROJECT_ROOT / "configs/experiments/closed_loop_v2.schema.json"
DEFAULT_V2_1_CLOSED_LOOP_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/closed_loop_v2_1.yaml"
DEFAULT_V2_1_CLOSED_LOOP_SCHEMA_PATH = (
    PROJECT_ROOT / "configs/experiments/closed_loop_v2_1.schema.json"
)


class ClosedLoopExposureError(ValueError):
    """Trajetória ou configuração incompatível com a exposição fechada v2."""


@dataclass(frozen=True)
class ClosedLoopExposureConfig:
    version: str
    protocol_version: str
    exposure_policy: str
    condition_events: Mapping[str, int | str]
    full_history_events: int
    feedback_rank: int
    clock_step_seconds: int
    tie_breaker: str
    exhaustion_status: str
    paired_methods: tuple[str, ...]
    pairing_fields: tuple[str, ...]
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]

    def events_for(self, condition: str) -> int:
        try:
            value = self.condition_events[condition]
        except KeyError as exc:
            raise ClosedLoopExposureError(f"Condição desconhecida: {condition}") from exc
        return self.full_history_events if value == "all" else int(value)

    def as_simulation_config(self) -> SimulationConfig:
        """Adapta o contrato v2 ao replay temporal sem reutilizar a política v1."""
        return SimulationConfig(
            self.version,
            self.protocol_version,
            self.condition_events,
            self.full_history_events,
            self.feedback_rank,
            self.clock_step_seconds,
            self.source_path,
            self.source_sha256,
            self.snapshot,
        )

    def manifest(self) -> dict[str, Any]:
        manifest = {
            "exposure_policy": self.exposure_policy,
            "version": self.version,
            "protocol_version": self.protocol_version,
            "config_sha256": self.source_sha256,
            "selection": {
                "rank": self.feedback_rank,
                "tie_breaker": self.tie_breaker,
            },
            "exhaustion": {"status": self.exhaustion_status},
            "pairing": {
                "methods": list(self.paired_methods),
                "fields": list(self.pairing_fields),
            },
        }
        shared_exposure = self.snapshot.get("shared_exposure")
        if isinstance(shared_exposure, Mapping):
            manifest["shared_exposure"] = dict(shared_exposure)
        return manifest


def load_closed_loop_exposure_config(
    path: Path = DEFAULT_CLOSED_LOOP_CONFIG_PATH,
    *,
    schema_path: Path = DEFAULT_CLOSED_LOOP_SCHEMA_PATH,
) -> ClosedLoopExposureConfig:
    """Carrega schema e lock antes de executar uma trajetória v2."""
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.resolve().read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ClosedLoopExposureError(f"Não foi possível carregar exposição fechada: {exc}") from exc
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        raise ClosedLoopExposureError(f"Configuração de exposição fechada inválida: {error.message}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ClosedLoopExposureError("Lock da exposição fechada ausente ou inválido.") from exc
    if lock.get("sha256") != digest:
        raise ClosedLoopExposureError("A exposição fechada foi alterada sem nova versão.")
    selection = payload["selection"]
    return ClosedLoopExposureConfig(
        str(payload["version"]),
        str(payload["protocol_version"]),
        str(payload["exposure_policy"]),
        dict(payload["conditions"]),
        int(payload["full_history_events"]),
        int(selection["rank"]),
        int(payload["logical_clock_step_seconds"]),
        str(selection["tie_breaker"]),
        str(payload["exhaustion"]["status"]),
        tuple(payload["pairing"]["methods"]),
        tuple(payload["pairing"]["fields"]),
        path,
        digest,
        dict(payload),
    )


def audit_closed_loop_trajectory(
    result: SimulationResult,
    config: ClosedLoopExposureConfig,
) -> dict[str, Any]:
    """Verifica causalidade, seleção e ordem de uma trajetória serializada."""
    failures: list[dict[str, Any]] = []
    target = config.events_for(result.manifest["scenario"]["condition"])
    if result.target_events != target:
        failures.append({"reason": "target_events_mismatch"})
    if result.status == "completed" and len(result.steps) != target:
        failures.append({"reason": "incomplete_completed_trajectory"})
    if result.status == config.exhaustion_status and len(result.steps) >= target:
        failures.append({"reason": "invalid_exhaustion"})
    if result.status not in {"completed", config.exhaustion_status}:
        failures.append({"reason": "invalid_trajectory_status", "status": result.status})
    for index, step in enumerate(result.steps):
        if step.interaction != index or step.feedback_event.sequence != index + 1:
            failures.append({"step": index, "reason": "sequence_mismatch"})
        if len(step.ranking.ranked_items) < config.feedback_rank:
            failures.append({"step": index, "reason": "selection_rank_missing"})
            continue
        selected = step.ranking.ranked_items[config.feedback_rank - 1].movie_id
        if step.feedback_event.movie_id != selected:
            failures.append({"step": index, "reason": "feedback_not_from_prior_ranking"})
        if tuple(item.movie_id for item in step.ranking.ranked_items) != step.presented_movie_ids:
            failures.append({"step": index, "reason": "presented_items_mismatch"})
        input_history = step.input_snapshot.state.history
        if any(event.timestamp >= step.logical_timestamp for event in input_history):
            failures.append({"step": index, "reason": "future_feedback_in_request"})
        if any(event.event_id == step.feedback_event.event_id for event in input_history):
            failures.append({"step": index, "reason": "feedback_visible_before_update"})
        output_history = step.output_snapshot.state.history
        if not output_history or output_history[-1].event_id != step.feedback_event.event_id:
            failures.append({"step": index, "reason": "update_missing_feedback"})
        if step.ranking.metadata.get("candidate_set_id") != step.candidate_set_id:
            failures.append({"step": index, "reason": "candidate_set_mismatch"})
    return {
        "complete": not failures,
        "status": result.status,
        "target_events": target,
        "observed_events": len(result.steps),
        "failures": failures,
    }


def audit_shared_exposure_trajectory(
    trajectory: Mapping[str, Any],
    config: ClosedLoopExposureConfig,
) -> dict[str, Any]:
    """Confere a trajetória v2.1 comum antes de derivar B4 e B5."""
    failures: list[dict[str, Any]] = []
    if config.exposure_policy != "shared-seeded-v2.1":
        failures.append({"reason": "unexpected_exposure_policy"})
    if trajectory.get("exposure_policy") != config.exposure_policy:
        failures.append({"reason": "exposure_policy_mismatch"})
    if not trajectory.get("comparison_id"):
        failures.append({"reason": "comparison_id_missing"})
    if "method" in trajectory:
        failures.append({"reason": "method_dependent_trajectory"})
    target = config.events_for(str(trajectory.get("condition", "")))
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        failures.append({"reason": "steps_missing"})
        steps = []
    if trajectory.get("target_events") != target:
        failures.append({"reason": "target_events_mismatch"})
    if trajectory.get("status") == "completed" and len(steps) != target:
        failures.append({"reason": "incomplete_completed_trajectory"})
    if trajectory.get("status") not in {"completed", config.exhaustion_status}:
        failures.append({"reason": "invalid_trajectory_status"})

    seen: set[Any] = set()
    prior_event_ids: list[str] = []
    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            failures.append({"step": index, "reason": "invalid_step"})
            continue
        feedback = step.get("feedback_event", {})
        candidates = step.get("candidate_movie_ids", [])
        movie_id = step.get("exposed_movie_id")
        if step.get("interaction") != index:
            failures.append({"step": index, "reason": "sequence_mismatch"})
        if not step.get("candidate_set_id") or not isinstance(candidates, list):
            failures.append({"step": index, "reason": "candidate_set_missing"})
        elif movie_id not in candidates:
            failures.append({"step": index, "reason": "exposure_not_eligible"})
        if movie_id in seen:
            failures.append({"step": index, "reason": "repeated_exposure"})
        seen.add(movie_id)
        if not isinstance(feedback, Mapping) or feedback.get("movie_id") != movie_id:
            failures.append({"step": index, "reason": "feedback_not_from_shared_exposure"})
        if feedback.get("sequence") != index + 1:
            failures.append({"step": index, "reason": "feedback_sequence_mismatch"})
        input_history = step.get("input_snapshot", {}).get("state", {}).get("history", [])
        output_history = step.get("output_snapshot", {}).get("state", {}).get("history", [])
        if [event.get("event_id") for event in input_history] != prior_event_ids:
            failures.append({"step": index, "reason": "input_history_mismatch"})
        if not output_history or output_history[-1].get("event_id") != feedback.get("event_id"):
            failures.append({"step": index, "reason": "update_missing_feedback"})
        prior_event_ids.append(str(feedback.get("event_id")))
    final_snapshot = trajectory.get("final_snapshot", {})
    if steps and final_snapshot != steps[-1].get("output_snapshot"):
        failures.append({"reason": "final_snapshot_mismatch"})
    return {
        "complete": not failures,
        "status": trajectory.get("status"),
        "target_events": target,
        "observed_events": len(steps),
        "failures": failures,
    }
