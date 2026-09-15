"""Replay temporal reproduzível das condições incrementais C0--C5."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.dataset import load_catalog
from cinebot_ml.personalization import FeedbackEvent, ProfileUpdater, StateSnapshot, UserState
from cinebot_ml.ranking import (
    IncrementalRecommender,
    RankingResult,
    RecommendationRequest,
    StaticPersonalizedRecommender,
    build_candidate_set,
    load_incremental_config,
    load_static_personalized_config,
)
from cinebot_ml.simulation.agents import SyntheticAgent, SyntheticJudgment, build_agent


DEFAULT_SIMULATION_CONFIG_PATH = PROJECT_ROOT / "configs" / "simulation_v1.yaml"
DEFAULT_SIMULATION_SCHEMA_PATH = PROJECT_ROOT / "configs" / "simulation_v1.schema.json"
SUPPORTED_METHODS = {"B4", "B5"}


class TemporalSimulationError(ValueError):
    """Cenário, configuração ou retomada inválida."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise TemporalSimulationError("base_timestamp deve ser ISO 8601 com fuso horário.") from exc
    if parsed.tzinfo is None:
        raise TemporalSimulationError("base_timestamp deve incluir fuso horário.")
    return parsed.astimezone(timezone.utc)


def _timestamp(base: datetime, seconds: int) -> str:
    return (base + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class SimulationConfig:
    version: str
    protocol_version: str
    condition_events: Mapping[str, int | str]
    full_history_events: int
    feedback_rank: int
    clock_step_seconds: int
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]

    def events_for(self, condition: str) -> int:
        if condition not in self.condition_events:
            raise TemporalSimulationError(f"Condição desconhecida: {condition}")
        value = self.condition_events[condition]
        return self.full_history_events if value == "all" else int(value)


def load_simulation_config(path: Path = DEFAULT_SIMULATION_CONFIG_PATH) -> SimulationConfig:
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_SIMULATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise TemporalSimulationError(f"Não foi possível carregar a simulação: {exc}") from exc
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        raise TemporalSimulationError(f"Configuração de simulação inválida: {error.message}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TemporalSimulationError("Lock da simulação ausente ou inválido.") from exc
    if lock.get("sha256") != digest:
        raise TemporalSimulationError("A configuração de simulação foi alterada sem nova versão.")
    return SimulationConfig(
        version=str(payload["version"]),
        protocol_version=str(payload["protocol_version"]),
        condition_events=dict(payload["conditions"]),
        full_history_events=int(payload["full_history_events"]),
        feedback_rank=int(payload["feedback_rank"]),
        clock_step_seconds=int(payload["logical_clock_step_seconds"]),
        source_path=path,
        source_sha256=digest,
        snapshot=dict(payload),
    )


@dataclass(frozen=True)
class SimulationScenario:
    method: str
    condition: str
    profile: str
    seed: int
    k: int
    catalog_path: str
    catalog_sha256: str
    initial_state: StateSnapshot
    base_timestamp: str
    experiment_id: str = "movie-recommender-acm-sac-2027"

    def __post_init__(self) -> None:
        if self.method not in SUPPORTED_METHODS:
            raise TemporalSimulationError("A simulação temporal aceita somente B4 ou B5.")
        if self.condition not in {f"C{index}" for index in range(6)}:
            raise TemporalSimulationError(f"Condição desconhecida: {self.condition}")
        if self.profile not in {f"P{index}" for index in range(6)}:
            raise TemporalSimulationError(f"Perfil desconhecido: {self.profile}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TemporalSimulationError("seed deve ser inteira.")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k <= 0:
            raise TemporalSimulationError("k deve ser maior que zero.")
        if self.initial_state.state.version != 0:
            raise TemporalSimulationError("O cenário deve começar em um snapshot sem feedback.")
        _parse_time(self.base_timestamp)

    @property
    def scenario_id(self) -> str:
        return _canonical_hash(self.to_manifest())

    def to_manifest(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "method": self.method,
            "condition": self.condition,
            "profile": self.profile,
            "seed": self.seed,
            "k": self.k,
            "catalog_path": self.catalog_path,
            "catalog_sha256": self.catalog_sha256,
            "initial_snapshot_id": self.initial_state.snapshot_id,
            "base_timestamp": self.base_timestamp,
        }


@dataclass(frozen=True)
class SimulationStep:
    interaction: int
    logical_timestamp: str
    input_snapshot: StateSnapshot
    candidate_set_id: str
    presented_movie_ids: tuple[int | str, ...]
    ranking: RankingResult
    judgment: SyntheticJudgment
    feedback_event: FeedbackEvent
    output_snapshot: StateSnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "interaction": self.interaction,
            "logical_timestamp": self.logical_timestamp,
            "input_snapshot": self.input_snapshot.to_dict(),
            "candidate_set_id": self.candidate_set_id,
            "presented_movie_ids": list(self.presented_movie_ids),
            "ranking": self.ranking.to_dict(),
            "judgment": self.judgment.__dict__,
            "feedback_event": self.feedback_event.to_dict(),
            "output_snapshot": self.output_snapshot.to_dict(),
        }


@dataclass(frozen=True)
class SimulationResult:
    scenario_id: str
    simulation_id: str
    status: str
    target_events: int
    steps: tuple[SimulationStep, ...]
    final_snapshot: StateSnapshot
    final_ranking: RankingResult | None
    manifest: Mapping[str, Any]
    failure: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {"partial", "completed", "exhausted", "failed"}:
            raise TemporalSimulationError(f"Status de simulação inválido: {self.status}")
        if len(self.steps) != self.final_snapshot.state.version:
            raise TemporalSimulationError("Passos e versão final do estado são incompatíveis.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "simulation_id": self.simulation_id,
            "status": self.status,
            "target_events": self.target_events,
            "steps": [step.to_dict() for step in self.steps],
            "final_snapshot": self.final_snapshot.to_dict(),
            "final_ranking": self.final_ranking.to_dict() if self.final_ranking else None,
            "manifest": dict(self.manifest),
            "failure": self.failure,
        }


Checkpoint = Callable[[SimulationResult], None]


class TemporalSimulator:
    """Coordena ranking, feedback e atualização sem acesso a eventos futuros."""

    def __init__(self, catalog: Sequence[Mapping[str, Any]], config: SimulationConfig | None = None):
        self.catalog = tuple(dict(movie) for movie in catalog)
        self.config = config or load_simulation_config()
        self.updater = ProfileUpdater()

    def _request(self, scenario: SimulationScenario, snapshot: StateSnapshot, logical_time: str) -> RecommendationRequest:
        state = snapshot.state
        version = (
            load_static_personalized_config().version
            if scenario.method == "B4"
            else load_incremental_config().version
        )
        history = tuple(event.to_dict() for event in state.history)
        return RecommendationRequest(
            experiment_id=scenario.experiment_id,
            protocol_version=self.config.protocol_version,
            method=scenario.method,
            method_version=version,
            condition=scenario.condition,
            profile=scenario.profile,
            seed=scenario.seed,
            logical_timestamp=logical_time,
            candidate_movie_ids=(),
            k=scenario.k,
            unit_id=state.subject_id,
            session_id=scenario.scenario_id,
            profile_data=state.to_dict()["initial_profile"],
            history=history,
        )

    def _rank(self, scenario: SimulationScenario, snapshot: StateSnapshot, logical_time: str) -> tuple[RankingResult, Any]:
        request = self._request(scenario, snapshot, logical_time)
        candidates = build_candidate_set(
            self.catalog,
            request,
            catalog_path=scenario.catalog_path,
            catalog_sha256=scenario.catalog_sha256,
        )
        request = candidates.attach_to_request(request)
        if scenario.method == "B4":
            recommender = StaticPersonalizedRecommender(candidates, scenario.initial_state.state)
        else:
            recommender = IncrementalRecommender(candidates, snapshot)
        return recommender.recommend(request), candidates

    def _manifest(self, scenario: SimulationScenario, agent: SyntheticAgent) -> dict[str, Any]:
        return {
            "simulation_version": self.config.version,
            "protocol_version": self.config.protocol_version,
            "simulation_config_sha256": self.config.source_sha256,
            "scenario": scenario.to_manifest(),
            "agent": agent.manifest(),
            "profile_update": self.updater.method_manifest(),
            "temporal_policy": "rank_then_feedback_then_update",
            "target_events": self.config.events_for(scenario.condition),
        }

    def _result(
        self,
        scenario: SimulationScenario,
        agent: SyntheticAgent,
        status: str,
        steps: Sequence[SimulationStep],
        snapshot: StateSnapshot,
        final_ranking: RankingResult | None,
        failure: str | None = None,
    ) -> SimulationResult:
        manifest = self._manifest(scenario, agent)
        simulation_id = _canonical_hash({"scenario_id": scenario.scenario_id, "manifest": manifest})
        return SimulationResult(
            scenario.scenario_id,
            simulation_id,
            status,
            int(manifest["target_events"]),
            tuple(steps),
            snapshot,
            final_ranking,
            manifest,
            failure,
        )

    def run(
        self,
        scenario: SimulationScenario,
        agent: SyntheticAgent,
        *,
        resume_from: SimulationResult | None = None,
        checkpoint: Checkpoint | None = None,
    ) -> SimulationResult:
        if agent.agent_id != scenario.initial_state.state.subject_id:
            raise TemporalSimulationError("O agente diverge do sujeito do cenário.")
        target = self.config.events_for(scenario.condition)
        if resume_from:
            if resume_from.scenario_id != scenario.scenario_id:
                raise TemporalSimulationError("Resultado parcial pertence a outro cenário.")
            expected_id = self._result(
                scenario, agent, "partial", (), scenario.initial_state, None
            ).simulation_id
            if resume_from.simulation_id != expected_id:
                raise TemporalSimulationError("Resultado parcial usa políticas ou agente incompatíveis.")
            steps = list(resume_from.steps)
            current = resume_from.final_snapshot
        else:
            steps = []
            current = scenario.initial_state
        if len(steps) > target:
            raise TemporalSimulationError("Resultado parcial ultrapassa a condição solicitada.")
        base = _parse_time(scenario.base_timestamp)
        try:
            while len(steps) < target:
                interaction = len(steps)
                rank_time = _timestamp(base, interaction * self.config.clock_step_seconds + 1)
                ranking, candidates = self._rank(scenario, current, rank_time)
                if not ranking.ranked_items:
                    result = self._result(scenario, agent, "exhausted", steps, current, ranking)
                    if checkpoint:
                        checkpoint(result)
                    return result
                selected = ranking.ranked_items[self.config.feedback_rank - 1]
                movie = next(movie for movie in candidates.movies if movie["id"] == selected.movie_id)
                judgment = agent.evaluate(movie, interaction)
                event_time = _timestamp(base, (interaction + 1) * self.config.clock_step_seconds)
                event = FeedbackEvent(
                    event_id=f"{scenario.scenario_id}-event-{interaction + 1:04d}",
                    movie_id=selected.movie_id,
                    feedback=judgment.feedback,
                    timestamp=event_time,
                    sequence=interaction + 1,
                    source=judgment.source,
                    metadata={"utility": judgment.utility, "graded_relevance": judgment.graded_relevance},
                )
                updated = self.updater.update(current, event, movie, logical_timestamp=event_time)
                steps.append(
                    SimulationStep(
                        interaction,
                        rank_time,
                        current,
                        candidates.candidate_set_id,
                        tuple(item.movie_id for item in ranking.ranked_items),
                        ranking,
                        judgment,
                        event,
                        updated,
                    )
                )
                current = updated
                if checkpoint:
                    checkpoint(self._result(scenario, agent, "partial", steps, current, None))
            final_time = _timestamp(base, target * self.config.clock_step_seconds + 1)
            final_ranking, _ = self._rank(scenario, current, final_time)
            result = self._result(scenario, agent, "completed", steps, current, final_ranking)
        except Exception as exc:  # preservar a evidência parcial antes de propagar o estado de falha
            result = self._result(scenario, agent, "failed", steps, current, None, f"{type(exc).__name__}: {exc}")
        if checkpoint:
            checkpoint(result)
        return result

    def run_many(
        self,
        runs: Sequence[tuple[SimulationScenario, SyntheticAgent]],
        *,
        checkpoint: Checkpoint | None = None,
    ) -> tuple[SimulationResult, ...]:
        """Executa cenários de múltiplos agentes e seeds na ordem fornecida."""
        return tuple(
            self.run(scenario, agent, checkpoint=checkpoint)
            for scenario, agent in runs
        )


def write_simulation_result(result: SimulationResult, path: Path) -> None:
    """Persiste um checkpoint de forma atômica, inclusive quando parcial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _initial_profile(agent: SyntheticAgent, profile: str) -> dict[str, Any]:
    if profile == "P0":
        return {}
    genres = [genre for genre, _ in agent.preference.genres]
    payload: dict[str, Any] = {"ranked_genres": genres[:1] if profile == "P1" else genres[:3]}
    if profile in {"P3", "P4", "P5"}:
        payload["decade"] = agent.preference.decade
    if profile in {"P4", "P5"}:
        payload["popularity"] = "popular" if agent.preference.popularity_weight >= 0 else "niche"
    if profile == "P5":
        payload["directors"] = list(agent.preference.directors)
        payload["keywords"] = list(agent.preference.keywords)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Executa uma condição temporal com agente sintético.")
    parser.add_argument("--method", choices=sorted(SUPPORTED_METHODS), default="B5")
    parser.add_argument("--condition", choices=[f"C{i}" for i in range(6)], required=True)
    parser.add_argument("--profile", choices=[f"P{i}" for i in range(6)], default="P4")
    parser.add_argument("--persona", default="consistent")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--catalog", type=Path, default=PROJECT_ROOT / "data" / "filmes.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    catalog = load_catalog(args.catalog)
    digest = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    agent = build_agent(args.persona, args.seed, catalog)
    initial = StateSnapshot(
        UserState(
            agent.agent_id,
            "synthetic",
            "2027-01-01T00:00:00Z",
            initial_profile=_initial_profile(agent, args.profile),
        )
    )
    scenario = SimulationScenario(
        args.method,
        args.condition,
        args.profile,
        args.seed,
        args.k,
        args.catalog.as_posix(),
        digest,
        initial,
        "2027-01-01T00:00:00Z",
    )
    output = args.output or PROJECT_ROOT / "results" / "raw" / "simulations" / f"{scenario.scenario_id}.json"
    result = TemporalSimulator(catalog).run(
        scenario,
        agent,
        checkpoint=lambda partial: write_simulation_result(partial, output),
    )
    print(json.dumps({"simulation_id": result.simulation_id, "status": result.status, "output": output.as_posix()}))
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
