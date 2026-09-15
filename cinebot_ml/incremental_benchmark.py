"""Integra checkpoints sintéticos temporais ao benchmark unificado B0--B5."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping, Sequence

from cinebot_ml.ranking import (
    BenchmarkReport,
    BenchmarkUnit,
    CandidateSet,
    ContentRecommender,
    IncrementalRecommender,
    PopularityRecommender,
    RecommendationRequest,
    Recommender,
    RelevanceJudgments,
    StaticPersonalizedRecommender,
    build_candidate_set,
    run_benchmark,
)
from cinebot_ml.simulation import SimulationResult, SyntheticAgent


RecommenderFactory = Callable[[CandidateSet], Recommender]


class IncrementalBenchmarkError(ValueError):
    """Simulação e métodos incompatíveis com uma comparação temporal justa."""


def _checkpoints(simulation: SimulationResult):
    for step in simulation.steps:
        yield step.interaction, step.logical_timestamp, step.input_snapshot, step.candidate_set_id
    if simulation.final_ranking is not None:
        yield (
            len(simulation.steps),
            simulation.final_ranking.logical_timestamp,
            simulation.final_snapshot,
            simulation.final_ranking.metadata.get("candidate_set_id"),
        )


def build_incremental_benchmark_units(
    simulation: SimulationResult,
    catalog: Sequence[Mapping[str, object]],
    agent: SyntheticAgent,
    *,
    methods: Sequence[str] = ("B4", "B5"),
    factories: Mapping[str, RecommenderFactory] | None = None,
) -> tuple[BenchmarkUnit, ...]:
    """Converte cada checkpoint da trajetória em uma unidade não agregada."""
    enabled = tuple(methods)
    if not enabled or len(enabled) != len(set(enabled)):
        raise IncrementalBenchmarkError("Métodos devem formar uma lista não vazia e sem duplicatas.")
    invalid = sorted(set(enabled) - {f"B{index}" for index in range(6)})
    if invalid:
        raise IncrementalBenchmarkError(f"Método não suportado: {invalid[0]}")
    if simulation.manifest.get("agent", {}).get("agent_id") != agent.agent_id:
        raise IncrementalBenchmarkError("Agente diverge do manifesto da simulação.")
    if simulation.manifest.get("agent", {}).get("source") != "synthetic_user":
        raise IncrementalBenchmarkError("A fonte da relevância incremental deve ser synthetic_user.")
    if not simulation.steps and simulation.final_snapshot.state.version != 0:
        raise IncrementalBenchmarkError("Não foi possível reconstruir o estado inicial.")
    initial_snapshot = (
        simulation.steps[0].input_snapshot if simulation.steps else simulation.final_snapshot
    )
    scenario = simulation.manifest["scenario"]
    protocol_version = str(simulation.manifest.get("protocol_version", "protocol-v1.0"))
    supplied = dict(factories or {})
    units = []
    for interaction, logical_time, snapshot, expected_candidate_id in _checkpoints(simulation):
        history = tuple(event.to_dict() for event in snapshot.state.history)
        base_request = RecommendationRequest(
            experiment_id=str(scenario["experiment_id"]),
            protocol_version=protocol_version,
            method=enabled[0],
            method_version="1.0",
            condition=str(scenario["condition"]),
            profile=str(scenario["profile"]),
            seed=int(scenario["seed"]),
            logical_timestamp=logical_time,
            candidate_movie_ids=(),
            k=int(scenario["k"]),
            unit_id=agent.agent_id,
            session_id=simulation.simulation_id,
            profile_data=snapshot.state.to_dict()["initial_profile"],
            history=history,
        )
        candidates = build_candidate_set(
            catalog,
            base_request,
            catalog_path=str(scenario["catalog_path"]),
            catalog_sha256=str(scenario["catalog_sha256"]),
        )
        if expected_candidate_id != candidates.candidate_set_id:
            raise IncrementalBenchmarkError(
                f"Candidatos divergentes no checkpoint {interaction}; comparação bloqueada."
            )
        built_in: dict[str, RecommenderFactory] = {
            "B0": PopularityRecommender,
            "B1": ContentRecommender,
            "B4": lambda current: StaticPersonalizedRecommender(current, initial_snapshot.state),
            "B5": lambda current: IncrementalRecommender(current, snapshot),
        }
        recommenders = {}
        for method in enabled:
            factory = supplied.get(method) or built_in.get(method)
            if factory is None:
                raise IncrementalBenchmarkError(f"Factory obrigatória ausente para {method}.")
            recommenders[method] = factory(candidates)
        values = {
            movie["id"]: agent.evaluate(movie, interaction).graded_relevance
            for movie in candidates.movies
        }
        relevance = RelevanceJudgments(
            "synthetic_user",
            values,
            complete=True,
            version=f"agent-{agent.config.version}",
        )
        units.append(
            BenchmarkUnit(
                replace(base_request, candidate_movie_ids=candidates.movie_ids),
                candidates,
                relevance,
                recommenders,
                agent_id=agent.agent_id,
                persona=agent.persona,
                agent_version=agent.config.version,
                interaction=interaction,
                state_version=snapshot.state.version,
                simulation_id=simulation.simulation_id,
            )
        )
    return tuple(units)


def run_incremental_benchmark(
    simulation: SimulationResult,
    catalog: Sequence[Mapping[str, object]],
    agent: SyntheticAgent,
    *,
    methods: Sequence[str],
    k_values: Sequence[int],
    official_k_values: Sequence[int],
    config_sha256: str,
    git_commit: str = "unavailable",
    factories: Mapping[str, RecommenderFactory] | None = None,
) -> BenchmarkReport:
    units = build_incremental_benchmark_units(
        simulation, catalog, agent, methods=methods, factories=factories
    )
    if not units:
        raise IncrementalBenchmarkError("A simulação não possui checkpoints avaliáveis.")
    return run_benchmark(
        units,
        k_values=k_values,
        official_k_values=official_k_values,
        config_sha256=config_sha256,
        git_commit=git_commit,
    )
