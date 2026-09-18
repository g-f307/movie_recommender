"""Estudo pareado e reproduzível dos níveis de cold start P0--P5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.personalization import StateSnapshot, UserState
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
from cinebot_ml.simulation import SimulationScenario, SyntheticAgent, TemporalSimulator


DEFAULT_COLD_START_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/cold_start_v1.yaml"
DEFAULT_COLD_START_SCHEMA_PATH = PROJECT_ROOT / "configs/experiments/cold_start_v1.schema.json"


class ColdStartValidationError(ValueError):
    """Perfil ou execução viola o protocolo de cold start."""


@dataclass(frozen=True)
class ColdStartConfig:
    version: str
    profiles: Mapping[str, tuple[str, ...]]
    methods: tuple[str, ...]
    source_sha256: str
    snapshot: Mapping[str, Any]


@dataclass(frozen=True)
class ColdStartProfile:
    profile: str
    agent_id: str
    preference_id: str
    profile_data: Mapping[str, Any]
    state: StateSnapshot


@dataclass(frozen=True)
class ColdStartReport:
    study_id: str
    manifest: Mapping[str, Any]
    benchmark: BenchmarkReport
    availability: tuple[Mapping[str, Any], ...]
    summary: tuple[Mapping[str, Any], ...]


RecommenderFactory = Any


def load_cold_start_config(path: Path = DEFAULT_COLD_START_CONFIG_PATH) -> ColdStartConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_COLD_START_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ColdStartValidationError(f"Não foi possível carregar cold start: {exc}") from exc
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        raise ColdStartValidationError(f"Configuração de cold start inválida: {error.message}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ColdStartValidationError("Lock de cold start ausente ou inválido.") from exc
    if lock.get("sha256") != digest:
        raise ColdStartValidationError("Configuração de cold start alterada sem novo lock.")
    profiles = {key: tuple(value["allowed_fields"]) for key, value in payload["profiles"].items()}
    return ColdStartConfig(
        str(payload["version"]), profiles, tuple(payload["methods"]), digest, dict(payload)
    )


def profile_payload(agent: SyntheticAgent, profile: str) -> dict[str, Any]:
    """Projeta somente informação explicitamente autorizada pelo nível."""
    genres = [genre for genre, _ in agent.preference.genres]
    payload: dict[str, Any] = {}
    if profile in {"P1", "P2", "P3", "P4", "P5"}:
        payload["ranked_genres"] = genres[:1] if profile == "P1" else genres[:3]
    if profile in {"P3", "P4", "P5"}:
        payload["decade_preference"] = agent.preference.decade
    if profile in {"P4", "P5"}:
        payload["popularity_preference"] = (
            "popular" if agent.preference.popularity_weight >= 0 else "joia_escondida"
        )
    if profile == "P5":
        payload["preferred_directors"] = list(agent.preference.directors)
        payload["preferred_keywords"] = list(agent.preference.keywords)
    if profile not in {f"P{index}" for index in range(6)}:
        raise ColdStartValidationError(f"Perfil desconhecido: {profile}")
    return payload


def validate_profile_payload(
    profile: str, payload: Mapping[str, Any], config: ColdStartConfig
) -> None:
    if profile not in config.profiles:
        raise ColdStartValidationError(f"Perfil não configurado: {profile}")
    unknown = sorted(set(payload) - set(config.profiles[profile]))
    if unknown:
        raise ColdStartValidationError(
            f"{profile} tenta reconstruir informação não autorizada: {unknown[0]}"
        )
    genres = payload.get("ranked_genres", [])
    if "ranked_genres" in payload and (
        not isinstance(genres, (list, tuple)) or not genres or len(genres) > (1 if profile == "P1" else 3)
    ):
        raise ColdStartValidationError(f"Quantidade de gêneros inválida em {profile}.")


def build_cold_start_profiles(
    agent: SyntheticAgent,
    allowed_history: StateSnapshot,
    config: ColdStartConfig | None = None,
) -> tuple[ColdStartProfile, ...]:
    config = config or load_cold_start_config()
    if allowed_history.state.subject_id != agent.agent_id:
        raise ColdStartValidationError("Histórico permitido pertence a outro agente.")
    output = []
    for profile in sorted(config.profiles):
        payload = profile_payload(agent, profile)
        validate_profile_payload(profile, payload, config)
        state = (
            StateSnapshot(
                UserState(
                    agent.agent_id, "synthetic", "2027-01-01T00:00:00Z",
                    initial_profile=payload,
                )
            )
            if profile != "P5"
            else StateSnapshot(
                UserState(
                    agent.agent_id,
                    "synthetic",
                    allowed_history.state.logical_timestamp,
                    version=allowed_history.state.version,
                    initial_profile=payload,
                    learned_preferences=allowed_history.state.learned_preferences,
                    presented_movie_ids=allowed_history.state.presented_movie_ids,
                    consumed_movie_ids=allowed_history.state.consumed_movie_ids,
                    history=allowed_history.state.history,
                ),
                allowed_history.previous_state_id,
                allowed_history.transition_event_id,
            )
        )
        output.append(
            ColdStartProfile(profile, agent.agent_id, agent.preference_id, payload, state)
        )
    return tuple(output)


def _request(
    profile: ColdStartProfile, condition: str, seed: int, k: int,
    timestamp: str, protocol_version: str,
) -> RecommendationRequest:
    return RecommendationRequest(
        "movie-recommender-acm-sac-2027", protocol_version, "B0", "1.0",
        condition, profile.profile, seed, timestamp, (), k,
        unit_id=profile.agent_id, session_id=f"cold-start-{profile.preference_id}",
        profile_data=profile.profile_data,
        history=tuple(event.to_dict() for event in profile.state.state.history),
    )


def run_cold_start_study(
    catalog: Sequence[Mapping[str, Any]],
    agent: SyntheticAgent,
    *,
    condition: str,
    k: int,
    methods: Sequence[str],
    official_k_values: Sequence[int],
    experiment_config_sha256: str,
    catalog_path: str = "data/filmes.json",
    catalog_sha256: str = "unavailable",
    factories: Mapping[str, RecommenderFactory] | None = None,
    config: ColdStartConfig | None = None,
    protocol_version: str = "protocol-v1.0",
) -> ColdStartReport:
    config = config or load_cold_start_config()
    enabled = tuple(methods)
    if not enabled or not set(enabled).issubset(set(config.methods)):
        raise ColdStartValidationError("Métodos devem pertencer ao protocolo de cold start.")
    initial_payload = profile_payload(agent, "P5")
    initial = StateSnapshot(UserState(
        agent.agent_id, "synthetic", "2027-01-01T00:00:00Z", initial_profile=initial_payload
    ))
    reference_scenario = SimulationScenario(
        "B5", condition, "P5", agent.seed, k, catalog_path, catalog_sha256,
        initial, "2027-01-01T00:00:00Z",
    )
    reference = TemporalSimulator(catalog).run(reference_scenario, agent)
    if reference.status == "failed" or reference.final_ranking is None:
        raise ColdStartValidationError("Não foi possível produzir o histórico causal de referência.")
    profiles = build_cold_start_profiles(agent, reference.final_snapshot, config)
    supplied = dict(factories or {})
    units = []
    availability = []
    for projected in profiles:
        request = _request(
            projected, condition, agent.seed, k, reference.final_ranking.logical_timestamp,
            protocol_version,
        )
        candidates = build_candidate_set(
            catalog, request, catalog_path=catalog_path, catalog_sha256=catalog_sha256
        )
        built_in = {
            "B0": lambda current: PopularityRecommender(current),
            "B1": lambda current: ContentRecommender(current),
            "B4": lambda current, state=projected.state.state: StaticPersonalizedRecommender(current, state),
            "B5": lambda current, state=projected.state: IncrementalRecommender(current, state),
        }
        recommenders: dict[str, Recommender] = {}
        for method in enabled:
            factory = supplied.get(method) or built_in.get(method)
            if factory is None:
                raise ColdStartValidationError(f"Factory obrigatória ausente para {method}.")
            recommenders[method] = factory(candidates)
        relevance = RelevanceJudgments(
            "synthetic_user",
            {movie["id"]: agent.evaluate(movie, projected.state.state.version).graded_relevance for movie in candidates.movies},
            complete=True,
            version=f"agent-{agent.config.version}",
        )
        units.append(BenchmarkUnit(
            request, candidates, relevance, recommenders,
            agent_id=agent.agent_id, persona=agent.persona,
            agent_version=agent.config.version, interaction=projected.state.state.version,
            state_version=projected.state.state.version,
            simulation_id=reference.simulation_id,
        ))
        availability.append({
            "profile": projected.profile, "candidate_set_id": candidates.candidate_set_id,
            "candidate_count": len(candidates.movies),
            "status": "available" if len(candidates.movies) >= k else "insufficient_candidates",
        })
    benchmark = run_benchmark(
        units, k_values=(k,), official_k_values=official_k_values,
        config_sha256=experiment_config_sha256,
    )
    summary = _summary(benchmark)
    identity = {
        "version": config.version, "config_sha256": config.source_sha256,
        "protocol_version": protocol_version, "condition": condition, "k": k,
        "methods": sorted(enabled),
        "agent_id": agent.agent_id, "preference_id": agent.preference_id,
        "simulation_id": reference.simulation_id, "benchmark_id": benchmark.benchmark_id,
    }
    study_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    return ColdStartReport(study_id, {**identity, "study_id": study_id}, benchmark, tuple(availability), summary)


def _summary(benchmark: BenchmarkReport) -> tuple[Mapping[str, Any], ...]:
    groups: dict[tuple[str, str], list[Any]] = {}
    for record in benchmark.individual:
        groups.setdefault((record.method, record.profile), []).append(record)
    rows = []
    for (method, profile), records in sorted(groups.items()):
        row: dict[str, Any] = {
            "method": method, "profile": profile, "unit_count": len(records),
            "failed_count": sum(record.ranking_status == "failed" for record in records),
            "candidate_count": records[0].candidate_count,
        }
        metric_names = sorted({name for record in records for name in record.metrics})
        for name in metric_names:
            values = [record.metrics.get(name) for record in records]
            numeric = [float(value) for value in values if value is not None]
            row[name] = mean(numeric) if numeric else None
        rows.append(row)
    return tuple(rows)


def write_cold_start_report(report: ColdStartReport, output: Path) -> Mapping[str, Path]:
    root = output / report.study_id
    paths = {
        "manifest": root / "cold_start.manifest.json",
        "raw": root / "cold_start.individual.jsonl",
        "availability": root / "cold_start.availability.csv",
        "summary": root / "cold_start.summary.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Resultados de cold start já existem e não serão sobrescritos.")
    root.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(report.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["raw"].write_text("".join(
        json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
        for record in report.benchmark.individual
    ), encoding="utf-8")
    _write_csv(paths["availability"], report.availability)
    _write_csv(paths["summary"], report.summary)
    return paths


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    from cinebot_ml.dataset import load_catalog
    from cinebot_ml.experiment_config import DEFAULT_CONFIG_PATH, load_config, sha256_file
    from cinebot_ml.ranking import SupervisedArtifact, SupervisedRecommender, TfidfArtifact, TfidfRecommender
    from cinebot_ml.simulation import build_agent

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--experiment-config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--cold-start-config", type=Path, default=DEFAULT_COLD_START_CONFIG_PATH)
    parser.add_argument("--condition", choices=[f"C{i}" for i in range(6)], required=True)
    parser.add_argument("--persona", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--method", action="append", choices=[f"B{i}" for i in range(6)])
    parser.add_argument("--b2-artifact", type=Path)
    parser.add_argument("--b3-model", type=Path)
    parser.add_argument("--b3-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        experiment = load_config(args.experiment_config)
        cold_config = load_cold_start_config(args.cold_start_config)
        methods = tuple(args.method or cold_config.methods)
        if args.k not in experiment["k_values"] or args.seed not in experiment["seeds"]:
            raise ColdStartValidationError("Seed e K devem pertencer à configuração experimental.")
        factories = {}
        if "B2" in methods:
            if args.b2_artifact is None:
                raise ColdStartValidationError("--b2-artifact é obrigatório para B2.")
            artifact = TfidfArtifact.load(args.b2_artifact)
            factories["B2"] = lambda candidates: TfidfRecommender(candidates, artifact)
        if "B3" in methods:
            if args.b3_model is None or args.b3_metadata is None:
                raise ColdStartValidationError("--b3-model e --b3-metadata são obrigatórios para B3.")
            artifact = SupervisedArtifact.load(args.b3_model, args.b3_metadata)
            factories["B3"] = lambda candidates: SupervisedRecommender(candidates, artifact)
        catalog = load_catalog(args.catalog)
        agent = build_agent(args.persona, args.seed, catalog)
        report = run_cold_start_study(
            catalog, agent, condition=args.condition, k=args.k, methods=methods,
            official_k_values=experiment["k_values"],
            experiment_config_sha256=sha256_file(args.experiment_config),
            catalog_path=args.catalog.as_posix(), catalog_sha256=sha256_file(args.catalog),
            factories=factories, config=cold_config,
            protocol_version=experiment["experiment"]["protocol_version"],
        )
        paths = write_cold_start_report(report, args.output)
        print(json.dumps({"study_id": report.study_id, "paths": {key: str(value) for key, value in paths.items()}}))
        return 0
    except (ColdStartValidationError, ValueError, FileExistsError) as exc:
        raise SystemExit(f"Erro no estudo de cold start: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
