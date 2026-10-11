"""Unidades sintéticas pareadas e catálogos isolados por execução."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cinebot_ml.config import DEFAULT_DATA_PATH, PROJECT_ROOT
from cinebot_ml.dataset import load_catalog
from cinebot_ml.experiment_config import sha256_file
from cinebot_ml.experimental_splits import reconstruct_assignments
from cinebot_ml.experiments.cold_start import profile_payload
from cinebot_ml.experiments.closed_loop import (
    ClosedLoopExposureConfig,
    audit_closed_loop_trajectory,
    load_closed_loop_exposure_config,
)
from cinebot_ml.experiments.execution import (
    ExecutionSpec,
    ExecutionSpecError,
    validate_matrix_execution,
)
from cinebot_ml.experiments.matrix import ExperimentCell, ExperimentMatrix, load_experiment_matrix
from cinebot_ml.personalization import FeedbackEvent, ProfileUpdater, StateSnapshot, UserState
from cinebot_ml.ranking.candidates import build_candidate_set
from cinebot_ml.ranking.contracts import RecommendationRequest
from cinebot_ml.simulation.agents import (
    ProfileObservationConfig,
    build_agent,
    build_declared_preference,
    load_agent_config,
    load_profile_observation_config,
    profile_observation_manifest,
)
from cinebot_ml.simulation.temporal import (
    SimulationScenario,
    TemporalSimulator,
    load_simulation_config,
)


HOLDOUT_CONFIG = PROJECT_ROOT / "configs/experiment_v1_holdout.yaml"
SPLIT = PROJECT_ROOT / "results/manifests/splits/movie_id_split.json"
HOLDOUT_CATALOG = PROJECT_ROOT / "results/derived/test_catalog.json"
UNITS = PROJECT_ROOT / "results/units_v1_1"
EPOCH = datetime(2027, 1, 1, tzinfo=timezone.utc)
_SIMULATION = load_simulation_config()
EVENTS = {f"C{index}": _SIMULATION.events_for(f"C{index}") for index in range(6)}
PRIVATE_MARKERS = ("api_key", "apikey", "credential", "email", "name", "password",
                   "phone", "secret", "token")


def _time(step: int) -> str:
    return (EPOCH + timedelta(minutes=step)).isoformat().replace("+00:00", "Z")


def build_holdout_catalog(source: Path = DEFAULT_DATA_PATH, split_path: Path = SPLIT,
                          output: Path = HOLDOUT_CATALOG) -> tuple[list[dict], str]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = reconstruct_assignments(split)
    selected: dict[int | str, dict] = {}
    for movie in load_catalog(source):
        movie_id = movie["id"]
        if assignments.get(str(movie_id)) == "test":
            selected.setdefault(movie_id, movie)
    expected = {movie_id for movie_id, partition in assignments.items() if partition == "test"}
    if {str(movie_id) for movie_id in selected} != expected:
        raise ValueError("Catálogo não cobre todos os movie_id do holdout.")
    groups: dict[str, list[dict]] = defaultdict(list)
    for movie in sorted(selected.values(), key=lambda item: str(item["id"])):
        groups[str(movie["perfil"])].append(movie)
    encoded = json.dumps({"perfis": dict(sorted(groups.items()))}, ensure_ascii=False,
                         sort_keys=True) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError("Catálogo de holdout existente diverge da fonte.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return load_catalog(output), sha256_file(output)


def build_holdout_catalog_for_execution(execution_spec: ExecutionSpec) -> tuple[list[dict], str]:
    """Cria o catálogo da execução declarada sem recorrer a saídas legadas."""
    return build_holdout_catalog(
        source=execution_spec.resolve_path("source_catalog"),
        split_path=execution_spec.resolve_path("split_manifest"),
        output=execution_spec.resolve_path("catalog"),
    )


def _unit(
    cell: ExperimentCell,
    catalog: list[dict],
    catalog_sha256: str,
    *,
    catalog_path: str = "results/derived/test_catalog.json",
    exposure_policy: str = "neutral-exposure-v1",
    profile_observation_config: ProfileObservationConfig | None = None,
) -> dict:
    if exposure_policy != "neutral-exposure-v1":
        raise ExecutionSpecError(
            "A exposição fechada v2 exige write_units com uma especificação de execução versionada."
        )
    agent = build_agent(cell.persona, cell.seed, catalog)
    declared_preference = (
        build_declared_preference(agent, profile_observation_config)
        if profile_observation_config is not None
        else None
    )
    initial = StateSnapshot(
        UserState(
            agent.agent_id,
            "synthetic",
            _time(0),
            initial_profile=profile_payload(declared_preference or agent, cell.profile),
        ),
        provenance=(
            profile_observation_manifest(agent, declared_preference)
            if declared_preference is not None
            else {}
        ),
    )
    state = initial
    exposure = list(catalog)
    random.Random(f"{exposure_policy}:{agent.agent_id}").shuffle(exposure)
    if len(exposure) <= EVENTS[cell.condition]:
        raise ValueError("Holdout insuficiente para o histórico causal.")
    updater = ProfileUpdater()
    for index, movie in enumerate(exposure[:EVENTS[cell.condition]], 1):
        judgment = agent.evaluate(movie, index - 1)
        event = FeedbackEvent(
            event_id=f"neutral-{agent.agent_id}-{index:04d}", movie_id=movie["id"],
            feedback=judgment.feedback, timestamp=_time(index), sequence=index,
            source="synthetic_user", metadata={"graded_relevance": judgment.graded_relevance},
        )
        # O contrato de estado bloqueia chaves com marcadores de segredo; termos
        # públicos do TMDB que os contenham não podem entrar nas afinidades.
        exposure_movie = dict(movie)
        exposure_movie["palavras_chave"] = [
            term for term in movie.get("palavras_chave", []) or []
            if not any(marker in str(term).lower() for marker in PRIVATE_MARKERS)
        ]
        exposure_movie["sinopse"] = " ".join(
            word for word in str(movie.get("sinopse") or "").split()
            if not any(marker in word.lower() for marker in PRIVATE_MARKERS)
        )
        state = updater.update(state, event, exposure_movie, logical_timestamp=_time(index))
    request = RecommendationRequest(
        "movie-recommender-acm-sac-2027", "protocol-v1.0", "B0", "1.0",
        cell.condition, cell.profile, cell.seed, _time(EVENTS[cell.condition] + 1),
        (), cell.k, unit_id=agent.agent_id,
        session_id=f"neutral-{agent.agent_id}",
        profile_data=profile_payload(declared_preference or agent, cell.profile),
        history=tuple(event.to_dict() for event in state.state.history),
    )
    candidates = build_candidate_set(catalog, request,
                                     catalog_path=catalog_path,
                                     catalog_sha256=catalog_sha256)
    judgments = [{"movie_id": movie["id"],
                  "value": agent.evaluate(movie, EVENTS[cell.condition]).graded_relevance}
                 for movie in candidates.movies]
    result = {
        "relevance_source": "synthetic_user", "relevance_version": "neutral-exposure-v1",
        "units": [{"request": request.to_dict(), "relevance": judgments, "complete": True,
                   "initial_snapshot": initial.to_dict(), "state_snapshot": state.to_dict(),
                   "agent_id": agent.agent_id, "persona": cell.persona,
                   "agent_version": agent.config.version,
                   "simulation_id": hashlib.sha256(cell.comparison_id.encode()).hexdigest()[:24],
                   "interaction": EVENTS[cell.condition],
                   "state_version": state.state.version}],
    }
    if declared_preference is not None:
        result["profile_observation"] = profile_observation_manifest(
            agent, declared_preference
        )
    return result


def _safe_catalog(catalog: list[dict]) -> list[dict]:
    """Remove termos bloqueados antes de qualquer atualização de afinidade."""
    safe = []
    for movie in catalog:
        normalized = dict(movie)
        normalized["palavras_chave"] = [
            term for term in movie.get("palavras_chave", []) or []
            if not any(marker in str(term).lower() for marker in PRIVATE_MARKERS)
        ]
        normalized["sinopse"] = " ".join(
            word for word in str(movie.get("sinopse") or "").split()
            if not any(marker in word.lower() for marker in PRIVATE_MARKERS)
        )
        safe.append(normalized)
    return safe


def _closed_loop_unit(
    cell: ExperimentCell,
    catalog: list[dict],
    catalog_sha256: str,
    *,
    catalog_path: str,
    exposure_config: ClosedLoopExposureConfig,
    profile_observation_config: ProfileObservationConfig | None,
) -> dict:
    if cell.method not in exposure_config.paired_methods:
        raise ExecutionSpecError("A exposição fechada v2 aceita somente os métodos pareados B4 e B5.")
    safe_catalog = _safe_catalog(catalog)
    agent = build_agent(cell.persona, cell.seed, safe_catalog)
    declared_preference = (
        build_declared_preference(agent, profile_observation_config)
        if profile_observation_config is not None
        else None
    )
    initial = StateSnapshot(
        UserState(
            agent.agent_id,
            "synthetic",
            _time(0),
            initial_profile=profile_payload(declared_preference or agent, cell.profile),
        ),
        provenance=(
            profile_observation_manifest(agent, declared_preference)
            if declared_preference is not None
            else {}
        ),
    )
    scenario = SimulationScenario(
        cell.method,
        cell.condition,
        cell.profile,
        cell.seed,
        cell.k,
        catalog_path,
        catalog_sha256,
        initial,
        _time(0),
        experiment_id="movie-recommender-acm-sac-2027",
    )
    simulator = TemporalSimulator(safe_catalog, exposure_config.as_simulation_config())
    trajectory = simulator.run(scenario, agent)
    audit = audit_closed_loop_trajectory(trajectory, exposure_config)
    if not audit["complete"]:
        raise ExecutionSpecError(f"Auditoria causal falhou: {audit['failures'][0]['reason']}")
    if trajectory.status == "failed" or trajectory.final_ranking is None:
        raise ExecutionSpecError("A trajetória fechada falhou antes do ranking de avaliação.")
    _, candidates = simulator._rank(
        scenario,
        trajectory.final_snapshot,
        trajectory.final_ranking.logical_timestamp,
    )
    request = candidates.attach_to_request(
        simulator._request(
            scenario,
            trajectory.final_snapshot,
            trajectory.final_ranking.logical_timestamp,
        )
    )
    judgments = [
        {
            "movie_id": movie["id"],
            "value": agent.evaluate(movie, len(trajectory.steps)).graded_relevance,
        }
        for movie in candidates.movies
    ]
    pairing_key = {
        "agent_id": agent.agent_id,
        "persona": cell.persona,
        "seed": cell.seed,
        "condition": cell.condition,
        "profile": cell.profile,
        "k": cell.k,
        "protocol_version": exposure_config.protocol_version,
    }
    result = {
        "exposure_policy": "closed-loop-v2",
        "exposure_manifest": exposure_config.manifest(),
        "pairing": {
            "pairing_id": hashlib.sha256(
                json.dumps(pairing_key, sort_keys=True).encode("utf-8")
            ).hexdigest()[:24],
            "key": pairing_key,
            "method": cell.method,
        },
        "trajectory": trajectory.to_dict(),
        "trajectory_audit": audit,
        "relevance_source": "synthetic_user",
        "relevance_version": "closed-loop-v2",
        "units": [
            {
                "request": request.to_dict(),
                "relevance": judgments,
                "complete": True,
                "initial_snapshot": initial.to_dict(),
                "state_snapshot": trajectory.final_snapshot.to_dict(),
                "agent_id": agent.agent_id,
                "persona": cell.persona,
                "agent_version": agent.config.version,
                "simulation_id": trajectory.simulation_id,
                "interaction": len(trajectory.steps),
                "state_version": trajectory.final_snapshot.state.version,
            }
        ],
    }
    if declared_preference is not None:
        result["profile_observation"] = profile_observation_manifest(
            agent, declared_preference
        )
    return result


def write_units(
    matrix: ExperimentMatrix,
    catalog: list[dict],
    catalog_sha256: str,
    output: Path = UNITS,
    *,
    execution_spec: ExecutionSpec | None = None,
) -> int:
    catalog_path = "results/derived/test_catalog.json"
    exposure_policy = "neutral-exposure-v1"
    profile_observation_config: ProfileObservationConfig | None = None
    closed_loop_config: ClosedLoopExposureConfig | None = None
    if execution_spec is not None:
        validate_matrix_execution(matrix, execution_spec)
        if output.resolve() != execution_spec.resolve_path("units"):
            raise ExecutionSpecError("As unidades v2 devem usar exclusivamente paths.units da execução.")
        catalog_path = execution_spec.paths["catalog"]
        exposure_policy = execution_spec.exposure_policy
        if execution_spec.profile_observation_config is not None:
            profile_observation_config = load_profile_observation_config(
                execution_spec.resolve_path("profile_observation_config")
            )
        if exposure_policy == "closed-loop-v2":
            closed_loop_config = (
                load_closed_loop_exposure_config(
                    execution_spec.resolve_path("closed_loop_config")
                )
                if execution_spec.closed_loop_config is not None
                else load_closed_loop_exposure_config()
            )
        elif exposure_policy != "neutral-exposure-v1":
            raise ExecutionSpecError(f"Política de exposição desconhecida: {exposure_policy}")
    unique = (
        {cell.cell_id: cell for cell in matrix.cells}
        if exposure_policy == "closed-loop-v2"
        else {cell.comparison_id: cell for cell in matrix.cells}
    )
    for unit_id, cell in sorted(unique.items()):
        path = output / f"{unit_id}.json"
        payload = (
            _closed_loop_unit(
                cell,
                catalog,
                catalog_sha256,
                catalog_path=catalog_path,
                exposure_config=closed_loop_config,
                profile_observation_config=profile_observation_config,
            )
            if closed_loop_config is not None
            else _unit(
                cell,
                catalog,
                catalog_sha256,
                catalog_path=catalog_path,
                exposure_policy=exposure_policy,
                profile_observation_config=profile_observation_config,
            )
        )
        encoded = json.dumps(payload,
                             ensure_ascii=False, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise FileExistsError(f"Unidade divergente já existente: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded, encoding="utf-8")
    return len(unique)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    from cinebot_ml.experiments.freeze import pilot_matrix
    catalog, digest = build_holdout_catalog()
    matrix = pilot_matrix() if args.pilot else load_experiment_matrix(
        experiment_config_path=HOLDOUT_CONFIG)
    print(json.dumps({"catalog_sha256": digest,
                      "units": write_units(matrix, catalog, digest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
