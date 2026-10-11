"""Runner isolado para células B4/B5 da execução complementar v2."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.dataset import load_catalog
from cinebot_ml.experiment_config import sha256_file
from cinebot_ml.experiments.execution import ExecutionSpec, ExecutionSpecError
from cinebot_ml.experiments.matrix import ExperimentCell, MatrixValidationError
from cinebot_ml.experiments.policy_sensitivity import build_sensitivity_runtime, load_policy_sensitivity_config
from cinebot_ml.personalization import StateSnapshot
from cinebot_ml.ranking.candidates import build_candidate_set
from cinebot_ml.ranking.evaluation import BenchmarkUnit, RelevanceJudgments, run_benchmark
from cinebot_ml.ranking.incremental import IncrementalRecommender
from cinebot_ml.ranking.static_personalized import StaticPersonalizedRecommender
from cinebot_ml.ranking.contracts import RecommendationRequest
from cinebot_ml.experiments.units import build_holdout_catalog_for_execution, write_units
from cinebot_ml.experiments.matrix import ExperimentMatrix


def _validate_cell(cell: ExperimentCell, execution_spec: ExecutionSpec) -> None:
    if cell.method not in {"B4", "B5"}:
        raise MatrixValidationError("O runner v2 aceita somente B4 e B5.")
    if cell.partition not in {"validation", "holdout"}:
        raise MatrixValidationError("A célula deve declarar uma partição v2.")
    if not cell.policy_scenario_id or not cell.policy_scenario_hash:
        raise MatrixValidationError("A célula v2 deve declarar o cenário de política.")
    if cell.execution_id != execution_spec.execution_id:
        raise ExecutionSpecError("A célula não pertence à execução v2 declarada.")
    if cell.execution_config_sha256 != execution_spec.sha256:
        raise ExecutionSpecError("O hash da célula diverge da execução v2 declarada.")


def _load_unit(cell: ExperimentCell, units_root: Path) -> tuple[dict[str, Any], Mapping[str, Any]]:
    path = units_root / f"{cell.cell_id}.json"
    if not path.is_file():
        raise MatrixValidationError(f"Unidade v2 ausente: {path}")
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        units = payload["units"]
        unit = units[0]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
        raise MatrixValidationError("Unidade v2 inválida ou incompleta.") from exc
    if len(units) != 1 or payload.get("exposure_policy") != "closed-loop-v2":
        raise MatrixValidationError("A unidade v2 deve conter uma única trajetória em ciclo fechado.")
    if payload.get("trajectory_audit", {}).get("complete") is not True:
        raise MatrixValidationError("A unidade v2 não possui auditoria causal aprovada.")
    pairing = payload.get("pairing", {})
    if not pairing.get("pairing_id") or pairing.get("method") != cell.method:
        raise MatrixValidationError("A unidade v2 não possui pareamento compatível.")
    return payload, unit


def _commit() -> str:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        capture_output=True, check=False,
    )
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def run_v2_cell(
    cell: ExperimentCell,
    *,
    execution_spec: ExecutionSpec,
    units_root: Path | None = None,
    catalog_path: Path | None = None,
) -> dict[str, Any]:
    """Avalia uma unidade v2 pré-gerada sem consultar contratos ou saídas v1.1."""
    _validate_cell(cell, execution_spec)
    units_root = (units_root or execution_spec.resolve_path("units")).resolve()
    payload, raw_unit = _load_unit(cell, units_root)
    try:
        request = RecommendationRequest.from_dict(raw_unit["request"])
        initial = StateSnapshot.from_dict(raw_unit["initial_snapshot"])
        state = StateSnapshot.from_dict(raw_unit["state_snapshot"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MatrixValidationError("A unidade v2 não possui snapshots ou request válidos.") from exc
    if (request.condition, request.profile, request.seed, request.k) != (
        cell.condition, cell.profile, cell.seed, cell.k
    ) or raw_unit.get("persona") != cell.persona:
        raise MatrixValidationError("Unidade e dimensões da célula v2 divergem.")
    if initial.state.subject_id != state.state.subject_id or raw_unit.get("agent_id") != state.state.subject_id:
        raise MatrixValidationError("A identidade do agente v2 diverge entre snapshots e unidade.")
    if tuple(event.to_dict() for event in state.state.history) != request.history:
        raise MatrixValidationError("O ranking v2 recebeu feedback futuro ou histórico divergente.")
    source = (catalog_path or execution_spec.resolve_path("catalog")).resolve()
    if source == PROJECT_ROOT / "results" / "derived" / "test_catalog.json":
        raise MatrixValidationError("O runner v2 não pode reutilizar o catálogo oficial v1.1.")
    if not source.is_file():
        raise MatrixValidationError(f"Catálogo v2 ausente: {source}")
    catalog = load_catalog(source)
    candidates = build_candidate_set(
        catalog, request, catalog_path=execution_spec.paths["catalog"],
        catalog_sha256=sha256_file(source),
    )
    if request.candidate_movie_ids and tuple(request.candidate_movie_ids) != candidates.movie_ids:
        raise MatrixValidationError("Candidatos reconstruídos divergem da unidade v2.")
    values = {item["movie_id"]: item["value"] for item in raw_unit.get("relevance", [])}
    relevance = RelevanceJudgments(
        str(payload.get("relevance_source")), values, bool(raw_unit.get("complete")),
        str(payload.get("relevance_version")),
    )
    if cell.method == "B4":
        recommender = StaticPersonalizedRecommender(candidates, initial.state)
    else:
        sensitivity = load_policy_sensitivity_config()
        scenario = next(
            (item for item in sensitivity.scenarios if item.scenario_id == cell.policy_scenario_id),
            None,
        )
        if scenario is None or scenario.scenario_hash != cell.policy_scenario_hash:
            raise MatrixValidationError("O cenário B5 da célula não pertence à matriz congelada.")
        recommender = IncrementalRecommender(candidates, state, config=build_sensitivity_runtime(sensitivity, scenario).incremental_config)
    benchmark = BenchmarkUnit(
        request, candidates, relevance, {cell.method: recommender},
        agent_id=raw_unit.get("agent_id"), persona=raw_unit.get("persona"),
        agent_version=raw_unit.get("agent_version"), interaction=int(raw_unit.get("interaction", 0)),
        state_version=int(raw_unit.get("state_version", state.state.version)),
        simulation_id=raw_unit.get("simulation_id"),
    )
    report = run_benchmark(
        (benchmark,), k_values=(cell.k,), official_k_values=(cell.k,),
        config_sha256=execution_spec.sha256, git_commit=_commit(),
    )
    if len(report.individual) != 1:
        raise MatrixValidationError("O runner v2 não produziu uma avaliação individual.")
    individual = report.individual[0]
    individual_payload = individual.to_dict()
    individual_payload.update({
        "partition": cell.partition,
        "policy_scenario_id": cell.policy_scenario_id,
        "policy_scenario_hash": cell.policy_scenario_hash,
        "profile_noise_level": "configured-v2",
        "exposure_policy": payload["exposure_policy"],
        "trajectory_status": "completed",
    })
    return {
        "benchmark_id": report.benchmark_id,
        "benchmark_manifest": report.manifest,
        "individual": individual_payload,
        "execution_id": cell.execution_id,
        "execution_config_sha256": cell.execution_config_sha256,
        "partition": cell.partition,
        "policy_scenario_id": cell.policy_scenario_id,
        "policy_scenario_hash": cell.policy_scenario_hash,
        "profile_observation": payload.get("profile_observation"),
        "exposure_policy": payload["exposure_policy"],
        "exposure_manifest": payload.get("exposure_manifest"),
        "trajectory_audit": payload["trajectory_audit"],
        "pairing": payload["pairing"],
        "candidate_set_id": candidates.candidate_set_id,
        "relevance_id": relevance.relevance_id,
    }


def prepare_v2_units(matrix: ExperimentMatrix, execution_spec: ExecutionSpec) -> int:
    """Materializa catálogo isolado e uma unidade v2 por célula antes do runner."""
    if matrix.execution_id != execution_spec.execution_id:
        raise ExecutionSpecError("A matriz não pertence à execução v2 declarada.")
    catalog, digest = build_holdout_catalog_for_execution(execution_spec)
    return write_units(
        matrix, catalog, digest, execution_spec.resolve_path("units"),
        execution_spec=execution_spec,
    )


def build_v2_runner(execution_spec: ExecutionSpec) -> Callable[[ExperimentCell], dict[str, Any]]:
    """Retorna um runner sem caminhos globais, ambiente ou artefatos da v1.1."""
    return lambda cell: run_v2_cell(cell, execution_spec=execution_spec)
