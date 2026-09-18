"""Estudo longitudinal pareado de convergência C0--C5."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.incremental_benchmark import build_incremental_benchmark_units
from cinebot_ml.ranking import BenchmarkReport, run_benchmark
from cinebot_ml.ranking.discovery_metrics import ranking_overlap_at_k
from cinebot_ml.simulation import SimulationResult, SyntheticAgent


DEFAULT_CONVERGENCE_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/convergence_v1.yaml"
DEFAULT_CONVERGENCE_SCHEMA_PATH = PROJECT_ROOT / "configs/experiments/convergence_v1.schema.json"


class ConvergenceValidationError(ValueError):
    """Trajetórias incompatíveis com uma comparação longitudinal pareada."""


@dataclass(frozen=True)
class ConvergenceConfig:
    version: str
    checkpoints: Mapping[str, int]
    methods: tuple[str, ...]
    metrics: tuple[str, ...]
    source_sha256: str


@dataclass(frozen=True)
class ConvergenceReport:
    study_id: str
    manifest: Mapping[str, Any]
    benchmark: BenchmarkReport
    longitudinal: tuple[Mapping[str, Any], ...]
    curves: tuple[Mapping[str, Any], ...]


def load_convergence_config(path: Path = DEFAULT_CONVERGENCE_CONFIG_PATH) -> ConvergenceConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_CONVERGENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise ConvergenceValidationError(f"Não foi possível carregar convergência: {exc}") from exc
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        raise ConvergenceValidationError(f"Configuração de convergência inválida: {error.message}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConvergenceValidationError("Lock de convergência ausente ou inválido.") from exc
    if lock.get("sha256") != digest:
        raise ConvergenceValidationError("Configuração de convergência alterada sem novo lock.")
    checkpoints = {key: int(value) for key, value in payload["checkpoints"].items()}
    return ConvergenceConfig(
        str(payload["version"]), checkpoints, tuple(payload["methods"]),
        tuple(payload["metrics"]), digest,
    )


def run_convergence_study(
    simulations: Mapping[str, SimulationResult],
    catalog: Sequence[Mapping[str, Any]],
    agent: SyntheticAgent,
    *,
    k: int,
    official_k_values: Sequence[int],
    experiment_config_sha256: str,
    factories=None,
    config: ConvergenceConfig | None = None,
) -> ConvergenceReport:
    config = config or load_convergence_config()
    if set(simulations) != set(config.checkpoints):
        raise ConvergenceValidationError("Trajetórias devem identificar exatamente C0–C5.")
    units = []
    statuses = {}
    preference_ids = set()
    for condition in sorted(config.checkpoints, key=lambda value: config.checkpoints[value]):
        simulation = simulations[condition]
        manifest_agent = simulation.manifest.get("agent", {})
        if manifest_agent.get("agent_id") != agent.agent_id:
            raise ConvergenceValidationError("Trajetórias pertencem a agentes diferentes.")
        preference_ids.add(manifest_agent.get("preference_id"))
        actual = simulation.final_snapshot.state.version
        target = config.checkpoints[condition]
        statuses[condition] = {
            "condition": condition, "target_events": target, "actual_events": actual,
            "sequence_status": simulation.status,
            "profile_state_id": simulation.final_snapshot.state.state_id,
        }
        if simulation.status == "failed" or simulation.final_ranking is None:
            continue
        built = build_incremental_benchmark_units(
            simulation, catalog, agent, methods=config.methods, factories=factories
        )
        if built:
            units.append(built[-1])
    if len(preference_ids) != 1:
        raise ConvergenceValidationError("Preferência latente diverge entre checkpoints.")
    if not units:
        raise ConvergenceValidationError("Nenhum checkpoint avaliável foi produzido.")
    benchmark = run_benchmark(
        units, k_values=(k,), official_k_values=official_k_values,
        config_sha256=experiment_config_sha256,
    )
    longitudinal = _longitudinal_rows(benchmark, statuses, config)
    identity = {
        "version": config.version, "config_sha256": config.source_sha256,
        "agent_id": agent.agent_id, "preference_id": agent.preference_id,
        "seed": agent.seed, "k": k, "benchmark_id": benchmark.benchmark_id,
        "conditions": statuses,
    }
    study_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    curves = tuple({
        key: row.get(key) for key in (
            "agent_id", "method", "condition", "target_events", "actual_events",
            "metric", "value", "gain_vs_c0", "gain_b5_vs_b4", "marginal_gain",
        )
    } for row in longitudinal)
    return ConvergenceReport(study_id, {**identity, "study_id": study_id}, benchmark, longitudinal, curves)


def _longitudinal_rows(benchmark, statuses, config):
    ordered = sorted(config.checkpoints, key=lambda value: config.checkpoints[value])
    records = {(record.method, record.condition): record for record in benchmark.individual}
    rows = []
    previous_rankings = {}
    for condition in ordered:
        for method in config.methods:
            record = records.get((method, condition))
            status = statuses[condition]
            for metric in config.metrics:
                value = record.metrics.get(metric) if record else None
                baseline = records.get((method, "C0"))
                baseline_value = baseline.metrics.get(metric) if baseline else None
                b4 = records.get(("B4", condition))
                b4_value = b4.metrics.get(metric) if b4 else None
                previous_conditions = [item for item in ordered if config.checkpoints[item] < config.checkpoints[condition]]
                previous = records.get((method, previous_conditions[-1])) if previous_conditions else None
                previous_value = previous.metrics.get(metric) if previous else None
                prior_ranking = previous_rankings.get(method)
                stability = (
                    ranking_overlap_at_k(prior_ranking, record.ranked_movie_ids, record.k)
                    if record and prior_ranking is not None else None
                )
                row = {
                    "agent_id": record.agent_id if record else benchmark.manifest["units"][0].get("agent_id"),
                    "method": method, "condition": condition, **status,
                    "coincident_with_previous": bool(previous and record and previous.state_version == record.state_version),
                    "metric": metric, "value": value,
                    "gain_vs_c0": _difference(value, baseline_value),
                    "gain_b5_vs_b4": _difference(value, b4_value) if method == "B5" else None,
                    "marginal_gain": _difference(value, previous_value),
                    "ranking_stability": stability,
                    "profile_stability": (
                        None if not previous_conditions else
                        1.0 if method == "B4" else float(
                            statuses[condition]["profile_state_id"]
                            == statuses[previous_conditions[-1]]["profile_state_id"]
                        )
                    ),
                }
                rows.append(row)
            if record:
                previous_rankings[method] = record.ranked_movie_ids
    return tuple(rows)


def _difference(value, reference):
    return None if value is None or reference is None else float(value) - float(reference)


def write_convergence_report(report: ConvergenceReport, output: Path) -> Mapping[str, Path]:
    root = output / report.study_id
    paths = {
        "manifest": root / "convergence.manifest.json",
        "raw": root / "convergence.longitudinal.csv",
        "curves": root / "convergence.curves.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Resultados de convergência já existem e não serão sobrescritos.")
    root.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(json.dumps(report.manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(paths["raw"], report.longitudinal)
    _write_csv(paths["curves"], report.curves)
    return paths


def _write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8"); return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
