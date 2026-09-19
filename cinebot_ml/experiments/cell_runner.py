"""Adaptador real da matriz para o benchmark B0--B5, sem gerar julgamentos."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.experiment_config import DEFAULT_CONFIG_PATH, load_config, sha256_file
from cinebot_ml.experiments.matrix import ExperimentCell, MatrixValidationError
from cinebot_ml.ranking.evaluation import load_benchmark_units, run_benchmark


def run_cell(cell: ExperimentCell) -> dict[str, Any]:
    """Exige unidade pré-gerada e pareada; não inventa relevância ou estado."""
    units_root = Path(os.environ.get("CINEBOT_CELL_UNITS_DIR", PROJECT_ROOT / "results/units"))
    unit_path = units_root / f"{cell.comparison_id}.json"
    if not unit_path.is_file():
        raise MatrixValidationError(f"Unidade experimental ausente: {unit_path}")
    b2_path = PROJECT_ROOT / "artifacts/b2_tfidf_v1.json"
    b3_model = PROJECT_ROOT / "artifacts/b3_experiment_v1.joblib"
    b3_metadata = PROJECT_ROOT / "artifacts/b3_experiment_v1.json"
    units = load_benchmark_units(
        unit_path, config_path=DEFAULT_CONFIG_PATH,
        b2_artifact_path=b2_path if cell.method == "B2" else None,
        b3_model_path=b3_model if cell.method == "B3" else None,
        b3_metadata_path=b3_metadata if cell.method == "B3" else None,
        methods=(cell.method,),
    )
    if len(units) != 1:
        raise MatrixValidationError("Cada célula exige exatamente uma unidade de avaliação.")
    unit = units[0]
    request = unit.request
    if (request.condition, request.profile, request.seed, request.k) != (
        cell.condition, cell.profile, cell.seed, cell.k
    ) or unit.persona != cell.persona:
        raise MatrixValidationError("Unidade e dimensões da célula divergem.")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    config = load_config(DEFAULT_CONFIG_PATH)
    report = run_benchmark(
        units, k_values=(cell.k,), official_k_values=config["k_values"],
        config_sha256=sha256_file(DEFAULT_CONFIG_PATH), git_commit=commit,
    )
    if len(report.individual) != 1:
        raise MatrixValidationError("Benchmark não produziu um único registro individual.")
    record = report.individual[0]
    if record.method != cell.method or record.ranking_status != "completed":
        raise MatrixValidationError(f"Ranking da célula falhou: {record.error_message}")
    return {"benchmark_id": report.benchmark_id, "benchmark_manifest": report.manifest,
            "individual": record.to_dict()}
