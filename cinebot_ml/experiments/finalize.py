"""Audita, identifica e sela saídas oficiais completas; nunca corrige resultados."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.experiment_config import sha256_file
from cinebot_ml.experiments.freeze import audit_execution, pilot_matrix, preflight
from cinebot_ml.experiments.matrix import load_experiment_matrix
from cinebot_ml.experiments.units import HOLDOUT_CONFIG, UNITS


OFFICIAL_ROOT = PROJECT_ROOT / "results/raw/official_v1_1"
PILOT_ROOT = PROJECT_ROOT / "results/raw/pilot"


def finalize(official_root: Path = OFFICIAL_ROOT, pilot_root: Path = PILOT_ROOT,
             units_root: Path = UNITS, *, seal: bool = True) -> dict:
    matrix = load_experiment_matrix(experiment_config_path=HOLDOUT_CONFIG)
    first_cell = matrix.cells[0]
    first_record = json.loads((official_root / matrix.matrix_id / "cells" /
                               f"{first_cell.cell_id}.json").read_text(encoding="utf-8"))
    commit = first_record["result"]["benchmark_manifest"]["git_commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValueError("Commit gerador ausente ou inválido.")
    official = audit_execution(matrix, official_root, expected_commit=commit)
    pilot = audit_execution(pilot_matrix(), pilot_root)
    if not official["complete"] or not pilot["complete"]:
        raise ValueError("Matriz oficial ou piloto incompleto; congelamento recusado.")
    if official_root.resolve() == pilot_root.resolve():
        raise ValueError("Saídas piloto e oficiais não podem compartilhar diretório.")
    readiness = preflight()
    if readiness["blockers"]:
        raise ValueError(f"Pré-voo possui bloqueios: {readiness['blockers']}")
    root = official_root / matrix.matrix_id
    target = root / "experiment.manifest.json"
    if target.exists():
        raise FileExistsError("Experimento oficial já possui manifesto; não será sobrescrito.")
    required = {
        "catalog": PROJECT_ROOT / "results/derived/test_catalog.json",
        "dataset": PROJECT_ROOT / "datasets/movie_preferences.csv",
        "feedback": PROJECT_ROOT / "datasets/user_feedback.csv",
        "split": PROJECT_ROOT / "results/manifests/splits/movie_id_split.json",
        "b2": PROJECT_ROOT / "artifacts/b2_tfidf_v1.json",
        "b3_model": PROJECT_ROOT / "artifacts/b3_experiment_v1.joblib",
        "b3_metadata": PROJECT_ROOT / "artifacts/b3_experiment_v1.json",
    }
    for label, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Insumo ausente: {label}")
    config_hashes = {path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path)
                     for path in sorted((PROJECT_ROOT / "configs").rglob("*.yaml"))}
    unit_hashes = {path.stem: sha256_file(path) for path in sorted(units_root.glob("*.json"))}
    expected_units = {cell.comparison_id for cell in matrix.cells}
    if set(unit_hashes) != expected_units:
        raise ValueError("Conjunto de unidades não cobre exatamente a matriz oficial.")
    cell_hashes = {cell.cell_id: sha256_file(root / "cells" / f"{cell.cell_id}.json")
                   for cell in matrix.cells}
    payload = {
        "schema_version": "1.0", "status": "complete", "evidence": "exploratory_synthetic",
        "review_status": "advisor_unavailable_not_approved",
        "b6": "deferred_no_distinct_method", "git_commit": commit,
        "matrix_id": matrix.matrix_id, "total_cells": matrix.total_cells,
        "dimensions": {field: sorted({getattr(cell, field) for cell in matrix.cells})
                       for field in ("method", "condition", "profile", "persona", "seed", "k", "run")},
        "pilot": {"matrix_id": pilot["matrix_id"], "completed": pilot["completed"],
                  "cell_bytes": pilot["cell_bytes"]},
        "official": {"completed": official["completed"], "cell_bytes": official["cell_bytes"]},
        "matrix_manifest_sha256": sha256_file(root / "matrix.manifest.json"),
        "inputs_sha256": {label: sha256_file(path) for label, path in required.items()},
        "configs_sha256": config_hashes, "units_sha256": unit_hashes,
        "cells_sha256": cell_hashes,
        "python_version": sys.version.split()[0],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    os.replace(temporary, target)
    if seal:
        for path in root.rglob("*.json"):
            path.chmod(0o444)
        (root / "cells").chmod(0o555)
        root.chmod(0o555)
        for label in ("split", "catalog", "b2", "b3_model", "b3_metadata"):
            required[label].chmod(0o444)
        for path in units_root.glob("*.json"):
            path.chmod(0o444)
        units_root.chmod(0o555)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-seal", action="store_true")
    args = parser.parse_args()
    result = finalize(seal=not args.no_seal)
    print(json.dumps({"matrix_id": result["matrix_id"], "total_cells": result["total_cells"],
                      "git_commit": result["git_commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
