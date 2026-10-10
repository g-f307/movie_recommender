"""Audita, identifica e sela saídas oficiais completas; nunca corrige resultados."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.experiment_config import sha256_file
from cinebot_ml.experiments.execution import ExecutionSpec, load_execution_spec, validate_matrix_execution
from cinebot_ml.experiments.freeze import audit_execution, pilot_matrix, preflight
from cinebot_ml.experiments.matrix import ExperimentMatrix, load_experiment_matrix
from cinebot_ml.experiments.units import HOLDOUT_CONFIG, UNITS


OFFICIAL_ROOT = PROJECT_ROOT / "results/raw/official_v1_1"
PILOT_ROOT = PROJECT_ROOT / "results/raw/pilot"


def finalize_versioned_execution(
    matrix: ExperimentMatrix,
    execution_spec: ExecutionSpec,
    output: Path,
    *,
    project_root: Path = PROJECT_ROOT,
    seal: bool = False,
) -> dict:
    """Sela uma execução versionada sem tocar nos artefatos oficiais v1.1.

    A função é deliberadamente agnóstica ao método: as issues posteriores
    adicionam os insumos específicos do estudo v2. Nesta camada, ela garante
    identidade, integridade dos checkpoints, hashes dos insumos declarados e
    ausência de sobrescrita do manifesto final.
    """
    validate_matrix_execution(matrix, execution_spec)
    expected_output = execution_spec.resolve_path("raw", project_root)
    if output.resolve() != expected_output:
        raise ValueError("A finalização v2 só aceita o diretório raw declarado na execução.")
    audit = audit_execution(matrix, output, execution_spec=execution_spec)
    if not audit["complete"]:
        raise ValueError("Execução v2 incompleta ou com artefatos de outra versão; finalização recusada.")

    root = output / matrix.matrix_id
    target = root / "execution.manifest.json"
    if target.exists():
        raise FileExistsError("A execução já possui manifesto; não será sobrescrito.")
    input_hashes: dict[str, str] = {}
    for key in ("source_catalog", "split_manifest", "catalog"):
        path = execution_spec.resolve_path(key, project_root)
        if not path.is_file():
            raise FileNotFoundError(f"Insumo declarado ausente na execução v2: {key}")
        input_hashes[key] = sha256_file(path)
    payload = {
        "schema_version": "1.0",
        "status": "complete",
        "execution": {
            "id": execution_spec.execution_id,
            "version": execution_spec.version,
            "protocol_version": execution_spec.protocol_version,
            "config_sha256": execution_spec.sha256,
            "exposure_policy": execution_spec.exposure_policy,
        },
        "matrix": {
            "matrix_id": matrix.matrix_id,
            "total_cells": matrix.total_cells,
            "manifest_sha256": sha256_file(root / "matrix.manifest.json"),
        },
        "inputs_sha256": input_hashes,
        "cells_sha256": {
            cell.cell_id: sha256_file(root / "cells" / f"{cell.cell_id}.json")
            for cell in matrix.cells
        },
    }
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)
    if seal:
        for path in root.rglob("*.json"):
            path.chmod(0o444)
        (root / "cells").chmod(0o555)
        root.chmod(0o555)
    return payload


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
    parser.add_argument("--execution-spec", type=Path)
    parser.add_argument("--config", type=Path, default=HOLDOUT_CONFIG)
    parser.add_argument("--agents", type=Path)
    parser.add_argument("--matrix-config", type=Path)
    args = parser.parse_args()
    if args.execution_spec:
        execution = load_execution_spec(args.execution_spec)
        matrix = load_experiment_matrix(
            experiment_config_path=args.config,
            agent_config_path=args.agents or PROJECT_ROOT / "configs/agents/personas_v1.yaml",
            matrix_config_path=args.matrix_config or PROJECT_ROOT / "configs/experiments/matrix_v1.yaml",
            execution_spec_path=args.execution_spec,
        )
        result = finalize_versioned_execution(
            matrix, execution, execution.resolve_path("raw"), seal=not args.no_seal,
        )
        print(json.dumps({"matrix_id": result["matrix"]["matrix_id"], "execution_id": execution.execution_id}))
        return 0
    result = finalize(seal=not args.no_seal)
    print(json.dumps({"matrix_id": result["matrix_id"], "total_cells": result["total_cells"],
                      "git_commit": result["git_commit"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
