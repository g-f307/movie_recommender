"""Orquestra validation → seleção congelada → holdout da execução v2."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from cinebot_ml.experiment_config import sha256_file
from cinebot_ml.experiments.freeze import audit_execution
from cinebot_ml.experiments.matrix import ExperimentCell, ExperimentMatrix, MatrixExecutionReport, execute_matrix
from cinebot_ml.experiments.policy_sensitivity import (
    PolicySensitivityError,
    freeze_validation_selection,
    select_validation_scenario,
    write_validation_observations,
)
from cinebot_ml.experiments.v2_config import (
    DEFAULT_V2_HOLDOUT_CONFIG_PATH,
    DEFAULT_V2_VALIDATION_CONFIG_PATH,
    V2ExperimentConfigError,
    V2ExperimentPlan,
    build_v2_experiment_matrix,
    load_v2_experiment_plan,
)
from cinebot_ml.experiments.v2_cell_runner import build_v2_runner, prepare_v2_units


class V2OrchestrationError(ValueError):
    """Ordem, identidade ou integridade inválida na execução complementar."""


CellRunner = Callable[[ExperimentCell], Mapping[str, Any]]


def _output(plan: V2ExperimentPlan) -> Path:
    output = plan.execution.resolve_path("raw") / plan.partition
    if output.as_posix().endswith(plan.output_root):
        return output
    raise V2OrchestrationError("A configuração da partição diverge de paths.raw da execução.")


def _selection_dir(plan: V2ExperimentPlan, matrix: ExperimentMatrix) -> Path:
    return plan.execution.resolve_path("reports") / "validation" / matrix.matrix_id


def _selection_comparison_id(cell: ExperimentCell) -> str:
    payload = cell._identity()
    for field in ("method", "policy_scenario_id", "policy_scenario_hash"):
        payload.pop(field, None)
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:24]


def _records(matrix: ExperimentMatrix, output: Path) -> list[dict[str, Any]]:
    root = output / matrix.matrix_id / "cells"
    records = []
    for cell in matrix.cells:
        path = root / f"{cell.cell_id}.json"
        if not path.is_file():
            raise V2OrchestrationError(f"Checkpoint v2 ausente: {cell.cell_id}")
        try:
            records.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            raise V2OrchestrationError(f"Checkpoint v2 inválido: {cell.cell_id}") from exc
    return records


def execute_v2_validation(
    plan: V2ExperimentPlan, runner: CellRunner, **filters: Sequence[Any]
) -> tuple[ExperimentMatrix, MatrixExecutionReport]:
    if plan.partition != "validation":
        raise V2OrchestrationError("O comando validation exige o plano de validation.")
    matrix = build_v2_experiment_matrix(plan, **filters)
    output = _output(plan)
    return matrix, execute_matrix(matrix, output, runner, execution_spec=plan.execution)


def freeze_v2_selection(plan: V2ExperimentPlan, matrix: ExperimentMatrix) -> Path:
    if plan.partition != "validation" or matrix.partition != "validation":
        raise V2OrchestrationError("A seleção só pode ser congelada a partir da validation.")
    output = _output(plan)
    audit = audit_execution(matrix, output, execution_spec=plan.execution)
    rows: list[dict[str, Any]] = []
    for record in _records(matrix, output):
        cell = record.get("cell", {})
        if cell.get("method") != "B5":
            continue
        individual = record.get("result", {}).get("individual", {})
        status = "evaluated" if record.get("status") == "completed" and individual.get("evaluation_status") == "evaluated" else "failed"
        rows.append({
            "scenario_id": cell.get("policy_scenario_id"),
            "partition": "validation",
            "comparison_id": _selection_comparison_id(ExperimentCell(**{
                key: value for key, value in cell.items() if key != "cell_id" and key != "comparison_id"
            })),
            "agent_id": individual.get("agent_id"),
            "seed": cell.get("seed"),
            "candidate_set_id": individual.get("candidate_set_id"),
            "ndcg_at_5": individual.get("metrics", {}).get("ndcg_at_k"),
            "status": status,
            "cell_id": cell.get("cell_id"),
        })
    report_dir = _selection_dir(plan, matrix)
    observations = report_dir / "validation_observations.jsonl"
    write_validation_observations(rows, observations)
    if not audit["complete"] or any(row["status"] != "evaluated" for row in rows):
        raise V2OrchestrationError("Validation incompleta; a seleção congelada foi bloqueada.")
    try:
        selection = select_validation_scenario(plan.policy_sensitivity, rows)
    except PolicySensitivityError as exc:
        raise V2OrchestrationError(f"Validation incompatível para seleção: {exc}") from exc
    selection_path = report_dir / "selection.json"
    freeze_validation_selection(selection, selection_path)
    return selection_path


def execute_v2_holdout(
    plan: V2ExperimentPlan, runner: CellRunner, *, selection_path: Path | None = None,
    **filters: Sequence[Any],
) -> tuple[ExperimentMatrix, MatrixExecutionReport]:
    if plan.partition != "holdout":
        raise V2OrchestrationError("O comando holdout exige o plano de holdout.")
    if selection_path is None:
        raise V2OrchestrationError("O holdout exige uma seleção congelada compatível.")
    try:
        matrix = build_v2_experiment_matrix(plan, selection_path=selection_path, **filters)
    except V2ExperimentConfigError as exc:
        raise V2OrchestrationError(str(exc)) from exc
    output = _output(plan)
    return matrix, execute_matrix(matrix, output, runner, execution_spec=plan.execution)


def finalize_v2_execution(
    validation_plan: V2ExperimentPlan, validation: ExperimentMatrix,
    holdout_plan: V2ExperimentPlan, holdout: ExperimentMatrix, selection_path: Path,
) -> Path:
    """Audita ambas as partições e cria uma cópia selada consumível pela análise."""
    if validation_plan.execution.sha256 != holdout_plan.execution.sha256:
        raise V2OrchestrationError("Validation e holdout pertencem a execuções distintas.")
    validation_output, holdout_output = _output(validation_plan), _output(holdout_plan)
    audits = {
        "validation": audit_execution(validation, validation_output, execution_spec=validation_plan.execution),
        "holdout": audit_execution(holdout, holdout_output, execution_spec=holdout_plan.execution),
    }
    if not all(report["complete"] for report in audits.values()):
        raise V2OrchestrationError("A auditoria v2 encontrou células incompletas ou incompatíveis.")
    try:
        selection_sha = sha256_file(selection_path)
    except OSError as exc:
        raise V2OrchestrationError("Seleção congelada indisponível para finalização.") from exc
    if holdout.policy_selection_sha256 != selection_sha:
        raise V2OrchestrationError("A matriz de holdout não referencia a seleção congelada.")

    cells: list[dict[str, Any]] = []
    sources: list[tuple[str, Path, ExperimentMatrix]] = [
        ("validation", validation_output, validation), ("holdout", holdout_output, holdout),
    ]
    for _, output, matrix in sources:
        cells.extend(matrix.to_manifest()["cells"])
    combined_id = hashlib.sha256(
        json.dumps(sorted(cell["cell_id"] for cell in cells), separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    root = validation_plan.execution.resolve_path("manifests") / combined_id
    target = root / "execution.manifest.json"
    if target.exists():
        raise FileExistsError("A execução v2 já possui manifesto; não será sobrescrito.")
    root.mkdir(parents=True, exist_ok=False)
    matrix_payload = {
        "matrix_id": combined_id, "total_cells": len(cells), "cells": cells,
        "execution_id": validation_plan.execution.execution_id,
        "execution_config_sha256": validation_plan.execution.sha256,
    }
    matrix_path = root / "matrix.manifest.json"
    matrix_path.write_text(json.dumps(matrix_payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    hashes = {}
    cells_dir = root / "cells"
    cells_dir.mkdir()
    for partition, output, matrix in sources:
        for cell in matrix.cells:
            source = output / matrix.matrix_id / "cells" / f"{cell.cell_id}.json"
            destination = cells_dir / f"{cell.cell_id}.json"
            shutil.copyfile(source, destination)
            hashes[cell.cell_id] = sha256_file(destination)
    manifest = {
        "schema_version": "1.0", "status": "complete",
        "execution": {
            "id": validation_plan.execution.execution_id,
            "version": validation_plan.execution.version,
            "protocol_version": validation_plan.execution.protocol_version,
            "config_sha256": validation_plan.execution.sha256,
            "exposure_policy": validation_plan.execution.exposure_policy,
        },
        "matrix": {"matrix_id": combined_id, "total_cells": len(cells), "manifest_sha256": sha256_file(matrix_path)},
        "partitions": {
            partition: {"matrix_id": matrix.matrix_id, "audit": audits[partition], "matrix_manifest_sha256": sha256_file(output / matrix.matrix_id / "matrix.manifest.json")}
            for partition, output, matrix in sources
        },
        "selection": {"path": str(selection_path), "sha256": selection_sha},
        "inputs_sha256": {key: sha256_file(validation_plan.execution.resolve_path(key)) for key in ("source_catalog", "split_manifest", "catalog")},
        "cells_sha256": hashes,
    }
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validation", "freeze-selection", "holdout", "audit", "finalize"))
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--condition", action="append")
    parser.add_argument("--profile", action="append")
    parser.add_argument("--persona", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--k", type=int, action="append")
    args = parser.parse_args(argv)
    filters = {
        "conditions": args.condition,
        "profiles": args.profile,
        "personas": args.persona,
        "seeds": args.seed,
        "k_values": args.k,
    }
    validation = load_v2_experiment_plan(DEFAULT_V2_VALIDATION_CONFIG_PATH)
    try:
        if args.command == "validation":
            matrix = build_v2_experiment_matrix(validation, **filters)
            prepare_v2_units(matrix, validation.execution)
            _, report = execute_v2_validation(validation, build_v2_runner(validation.execution), **filters)
            print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
            return 0 if report.failed == 0 else 2

        validation_matrix = build_v2_experiment_matrix(validation, **filters)
        if args.command == "freeze-selection":
            print(json.dumps({"selection": str(freeze_v2_selection(validation, validation_matrix))}, ensure_ascii=False))
            return 0

        if args.selection is None:
            raise V2OrchestrationError("--selection é obrigatório após a validation congelada.")
        holdout = load_v2_experiment_plan(DEFAULT_V2_HOLDOUT_CONFIG_PATH)
        holdout_matrix = build_v2_experiment_matrix(holdout, selection_path=args.selection, **filters)
        if args.command == "holdout":
            prepare_v2_units(holdout_matrix, holdout.execution)
            _, report = execute_v2_holdout(
                holdout, build_v2_runner(holdout.execution), selection_path=args.selection, **filters
            )
            print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True))
            return 0 if report.failed == 0 else 2
        if args.command == "audit":
            report = {
                "validation": audit_execution(validation_matrix, _output(validation), execution_spec=validation.execution),
                "holdout": audit_execution(holdout_matrix, _output(holdout), execution_spec=holdout.execution),
            }
            print(json.dumps(report, ensure_ascii=False, sort_keys=True))
            return 0 if all(value["complete"] for value in report.values()) else 2
        print(json.dumps({"manifest": str(finalize_v2_execution(
            validation, validation_matrix, holdout, holdout_matrix, args.selection,
        ))}, ensure_ascii=False))
        return 0
    except (V2ExperimentConfigError, V2OrchestrationError, OSError, ValueError) as exc:
        raise SystemExit(f"Erro na orquestração v2: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
