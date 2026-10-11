"""Enumeração, execução sequencial e retomada da matriz experimental."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.experiment_config import DEFAULT_CONFIG_PATH, load_config, sha256_file
from cinebot_ml.experiments.execution import (
    ExecutionSpec,
    ExecutionSpecError,
    load_execution_spec,
    validate_matrix_execution,
    validate_matrix_output,
)
from cinebot_ml.simulation.agents import DEFAULT_AGENT_CONFIG_PATH, load_agent_config


DEFAULT_MATRIX_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/matrix_v1.yaml"
DEFAULT_MATRIX_SCHEMA_PATH = PROJECT_ROOT / "configs/experiments/matrix_v1.schema.json"


class MatrixValidationError(ValueError):
    """Configuração, célula ou checkpoint incompatível com a matriz."""


def _hash(value: Any, length: int = 24) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:length]


@dataclass(frozen=True, order=True)
class ExperimentCell:
    method: str
    condition: str
    profile: str
    persona: str
    seed: int
    k: int
    run: int = 1
    execution_id: str | None = None
    execution_config_sha256: str | None = None
    partition: str | None = None
    policy_scenario_id: str | None = None
    policy_scenario_hash: str | None = None
    policy_selection_id: str | None = None
    policy_selection_sha256: str | None = None

    def _identity(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}

    @property
    def cell_id(self) -> str:
        return _hash(self._identity())

    @property
    def comparison_id(self) -> str:
        payload = self._identity()
        payload.pop("method")
        return _hash(payload)

    @property
    def exposure_id(self) -> str:
        """Identifica a entrada compartilhada, sem método ou cenário B5."""
        payload = self._identity()
        for field in (
            "method",
            "policy_scenario_id",
            "policy_scenario_hash",
            "policy_selection_id",
            "policy_selection_sha256",
        ):
            payload.pop(field, None)
        return _hash(payload)

    def to_dict(self) -> dict[str, Any]:
        return {**self._identity(), "cell_id": self.cell_id, "comparison_id": self.comparison_id}


@dataclass(frozen=True)
class ExperimentMatrix:
    version: str
    protocol_version: str
    config_sha256: str
    agent_config_sha256: str
    matrix_config_sha256: str
    cells: tuple[ExperimentCell, ...]
    excluded: tuple[Mapping[str, Any], ...]
    execution_id: str | None = None
    execution_config_sha256: str | None = None
    partition: str | None = None
    policy_sensitivity_config_sha256: str | None = None
    policy_selection_id: str | None = None
    policy_selection_sha256: str | None = None

    @property
    def matrix_id(self) -> str:
        identity = {
            "version": self.version,
            "protocol_version": self.protocol_version,
            "config_sha256": self.config_sha256,
            "agent_config_sha256": self.agent_config_sha256,
            "matrix_config_sha256": self.matrix_config_sha256,
            "cell_ids": sorted(cell.cell_id for cell in self.cells),
            "excluded": list(self.excluded),
        }
        if self.execution_id is not None:
            identity["execution_id"] = self.execution_id
            identity["execution_config_sha256"] = self.execution_config_sha256
        if self.partition is not None:
            identity["partition"] = self.partition
            identity["policy_sensitivity_config_sha256"] = self.policy_sensitivity_config_sha256
            identity["policy_selection_id"] = self.policy_selection_id
            identity["policy_selection_sha256"] = self.policy_selection_sha256
        return _hash(identity)

    @property
    def total_cells(self) -> int:
        return len(self.cells)

    def to_manifest(self) -> dict[str, Any]:
        manifest = {
            "matrix_id": self.matrix_id,
            "version": self.version,
            "protocol_version": self.protocol_version,
            "config_sha256": self.config_sha256,
            "agent_config_sha256": self.agent_config_sha256,
            "matrix_config_sha256": self.matrix_config_sha256,
            "total_cells": self.total_cells,
            "excluded": list(self.excluded),
            "cells": [cell.to_dict() for cell in self.cells],
        }
        if self.execution_id is not None:
            manifest["execution_id"] = self.execution_id
            manifest["execution_config_sha256"] = self.execution_config_sha256
        if self.partition is not None:
            manifest["partition"] = self.partition
            manifest["policy_sensitivity_config_sha256"] = self.policy_sensitivity_config_sha256
            if self.policy_selection_id is not None:
                manifest["policy_selection_id"] = self.policy_selection_id
                manifest["policy_selection_sha256"] = self.policy_selection_sha256
        return manifest


@dataclass(frozen=True)
class MatrixExecutionReport:
    matrix_id: str
    total_cells: int
    completed: int
    failed: int
    skipped: int
    pending: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CellRunner = Callable[[ExperimentCell], Mapping[str, Any]]


def _load_matrix_config(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_MATRIX_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise MatrixValidationError(f"Não foi possível carregar a matriz: {exc}") from exc
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        raise MatrixValidationError(f"Configuração da matriz inválida: {error.message}")
    digest = sha256_file(path)
    try:
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MatrixValidationError("Lock da matriz ausente ou inválido.") from exc
    if lock.get("sha256") != digest:
        raise MatrixValidationError("A configuração da matriz foi alterada sem novo lock.")
    return dict(payload), digest


def _selected(available: Sequence[Any], requested: Sequence[Any] | None, label: str) -> tuple[Any, ...]:
    values = tuple(sorted(set(available if requested is None else requested)))
    invalid = sorted(set(values) - set(available))
    if invalid:
        raise MatrixValidationError(f"{label} não habilitado: {invalid[0]}")
    if not values:
        raise MatrixValidationError(f"{label} deve possuir ao menos um valor.")
    return values


def _matches(cell: ExperimentCell, rule: Mapping[str, Any]) -> bool:
    return all(getattr(cell, key) in values for key, values in rule.get("when", {}).items())


def load_experiment_matrix(
    *,
    experiment_config_path: Path = DEFAULT_CONFIG_PATH,
    agent_config_path: Path = DEFAULT_AGENT_CONFIG_PATH,
    matrix_config_path: Path = DEFAULT_MATRIX_CONFIG_PATH,
    execution_spec_path: Path | None = None,
    methods: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
    personas: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    k_values: Sequence[int] | None = None,
) -> ExperimentMatrix:
    experiment = load_config(experiment_config_path)
    agents = load_agent_config(agent_config_path)
    matrix_config, matrix_digest = _load_matrix_config(matrix_config_path)
    execution = load_execution_spec(execution_spec_path) if execution_spec_path else None
    dimensions = {
        "methods": _selected(experiment["methods"]["enabled"], methods, "Método"),
        "conditions": _selected(experiment["conditions"], conditions, "Condição"),
        "profiles": _selected(experiment["profiles"], profiles, "Perfil"),
        "personas": _selected(tuple(agents.personas), personas, "Persona"),
        "seeds": _selected(experiment["seeds"], seeds, "Seed"),
        "k_values": _selected(experiment["k_values"], k_values, "K"),
    }
    deterministic = set(matrix_config["deterministic_methods"])
    if not deterministic.issubset(set(experiment["methods"]["enabled"])):
        raise MatrixValidationError("Método determinístico não está habilitado no experimento.")
    rule_dimensions = {
        "method": dimensions["methods"], "condition": dimensions["conditions"],
        "profile": dimensions["profiles"], "persona": dimensions["personas"],
        "seed": dimensions["seeds"], "k": dimensions["k_values"],
    }
    for rule in matrix_config["exclusions"]:
        for field, values in rule["when"].items():
            if field == "run":
                if any(isinstance(value, bool) or value <= 0 for value in values):
                    raise MatrixValidationError("Exclusão contém repetição inválida.")
                continue
            invalid = sorted(set(values) - set(rule_dimensions[field]))
            if invalid:
                raise MatrixValidationError(f"Exclusão referencia {field} inválido: {invalid[0]}")
    cells: list[ExperimentCell] = []
    excluded: list[Mapping[str, Any]] = []
    for method in dimensions["methods"]:
        repetitions = 1 if method in deterministic else int(experiment["execution"]["stochastic_repetitions"])
        for condition in dimensions["conditions"]:
            for profile in dimensions["profiles"]:
                for persona in dimensions["personas"]:
                    for seed in dimensions["seeds"]:
                        for k in dimensions["k_values"]:
                            for run in range(1, repetitions + 1):
                                cell = ExperimentCell(
                                    method, condition, profile, persona, seed, k, run,
                                    execution_id=execution.execution_id if execution else None,
                                    execution_config_sha256=execution.sha256 if execution else None,
                                )
                                rule = next((rule for rule in matrix_config["exclusions"] if _matches(cell, rule)), None)
                                if rule:
                                    excluded.append({"cell": cell.to_dict(), "reason": rule["reason"]})
                                else:
                                    cells.append(cell)
    return ExperimentMatrix(
        str(matrix_config["version"]),
        str(experiment["experiment"]["protocol_version"]),
        sha256_file(experiment_config_path),
        agents.source_sha256,
        matrix_digest,
        tuple(sorted(cells)),
        tuple(sorted(excluded, key=lambda value: value["cell"]["cell_id"])),
        execution_id=execution.execution_id if execution else None,
        execution_config_sha256=execution.sha256 if execution else None,
    )


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def execute_matrix(
    matrix: ExperimentMatrix,
    output: Path,
    runner: CellRunner,
    *,
    execution_spec: ExecutionSpec | None = None,
) -> MatrixExecutionReport:
    if execution_spec is not None:
        validate_matrix_output(matrix, output, execution_spec)
    root = output / matrix.matrix_id
    manifest_path = root / "matrix.manifest.json"
    manifest = matrix.to_manifest()
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MatrixValidationError("Manifesto existente está corrompido.") from exc
        if existing != manifest:
            raise MatrixValidationError("Diretório contém manifesto de uma matriz diferente.")
    else:
        _atomic_json(manifest_path, manifest)
    completed = failed = skipped = 0
    for cell in matrix.cells:
        path = root / "cells" / f"{cell.cell_id}.json"
        if path.exists():
            skipped += 1
            continue
        try:
            payload = runner(cell)
            if not isinstance(payload, Mapping):
                raise MatrixValidationError("CellRunner deve retornar um objeto.")
            record = {"status": "completed", "cell": cell.to_dict(), "result": dict(payload)}
            completed += 1
        except Exception as exc:
            record = {
                "status": "failed", "cell": cell.to_dict(),
                "error_type": type(exc).__name__, "error_message": str(exc),
            }
            failed += 1
        _atomic_json(path, record)
    report = MatrixExecutionReport(
        matrix.matrix_id, matrix.total_cells, completed, failed, skipped,
        matrix.total_cells - completed - failed - skipped,
    )
    _atomic_json(root / "matrix.status.json", report.to_dict())
    return report


def _load_runner(reference: str) -> CellRunner:
    try:
        module_name, function_name = reference.split(":", 1)
        runner = getattr(importlib.import_module(module_name), function_name)
    except (ValueError, ImportError, AttributeError) as exc:
        raise MatrixValidationError("Runner deve usar o formato modulo:funcao.") from exc
    if not callable(runner):
        raise MatrixValidationError("Runner informado não é executável.")
    return runner


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "run"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--agents", type=Path, default=DEFAULT_AGENT_CONFIG_PATH)
    parser.add_argument("--matrix-config", type=Path, default=DEFAULT_MATRIX_CONFIG_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execution-spec", type=Path)
    parser.add_argument("--runner", help="Callable modulo:funcao; obrigatório em run.")
    parser.add_argument("--method", action="append")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--profile", action="append")
    parser.add_argument("--persona", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--k", type=int, action="append")
    args = parser.parse_args(argv)
    try:
        execution = load_execution_spec(args.execution_spec) if args.execution_spec else None
        matrix = load_experiment_matrix(
            experiment_config_path=args.config, agent_config_path=args.agents,
            matrix_config_path=args.matrix_config, methods=args.method,
            conditions=args.condition, profiles=args.profile, personas=args.persona,
            seeds=args.seed, k_values=args.k, execution_spec_path=args.execution_spec,
        )
        if args.command == "list":
            print(json.dumps(matrix.to_manifest(), ensure_ascii=False, indent=2))
            return 0
        if not args.runner:
            raise MatrixValidationError("--runner é obrigatório para executar a matriz.")
        output = args.output or (execution.resolve_path("raw") if execution else PROJECT_ROOT / "results" / "raw" / "matrices")
        print(json.dumps(execute_matrix(matrix, output, _load_runner(args.runner), execution_spec=execution).to_dict()))
        return 0
    except (MatrixValidationError, ValueError) as exc:
        raise SystemExit(f"Erro na matriz experimental: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
