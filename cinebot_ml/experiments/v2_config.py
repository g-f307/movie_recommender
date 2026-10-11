"""Contratos operacionais das matrizes validation e holdout da execução v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.experiments.execution import ExecutionSpec, load_execution_spec
from cinebot_ml.experiments.matrix import ExperimentCell, ExperimentMatrix, MatrixValidationError
from cinebot_ml.experiments.policy_sensitivity import (
    PolicySensitivityConfig,
    PolicySensitivityError,
    load_frozen_validation_selection,
    load_policy_sensitivity_config,
)
from cinebot_ml.simulation.agents import DEFAULT_AGENT_CONFIG_PATH, load_agent_config


DEFAULT_V2_VALIDATION_CONFIG_PATH = PROJECT_ROOT / "configs/experiment_v2_validation.yaml"
DEFAULT_V2_HOLDOUT_CONFIG_PATH = PROJECT_ROOT / "configs/experiment_v2_holdout.yaml"
DEFAULT_V2_VALIDATION_SCHEMA_PATH = PROJECT_ROOT / "configs/experiment_v2_validation.schema.json"
DEFAULT_V2_HOLDOUT_SCHEMA_PATH = PROJECT_ROOT / "configs/experiment_v2_holdout.schema.json"
DEFAULT_V2_MATRIX_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/matrix_v2.yaml"
DEFAULT_V2_MATRIX_SCHEMA_PATH = PROJECT_ROOT / "configs/experiments/matrix_v2.schema.json"
DEFAULT_V2_1_VALIDATION_CONFIG_PATH = PROJECT_ROOT / "configs/experiment_v2_1_validation.yaml"
DEFAULT_V2_1_HOLDOUT_CONFIG_PATH = PROJECT_ROOT / "configs/experiment_v2_1_holdout.yaml"
DEFAULT_V2_1_VALIDATION_SCHEMA_PATH = PROJECT_ROOT / "configs/experiment_v2_1_validation.schema.json"
DEFAULT_V2_1_HOLDOUT_SCHEMA_PATH = PROJECT_ROOT / "configs/experiment_v2_1_holdout.schema.json"
DEFAULT_V2_1_MATRIX_CONFIG_PATH = PROJECT_ROOT / "configs/experiments/matrix_v2_1.yaml"
DEFAULT_V2_1_MATRIX_SCHEMA_PATH = PROJECT_ROOT / "configs/experiments/matrix_v2_1.schema.json"


class V2ExperimentConfigError(ValueError):
    """Contrato operacional da execução v2 inválido ou incompatível."""


@dataclass(frozen=True)
class V2ExperimentPlan:
    """Fontes congeladas necessárias para enumerar uma partição v2."""

    version: str
    protocol_version: str
    execution: ExecutionSpec
    partition: str
    output_root: str
    selection_required: bool
    policy_sensitivity: PolicySensitivityConfig
    config_path: Path
    config_sha256: str
    matrix_config_path: Path
    matrix_config_sha256: str
    matrix_config: Mapping[str, Any]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path, schema_path: Path, label: str) -> tuple[dict[str, Any], str]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise V2ExperimentConfigError(f"Não foi possível carregar {label}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise V2ExperimentConfigError(f"{label.capitalize()} deve ser um objeto.")
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        location = ".".join(str(part) for part in error.absolute_path) or label
        raise V2ExperimentConfigError(f"{label.capitalize()} inválido em {location}: {error.message}")
    digest = _sha256(path)
    try:
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V2ExperimentConfigError(f"Lock de {label} ausente ou inválido.") from exc
    if lock.get("sha256") != digest:
        raise V2ExperimentConfigError(f"{label.capitalize()} diverge do lock.")
    return dict(payload), digest


def _schema_for_partition(partition: str) -> Path:
    if partition == "validation":
        return DEFAULT_V2_VALIDATION_SCHEMA_PATH
    if partition == "holdout":
        return DEFAULT_V2_HOLDOUT_SCHEMA_PATH
    raise V2ExperimentConfigError("A partição v2 deve ser validation ou holdout.")


def _relative_under(child: str, parent: str) -> bool:
    child_path, parent_path = PurePosixPath(child), PurePosixPath(parent)
    return child_path != parent_path and parent_path in child_path.parents


def _selected(available: Sequence[Any], requested: Sequence[Any] | None, label: str) -> tuple[Any, ...]:
    values = tuple(sorted(set(available if requested is None else requested)))
    invalid = sorted(set(values) - set(available))
    if invalid:
        raise V2ExperimentConfigError(f"{label} não habilitado: {invalid[0]}")
    if not values:
        raise V2ExperimentConfigError(f"{label} deve possuir ao menos um valor.")
    return values


def _load_matrix_config(path: Path, schema_path: Path) -> tuple[dict[str, Any], str]:
    return _load_yaml(path.resolve(), schema_path.resolve(), "matriz v2")


def load_v2_experiment_plan(
    path: Path = DEFAULT_V2_VALIDATION_CONFIG_PATH,
    *,
    matrix_config_path: Path = DEFAULT_V2_MATRIX_CONFIG_PATH,
    execution_spec_path: Path | None = None,
    config_schema_path: Path | None = None,
    matrix_schema_path: Path = DEFAULT_V2_MATRIX_SCHEMA_PATH,
    execution_schema_path: Path | None = None,
    policy_sensitivity_schema_path: Path | None = None,
) -> V2ExperimentPlan:
    """Carrega uma partição v2 somente se todos os contratos estiverem congelados."""
    path = path.resolve()
    try:
        preliminary = yaml.safe_load(path.read_text(encoding="utf-8"))
        partition = preliminary["experiment"]["partition"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise V2ExperimentConfigError("Configuração v2 sem partição identificável.") from exc
    payload, digest = _load_yaml(
        path,
        (config_schema_path or _schema_for_partition(partition)).resolve(),
        "configuração v2",
    )
    matrix_config, matrix_digest = _load_matrix_config(matrix_config_path, matrix_schema_path)
    execution = load_execution_spec(
        execution_spec_path or PROJECT_ROOT / "configs/experiments/execution_v2.yaml",
        schema_path=execution_schema_path or PROJECT_ROOT / "configs/experiments/execution_v2.schema.json",
    )
    experiment = payload["experiment"]
    if experiment["execution_id"] != execution.execution_id:
        raise V2ExperimentConfigError("A configuração v2 pertence a outra execução.")
    if experiment["protocol_version"] != execution.protocol_version:
        raise V2ExperimentConfigError("A configuração v2 usa outro protocolo.")
    if experiment["exposure_policy"] != execution.exposure_policy:
        raise V2ExperimentConfigError("A configuração v2 usa outra política de exposição.")
    if experiment["profile_observation_config"] != execution.profile_observation_config:
        raise V2ExperimentConfigError("A configuração v2 usa outro contrato de perfil observado.")
    if experiment["closed_loop_config"] != execution.closed_loop_config:
        raise V2ExperimentConfigError("A configuração v2 usa outro contrato de ciclo fechado.")
    output_root = experiment["output_root"]
    if not _relative_under(output_root, execution.paths["raw"]):
        raise V2ExperimentConfigError("A saída da partição v2 deve ficar abaixo de paths.raw.")
    if matrix_config["execution_id"] != execution.execution_id:
        raise V2ExperimentConfigError("A matriz v2 pertence a outra execução.")
    if matrix_config["exposure_policy"] != execution.exposure_policy:
        raise V2ExperimentConfigError("A matriz v2 usa outra política de exposição.")
    try:
        sensitivity = load_policy_sensitivity_config(
            PROJECT_ROOT / str(experiment["policy_sensitivity_config"]),
            schema_path=(policy_sensitivity_schema_path or PROJECT_ROOT / "configs/experiments/policy_sensitivity_v2.schema.json"),
        )
    except PolicySensitivityError as exc:
        raise V2ExperimentConfigError(str(exc)) from exc
    if sensitivity.execution_id != execution.execution_id:
        raise V2ExperimentConfigError("A matriz de sensibilidade pertence a outra execução.")
    scenario_ids = tuple(scenario.scenario_id for scenario in sensitivity.scenarios)
    if tuple(matrix_config["policy_scenario_ids"]) != scenario_ids:
        raise V2ExperimentConfigError("A matriz v2 não corresponde aos cenários congelados.")
    return V2ExperimentPlan(
        version=str(experiment["version"]),
        protocol_version=str(experiment["protocol_version"]),
        execution=execution,
        partition=str(experiment["partition"]),
        output_root=output_root,
        selection_required=bool(experiment["selection"]["required"]),
        policy_sensitivity=sensitivity,
        config_path=path,
        config_sha256=digest,
        matrix_config_path=matrix_config_path.resolve(),
        matrix_config_sha256=matrix_digest,
        matrix_config=matrix_config,
    )


def build_v2_experiment_matrix(
    plan: V2ExperimentPlan,
    *,
    selection_path: Path | None = None,
    methods: Sequence[str] | None = None,
    conditions: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
    personas: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    k_values: Sequence[int] | None = None,
) -> ExperimentMatrix:
    """Enumera uma matriz v2 com partição e cenário na identidade de cada célula."""
    if plan.partition == "validation" and selection_path is not None:
        raise V2ExperimentConfigError("Validation não aceita referência de seleção congelada.")
    selection: Mapping[str, Any] | None = None
    if plan.selection_required:
        if selection_path is None:
            raise V2ExperimentConfigError("Holdout v2 exige uma seleção congelada compatível.")
        try:
            selection = load_frozen_validation_selection(plan.policy_sensitivity, selection_path)
        except PolicySensitivityError as exc:
            raise V2ExperimentConfigError(f"Seleção congelada incompatível: {exc}") from exc
    elif selection_path is not None:
        raise V2ExperimentConfigError("Esta partição não permite referência de seleção congelada.")

    agents = load_agent_config(DEFAULT_AGENT_CONFIG_PATH)
    configured = plan.matrix_config
    dimensions = {
        "methods": _selected(configured["methods"], methods, "Método"),
        "conditions": _selected(configured["conditions"], conditions, "Condição"),
        "profiles": _selected(configured["profiles"], profiles, "Perfil"),
        "personas": _selected(configured["personas"], personas, "Persona"),
        "seeds": _selected(configured["seeds"], seeds, "Seed"),
        "k_values": _selected(configured["k_values"], k_values, "K"),
    }
    unavailable_personas = sorted(set(dimensions["personas"]) - set(agents.personas))
    if unavailable_personas:
        raise V2ExperimentConfigError(
            f"Persona ausente da configuração de agentes: {unavailable_personas[0]}"
        )
    if set(dimensions["methods"]) != {"B4", "B5"}:
        raise V2ExperimentConfigError("A matriz v2 deve parear exclusivamente B4 e B5.")

    scenarios = tuple(plan.policy_sensitivity.scenarios)
    selection_id = selection_sha256 = None
    if selection is not None:
        selected_id = selection["selected_scenario_id"]
        scenarios = tuple(item for item in scenarios if item.scenario_id == selected_id)
        selection_id = str(selection["selection_id"])
        selection_sha256 = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    if not scenarios:
        raise V2ExperimentConfigError("A seleção congelada não autorizou um cenário v2.")

    cells = []
    for scenario in scenarios:
        for method in dimensions["methods"]:
            for condition in dimensions["conditions"]:
                for profile in dimensions["profiles"]:
                    for persona in dimensions["personas"]:
                        for seed in dimensions["seeds"]:
                            for k in dimensions["k_values"]:
                                cells.append(
                                    ExperimentCell(
                                        method, condition, profile, persona, seed, k,
                                        execution_id=plan.execution.execution_id,
                                        execution_config_sha256=plan.execution.sha256,
                                        partition=plan.partition,
                                        policy_scenario_id=scenario.scenario_id,
                                        policy_scenario_hash=scenario.scenario_hash,
                                        policy_selection_id=selection_id,
                                        policy_selection_sha256=selection_sha256,
                                    )
                                )
    return ExperimentMatrix(
        version=str(configured["version"]),
        protocol_version=plan.protocol_version,
        config_sha256=plan.config_sha256,
        agent_config_sha256=agents.source_sha256,
        matrix_config_sha256=plan.matrix_config_sha256,
        cells=tuple(sorted(cells)),
        excluded=(),
        execution_id=plan.execution.execution_id,
        execution_config_sha256=plan.execution.sha256,
        partition=plan.partition,
        policy_sensitivity_config_sha256=plan.policy_sensitivity.source_sha256,
        policy_selection_id=selection_id,
        policy_selection_sha256=selection_sha256,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Lista uma matriz v2 sem criar células, unidades ou resultados."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list",))
    parser.add_argument("--partition", choices=("validation", "holdout"), default="validation")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--method", action="append")
    parser.add_argument("--condition", action="append")
    parser.add_argument("--profile", action="append")
    parser.add_argument("--persona", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--k", type=int, action="append")
    args = parser.parse_args(argv)
    config_path = (
        DEFAULT_V2_VALIDATION_CONFIG_PATH
        if args.partition == "validation"
        else DEFAULT_V2_HOLDOUT_CONFIG_PATH
    )
    try:
        plan = load_v2_experiment_plan(config_path)
        matrix = build_v2_experiment_matrix(
            plan,
            selection_path=args.selection,
            methods=args.method,
            conditions=args.condition,
            profiles=args.profile,
            personas=args.persona,
            seeds=args.seed,
            k_values=args.k,
        )
    except (V2ExperimentConfigError, MatrixValidationError, ValueError) as exc:
        raise SystemExit(f"Erro na matriz v2: {exc}") from exc
    print(json.dumps(matrix.to_manifest(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
