"""Orquestração versionada dos experimentos científicos."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ExperimentCell": ("matrix", "ExperimentCell"),
    "ExperimentMatrix": ("matrix", "ExperimentMatrix"),
    "MatrixExecutionReport": ("matrix", "MatrixExecutionReport"),
    "MatrixValidationError": ("matrix", "MatrixValidationError"),
    "execute_matrix": ("matrix", "execute_matrix"),
    "load_experiment_matrix": ("matrix", "load_experiment_matrix"),
    "V2ExperimentConfigError": ("v2_config", "V2ExperimentConfigError"),
    "V2ExperimentPlan": ("v2_config", "V2ExperimentPlan"),
    "build_v2_experiment_matrix": ("v2_config", "build_v2_experiment_matrix"),
    "load_v2_experiment_plan": ("v2_config", "load_v2_experiment_plan"),
    "build_v2_runner": ("v2_cell_runner", "build_v2_runner"),
    "prepare_v2_units": ("v2_cell_runner", "prepare_v2_units"),
    "run_v2_cell": ("v2_cell_runner", "run_v2_cell"),
    "ColdStartProfile": ("cold_start", "ColdStartProfile"),
    "ColdStartReport": ("cold_start", "ColdStartReport"),
    "ColdStartValidationError": ("cold_start", "ColdStartValidationError"),
    "build_cold_start_profiles": ("cold_start", "build_cold_start_profiles"),
    "load_cold_start_config": ("cold_start", "load_cold_start_config"),
    "profile_payload": ("cold_start", "profile_payload"),
    "run_cold_start_study": ("cold_start", "run_cold_start_study"),
    "validate_profile_payload": ("cold_start", "validate_profile_payload"),
    "write_cold_start_report": ("cold_start", "write_cold_start_report"),
    "ConvergenceReport": ("convergence", "ConvergenceReport"),
    "ConvergenceValidationError": ("convergence", "ConvergenceValidationError"),
    "load_convergence_config": ("convergence", "load_convergence_config"),
    "run_convergence_study": ("convergence", "run_convergence_study"),
    "write_convergence_report": ("convergence", "write_convergence_report"),
    "AblationContext": ("ablation", "AblationContext"),
    "AblationReport": ("ablation", "AblationReport"),
    "AblationValidationError": ("ablation", "AblationValidationError"),
    "apply_ablation": ("ablation", "apply_ablation"),
    "execute_ablation": ("ablation", "execute_ablation"),
    "load_ablation_variants": ("ablation", "load_ablation_variants"),
    "RobustnessCase": ("robustness", "RobustnessCase"),
    "Scenario": ("robustness", "Scenario"),
    "load_robustness_config": ("robustness", "load_robustness_config"),
    "make_case": ("robustness", "make_case"),
    "run_robustness": ("robustness", "run_robustness"),
    "write_robustness_report": ("robustness", "write_robustness_report"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Preserva a API pública sem pré-carregar submódulos executáveis."""
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
