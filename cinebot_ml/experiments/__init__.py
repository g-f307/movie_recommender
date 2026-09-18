"""Orquestração versionada dos experimentos científicos."""

from cinebot_ml.experiments.matrix import (
    ExperimentCell,
    ExperimentMatrix,
    MatrixExecutionReport,
    MatrixValidationError,
    execute_matrix,
    load_experiment_matrix,
)
from cinebot_ml.experiments.cold_start import (
    ColdStartProfile,
    ColdStartReport,
    ColdStartValidationError,
    build_cold_start_profiles,
    load_cold_start_config,
    profile_payload,
    run_cold_start_study,
    validate_profile_payload,
    write_cold_start_report,
)
from cinebot_ml.experiments.convergence import (
    ConvergenceReport, ConvergenceValidationError, load_convergence_config,
    run_convergence_study, write_convergence_report,
)

__all__ = [
    "ExperimentCell",
    "ExperimentMatrix",
    "MatrixExecutionReport",
    "MatrixValidationError",
    "execute_matrix",
    "load_experiment_matrix",
    "ColdStartProfile",
    "ColdStartReport",
    "ColdStartValidationError",
    "build_cold_start_profiles",
    "load_cold_start_config",
    "profile_payload",
    "run_cold_start_study",
    "validate_profile_payload",
    "write_cold_start_report",
    "ConvergenceReport", "ConvergenceValidationError", "load_convergence_config",
    "run_convergence_study", "write_convergence_report",
]
