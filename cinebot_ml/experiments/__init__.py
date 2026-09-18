"""Orquestração versionada dos experimentos científicos."""

from cinebot_ml.experiments.matrix import (
    ExperimentCell,
    ExperimentMatrix,
    MatrixExecutionReport,
    MatrixValidationError,
    execute_matrix,
    load_experiment_matrix,
)

__all__ = [
    "ExperimentCell",
    "ExperimentMatrix",
    "MatrixExecutionReport",
    "MatrixValidationError",
    "execute_matrix",
    "load_experiment_matrix",
]
