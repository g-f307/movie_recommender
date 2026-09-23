"""Figuras científicas estáticas para a análise exploratória de subgrupos."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence


CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "C5")
PROFILES = ("P0", "P1", "P2", "P3", "P4", "P5")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _heatmap_values(details: Sequence[dict[str, Any]], k: int) -> list[list[float]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in details:
        if row["k"] == k:
            grouped[(row["condition"], row["profile"])].append(
                row["difference_b5_minus_b4"]
            )
    return [
        [_mean(grouped[(condition, profile)]) for profile in PROFILES]
        for condition in CONDITIONS
    ]


def _agent_distributions(
    details: Sequence[dict[str, Any]], dimension: str, k: int
) -> tuple[list[str], list[list[float]]]:
    eligible = [
        row for row in details
        if row["k"] == k and (dimension == "condition" or row["condition"] != "C0")
    ]
    categories = sorted(
        {str(row[dimension]) for row in eligible},
        key=(lambda value: int(value)) if dimension == "seed" else None,
    )
    values = []
    for category in categories:
        agents: dict[str, list[float]] = defaultdict(list)
        for row in eligible:
            if str(row[dimension]) == category:
                agents[row["agent_id"]].append(row["difference_b5_minus_b4"])
        values.append([_mean(agent_values) for agent_values in agents.values()])
    return categories, values


def write_subgroup_figures(
    details: Sequence[dict[str, Any]], output_dir: Path
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrices = {k: _heatmap_values(details, k) for k in (5, 10)}
    finite = [
        abs(value) for matrix in matrices.values()
        for row in matrix for value in row if math.isfinite(value)
    ]
    limit = max(finite, default=1.0) or 1.0
    for k, matrix in matrices.items():
        figure, axis = plt.subplots(figsize=(8.2, 5.6))
        image = axis.imshow(matrix, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        axis.set_xticks(range(len(PROFILES)), PROFILES)
        axis.set_yticks(range(len(CONDITIONS)), CONDITIONS)
        axis.set_xlabel("Perfil de informação")
        axis.set_ylabel("Condição de feedback")
        axis.set_title(f"Diferença média B5 − B4 em NDCG@{k}")
        for row_index, row in enumerate(matrix):
            for column_index, value in enumerate(row):
                label = f"{value:+.3f}" if math.isfinite(value) else "N/D"
                axis.text(column_index, row_index, label, ha="center", va="center", fontsize=8)
        colorbar = figure.colorbar(image, ax=axis)
        colorbar.set_label("B5 − B4 (positivo favorece B5)")
        figure.tight_layout()
        figure.savefig(output_dir / f"subgroup_heatmap_k{k}.png", dpi=180)
        plt.close(figure)

    labels = {
        "condition": "Condição",
        "profile": "Perfil",
        "persona": "Persona sintética",
        "seed": "Seed",
    }
    distribution_values = []
    for target_k in (5, 10):
        for dimension in ("condition", "profile", "persona", "seed"):
            _, groups = _agent_distributions(details, dimension, target_k)
            distribution_values.extend(
                abs(value) for group in groups for value in group
            )
    distribution_limit = (max(distribution_values, default=1.0) or 1.0) * 1.05
    for k in (5, 10):
        figure, axes = plt.subplots(2, 2, figsize=(14, 9))
        for axis, dimension in zip(axes.flat, ("condition", "profile", "persona", "seed")):
            categories, values = _agent_distributions(details, dimension, k)
            axis.boxplot(values, tick_labels=categories, showmeans=True)
            axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
            axis.set_title(labels[dimension])
            axis.set_ylabel(f"B5 − B4 em NDCG@{k}")
            axis.tick_params(axis="x", rotation=30 if dimension == "persona" else 0)
            axis.set_ylim(-distribution_limit, distribution_limit)
        figure.suptitle(
            f"Distribuições pareadas por agente — NDCG@{k}\n"
            "Análise exploratória; positivo favorece B5"
        )
        figure.tight_layout()
        figure.savefig(output_dir / f"subgroup_distributions_k{k}.png", dpi=180)
        plt.close(figure)
