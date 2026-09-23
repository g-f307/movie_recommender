"""Análise exploratória B5 × B4 por condição, perfil, persona e seed."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from cinebot_ml.config import PROJECT_ROOT

from .inference import describe, paired_inference
from .loader import MetricRecord, OfficialResultsError, load_official_records


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results/raw/official_v1_1/2942e51456add29e4c999307/experiment.manifest.json"
)
DIMENSIONS = ("condition", "profile", "persona", "seed")


class SubgroupAnalysisError(ValueError):
    """Indica pareamento ou configuração incompatível com a análise."""


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Ajusta uma família de valores-p pelo método step-down de Holm."""
    if any(value < 0.0 or value > 1.0 for value in p_values):
        raise ValueError("Valores-p devem pertencer a [0,1].")
    ordered = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running = 0.0
    total = len(ordered)
    for rank, (original, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[original] = running
    return adjusted


def _paired_details(records: Sequence[MetricRecord]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected = [
        record
        for record in records
        if record.method in {"B4", "B5"} and record.k in {5, 10}
    ]
    expected = {record.comparison_id for record in selected}
    pairs: dict[str, dict[str, MetricRecord]] = defaultdict(dict)
    invalid = 0
    for record in selected:
        if record.status != "evaluated" or record.metric is None:
            invalid += 1
            continue
        if record.method in pairs[record.comparison_id]:
            raise SubgroupAnalysisError(
                f"Método duplicado no par {record.comparison_id}."
            )
        pairs[record.comparison_id][record.method] = record

    rows: list[dict[str, Any]] = []
    incomplete = 0
    for comparison_id in sorted(expected):
        pair = pairs[comparison_id]
        if set(pair) != {"B4", "B5"}:
            incomplete += 1
            continue
        b4, b5 = pair["B4"], pair["B5"]
        identity4 = (
            b4.agent_id, b4.persona, b4.seed, b4.condition, b4.profile, b4.run, b4.k
        )
        identity5 = (
            b5.agent_id, b5.persona, b5.seed, b5.condition, b5.profile, b5.run, b5.k
        )
        if identity4 != identity5:
            raise SubgroupAnalysisError(f"Pareamento incompatível em {comparison_id}.")
        rows.append({
            "comparison_id": comparison_id,
            "b4_cell_id": b4.cell_id,
            "b5_cell_id": b5.cell_id,
            "agent_id": b4.agent_id,
            "persona": b4.persona,
            "seed": b4.seed,
            "condition": b4.condition,
            "profile": b4.profile,
            "run": b4.run,
            "k": b4.k,
            "b4_ndcg_at_k": float(b4.metric),
            "b5_ndcg_at_k": float(b5.metric),
            "difference_b5_minus_b4": float(b5.metric) - float(b4.metric),
        })
    if not rows:
        raise SubgroupAnalysisError("Nenhum par B5 × B4 válido.")
    return rows, {
        "selected_records": len(selected),
        "valid_comparisons": len(rows),
        "invalid_metric_or_status": invalid,
        "incomplete_comparisons": incomplete,
    }


def _category(row: dict[str, Any], dimension: str) -> str:
    return str(row[dimension])


def _eligible(row: dict[str, Any], dimension: str) -> bool:
    # C0 é necessária para a curva por condição, mas não representa uso de feedback.
    return dimension == "condition" or row["condition"] != "C0"


def _unit_values(
    details: Sequence[dict[str, Any]], dimension: str, category: str, k: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        if row["k"] == k and _eligible(row, dimension) and _category(row, dimension) == category:
            grouped[row["agent_id"]].append(row)
    units = []
    for agent_id in sorted(grouped):
        observations = grouped[agent_id]
        b4 = sum(row["b4_ndcg_at_k"] for row in observations) / len(observations)
        b5 = sum(row["b5_ndcg_at_k"] for row in observations) / len(observations)
        units.append({
            "agent_id": agent_id,
            "persona": observations[0]["persona"],
            "seed": observations[0]["seed"],
            "observations": len(observations),
            "b4": b4,
            "b5": b5,
            "difference": b5 - b4,
        })
    return units


def _dimension_categories(
    details: Sequence[dict[str, Any]], dimension: str
) -> list[str]:
    values = {_category(row, dimension) for row in details}
    if dimension == "seed":
        return [str(value) for value in sorted(int(value) for value in values)]
    return sorted(values)


def subgroup_summaries(
    details: Sequence[dict[str, Any]],
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 42,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for k in (5, 10):
        for dimension in DIMENSIONS:
            family: list[dict[str, Any]] = []
            for category in _dimension_categories(details, dimension):
                units = _unit_values(details, dimension, category, k)
                if not units:
                    continue
                b4 = [row["b4"] for row in units]
                b5 = [row["b5"] for row in units]
                differences = [row["difference"] for row in units]
                inference = paired_inference(
                    b4, b5, resamples=bootstrap_resamples, seed=bootstrap_seed
                )
                family.append({
                    "dimension": dimension,
                    "category": category,
                    "k": k,
                    "analysis_type": "exploratory",
                    "unit_of_analysis": "independent_synthetic_agent",
                    "n_units": len(units),
                    "small_subgroup_warning": len(units) < 30,
                    "observations_per_unit_min": min(row["observations"] for row in units),
                    "observations_per_unit_max": max(row["observations"] for row in units),
                    "b4_mean": sum(b4) / len(b4),
                    "b5_mean": sum(b5) / len(b5),
                    "difference_mean": inference["mean_difference"],
                    "difference_median": describe(differences)["median"],
                    "ci95_low": inference["confidence_interval_95"][0],
                    "ci95_high": inference["confidence_interval_95"][1],
                    "positive": inference["positive"],
                    "negative": inference["negative"],
                    "ties": inference["ties"],
                    "wilcoxon_statistic": inference["wilcoxon_statistic"],
                    "p_value_raw": inference["p_value_two_sided"],
                    "rank_biserial_correlation": inference["rank_biserial_correlation"],
                    "direction": (
                        "helps" if inference["mean_difference"] > 0
                        else "harms" if inference["mean_difference"] < 0
                        else "neutral"
                    ),
                })
            adjusted = holm_adjust([row["p_value_raw"] for row in family])
            for row, p_adjusted in zip(family, adjusted):
                row["p_value_holm"] = p_adjusted
                row["holm_family"] = f"{dimension}:k={k}"
                row["significant_after_holm"] = p_adjusted < 0.05
            summaries.extend(family)
    return summaries


def _extreme_cases(
    details: Sequence[dict[str, Any]], *, count: int = 10
) -> dict[str, list[dict[str, Any]]]:
    eligible = [row for row in details if row["condition"] != "C0"]
    fields = (
        "comparison_id", "b4_cell_id", "b5_cell_id", "agent_id", "persona",
        "seed", "condition", "profile", "k", "b4_ndcg_at_k", "b5_ndcg_at_k",
        "difference_b5_minus_b4",
    )
    compact = lambda row: {field: row[field] for field in fields}
    return {
        "largest_gains": [
            compact(row)
            for row in sorted(
                eligible, key=lambda item: item["difference_b5_minus_b4"], reverse=True
            )[:count]
        ],
        "largest_losses": [
            compact(row)
            for row in sorted(
                eligible, key=lambda item: item["difference_b5_minus_b4"]
            )[:count]
        ],
    }

def _seed_stability(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    stability = []
    for k in (5, 10):
        seed_rows = [
            row for row in summaries
            if row["dimension"] == "seed" and row["k"] == k
        ]
        values = [row["difference_mean"] for row in seed_rows]
        positive = sum(value > 0 for value in values)
        negative = sum(value < 0 for value in values)
        ties = sum(value == 0 for value in values)
        stability.append({
            "k": k,
            "seeds": len(values),
            "mean_of_seed_effects": sum(values) / len(values),
            "minimum_seed_effect": min(values),
            "maximum_seed_effect": max(values),
            "range": max(values) - min(values),
            "positive_seeds": positive,
            "negative_seeds": negative,
            "ties": ties,
            "sign_consistent": positive == len(values) or negative == len(values),
        })
    return stability



def analyze_subgroups(
    manifest_path: Path,
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        manifest, records = load_official_records(manifest_path)
    except OfficialResultsError as exc:
        raise SubgroupAnalysisError(str(exc)) from exc
    details, accounting = _paired_details(records)
    summaries = subgroup_summaries(
        details,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
    )
    report = {
        "schema_version": "1.0",
        "analysis": "exploratory_b5_vs_b4_subgroups",
        "analysis_type": "exploratory",
        "evidence": manifest["evidence"],
        "matrix_id": manifest["matrix_id"],
        "metric": "ndcg_at_k",
        "k_values": [5, 10],
        "dimensions": list(DIMENSIONS),
        "multiple_comparisons": {
            "method": "Holm",
            "alpha": 0.05,
            "families": "one family per dimension and K",
        },
        "accounting": accounting,
        "small_subgroup_rule": "n_units < 30",
        "summaries": summaries,
        "seed_stability": _seed_stability(summaries),
        "extreme_cases": _extreme_cases(details),
        "interpretation_guardrails": [
            "All subgroup findings are exploratory, including Holm-adjusted results.",
            "Personas are simulation behaviors and not human demographic groups.",
            "Small subgroups have low precision and cannot support confirmatory claims.",
            "K=10 is a sensitivity analysis; K=5 remains the primary endpoint.",
        ],
    }
    return report, summaries, details


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_subgroup_outputs(
    report: dict[str, Any],
    summaries: Sequence[dict[str, Any]],
    details: Sequence[dict[str, Any]],
    *,
    json_output: Path,
    summary_csv: Path,
    pairs_csv: Path,
    figures_dir: Path,
) -> list[Path]:
    from .subgroup_plotting import write_subgroup_figures

    targets = [json_output, summary_csv, pairs_csv]
    figure_targets = [
        figures_dir / f"subgroup_heatmap_k{k}.png" for k in (5, 10)
    ] + [
        figures_dir / f"subgroup_distributions_k{k}.png" for k in (5, 10)
    ]
    if any(path.exists() for path in [*targets, *figure_targets]):
        existing = [str(path) for path in [*targets, *figure_targets] if path.exists()]
        raise FileExistsError(f"Saídas existentes não serão sobrescritas: {existing}")
    for path in targets:
        path.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(summary_csv, summaries)
    _write_csv(pairs_csv, details)
    write_subgroup_figures(details, figures_dir)
    return [*targets, *figure_targets]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=PROJECT_ROOT / "results/reports/subgroup_analysis.json",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT / "results/tables/subgroup_summary.csv",
    )
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        default=PROJECT_ROOT / "results/tables/subgroup_pairs.csv",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=PROJECT_ROOT / "results/figures",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    report, summaries, details = analyze_subgroups(args.manifest)
    outputs = write_subgroup_outputs(
        report,
        summaries,
        details,
        json_output=args.json_output,
        summary_csv=args.summary_csv,
        pairs_csv=args.pairs_csv,
        figures_dir=args.figures_dir,
    )
    print(json.dumps({
        "matrix_id": report["matrix_id"],
        "valid_comparisons": report["accounting"]["valid_comparisons"],
        "subgroup_summaries": len(summaries),
        "outputs": [str(path) for path in outputs],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
