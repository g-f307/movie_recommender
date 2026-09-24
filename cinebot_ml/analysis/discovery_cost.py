"""Avaliação multidimensional de descoberta, não inferioridade e eficiência."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import time
import tracemalloc
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.experiments.units import HOLDOUT_CONFIG
from cinebot_ml.ranking.discovery_metrics import (
    PopularityReference,
    calculate_discovery_metrics,
    rank_position_variation_at_k,
    ranking_overlap_at_k,
    ranking_repetition_at_k,
)
from cinebot_ml.ranking.evaluation import load_benchmark_units

from .inference import bootstrap_mean_ci, describe, paired_inference
from .statistical import analyze_official_results


DEFAULT_MANIFEST = PROJECT_ROOT / "results/raw/official_v1_1/2942e51456add29e4c999307/experiment.manifest.json"
DEFAULT_STUDY_CONFIG = PROJECT_ROOT / "configs/experiments/discovery_cost_v1.json"
DEFAULT_CATALOG = PROJECT_ROOT / "results/derived/test_catalog.json"


class DiscoveryCostError(ValueError):
    """Indica evidência incompatível com o protocolo congelado."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("frozen") is not True or data.get("schema_version") != "1.0":
        raise DiscoveryCostError("Configuração deve estar congelada na versão 1.0.")
    margin = data.get("diversity_noninferiority", {}).get("absolute_margin")
    if not isinstance(margin, (int, float)) or not -1 < margin < 0:
        raise DiscoveryCostError("Margem de não inferioridade deve estar em (-1,0).")
    return data


def _catalog(path: Path) -> dict[int, dict[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    profiles = raw.get("perfis")
    if not isinstance(profiles, Mapping):
        raise DiscoveryCostError("Catálogo deve conter perfis.")
    movies: dict[int, dict[str, Any]] = {}
    for values in profiles.values():
        for movie in values:
            movie_id = int(movie["id"])
            if movie_id in movies and movies[movie_id] != movie:
                raise DiscoveryCostError(f"Metadados divergentes para movie_id={movie_id}.")
            movies[movie_id] = movie
    if not movies:
        raise DiscoveryCostError("Catálogo vazio.")
    return movies


def _official_cells(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise DiscoveryCostError("Manifesto oficial incompleto.")
    cells = []
    root = manifest_path.parent / "cells"
    for cell_id, expected_hash in sorted(manifest["cells_sha256"].items()):
        path = root / f"{cell_id}.json"
        if _sha256(path) != expected_hash:
            raise DiscoveryCostError(f"Hash divergente na célula {cell_id}.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        individual = payload.get("result", {}).get("individual", {})
        if payload.get("status") != "completed" or individual.get("ranking_status") != "completed":
            continue
        row = {**payload["cell"], **individual}
        row["ranked_movie_ids"] = [int(value) for value in individual["ranked_movie_ids"]]
        cells.append(row)
    if len(cells) != manifest["total_cells"]:
        raise DiscoveryCostError("Nem todas as células oficiais possuem ranking completo.")
    return manifest, cells


def _agent_summary(values: Sequence[float]) -> dict[str, Any]:
    summary = describe(values)
    low, high = bootstrap_mean_ci(values, resamples=10_000, seed=42)
    return {**summary, "confidence_interval_95": [low, high]}


def discovery_analysis(
    cells: Sequence[dict[str, Any]],
    catalog: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    counts = {movie_id: int(movie.get("votos", 0)) for movie_id, movie in catalog.items()}
    popularity = PopularityReference(
        counts,
        config["popularity_reference"]["partition"],
        config["popularity_reference"]["version"],
    )
    enriched = []
    for cell in cells:
        metrics = calculate_discovery_metrics(
            cell["ranked_movie_ids"], catalog, int(cell["k"]), popularity
        )
        if any(metrics[name] is None for name in (
            "diversity_at_k", "novelty_at_k",
            "popularity_exposure_at_k", "popularity_bias_at_k",
        )):
            raise DiscoveryCostError(f"Métrica de descoberta indefinida em {cell['cell_id']}.")
        enriched.append({
            "cell_id": cell["cell_id"],
            "comparison_id": cell["comparison_id"],
            "method": cell["method"],
            "condition": cell["condition"],
            "profile": cell["profile"],
            "persona": cell["persona"],
            "seed": cell["seed"],
            "run": cell["run"],
            "k": cell["k"],
            "agent_id": cell["agent_id"],
            "candidate_set_id": cell["candidate_set_id"],
            "ndcg_at_k": cell["metrics"]["ndcg_at_k"],
            **metrics,
        })

    by_unit: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in enriched:
        by_unit[(row["method"], int(row["k"]), row["agent_id"])].append(row)

    table = []
    for method in [f"B{i}" for i in range(6)]:
        for k in (5, 10):
            units = [rows for (m, item_k, _), rows in by_unit.items() if m == method and item_k == k]
            if not units:
                raise DiscoveryCostError(f"Sem unidades para {method}/K={k}.")
            aggregate = {}
            for metric in (
                "ndcg_at_k", "diversity_at_k", "novelty_at_k",
                "popularity_exposure_at_k", "popularity_bias_at_k",
            ):
                unit_values = [
                    statistics.fmean(float(row[metric]) for row in rows) for rows in units
                ]
                summary = _agent_summary(unit_values)
                aggregate[f"{metric}_mean"] = summary["mean"]
                aggregate[f"{metric}_ci95_low"] = summary["confidence_interval_95"][0]
                aggregate[f"{metric}_ci95_high"] = summary["confidence_interval_95"][1]
            recommended = {
                movie_id for row in enriched
                if row["method"] == method and int(row["k"]) == k
                for movie_id in next(cell["ranked_movie_ids"] for cell in cells if cell["cell_id"] == row["cell_id"])[:k]
            }
            aggregate["catalog_coverage_at_k"] = len(recommended) / len(catalog)
            aggregate["recommended_unique_items"] = len(recommended)
            table.append({
                "method": method, "k": k, "n_agents": len(units),
                "coverage_denominator": len(catalog), **aggregate,
            })

    stability = []
    groups: dict[tuple[Any, ...], dict[int, list[int]]] = defaultdict(dict)
    for cell in cells:
        key = (
            cell["method"], cell["profile"], cell["agent_id"], cell["run"], cell["k"]
        )
        groups[key][int(str(cell["condition"])[1:])] = cell["ranked_movie_ids"]
    by_method_k: dict[tuple[str, int], list[tuple[float, float, float]]] = defaultdict(list)
    for (method, _profile, _agent, _run, k), sequence in groups.items():
        for condition in range(1, 6):
            if condition - 1 not in sequence or condition not in sequence:
                continue
            previous, current = sequence[condition - 1], sequence[condition]
            by_method_k[(method, int(k))].append((
                float(ranking_overlap_at_k(previous, current, int(k))),
                float(ranking_repetition_at_k(previous, current, int(k))),
                float(rank_position_variation_at_k(previous, current, int(k)) or 0.0),
            ))
    for (method, k), values in sorted(by_method_k.items()):
        stability.append({
            "method": method, "k": k, "transitions": len(values),
            "ranking_overlap_at_k": statistics.fmean(row[0] for row in values),
            "ranking_repetition_at_k": statistics.fmean(row[1] for row in values),
            "rank_position_variation_at_k": statistics.fmean(row[2] for row in values),
        })
    metadata = {
        "popularity_distribution_id": popularity.distribution_id,
        "popularity_source": config["popularity_reference"],
        "catalog_items": len(catalog),
        "metric_directions": {
            "ndcg_at_k": "higher",
            "diversity_at_k": "higher",
            "novelty_at_k": "higher",
            "catalog_coverage_at_k": "higher",
            "popularity_exposure_at_k": "descriptive",
            "popularity_bias_at_k": "closer_to_zero",
            "ranking_overlap_at_k": "descriptive",
            "ranking_repetition_at_k": "lower",
            "rank_position_variation_at_k": "descriptive",
        },
    }
    return table, stability, metadata


def noninferiority(
    cells: Sequence[dict[str, Any]],
    catalog: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    relevance_result: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    post = set(config["post_feedback_conditions"])
    selected = [
        cell for cell in cells
        if cell["method"] in {"B4", "B5"} and cell["condition"] in post and cell["k"] == 5
    ]
    pairs: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for cell in selected:
        pairs[cell["comparison_id"]][cell["method"]] = cell
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for comparison_id, pair in pairs.items():
        if set(pair) != {"B4", "B5"}:
            raise DiscoveryCostError(f"Par incompleto em {comparison_id}.")
        left, right = pair["B4"], pair["B5"]
        if left["candidate_set_id"] != right["candidate_set_id"]:
            raise DiscoveryCostError(f"Candidatos divergentes em {comparison_id}.")
        b4 = calculate_discovery_metrics(left["ranked_movie_ids"], catalog, 5)["diversity_at_k"]
        b5 = calculate_discovery_metrics(right["ranked_movie_ids"], catalog, 5)["diversity_at_k"]
        if b4 is None or b5 is None:
            raise DiscoveryCostError(f"Diversidade indefinida em {comparison_id}.")
        grouped[left["agent_id"]].append((b4, b5))
    rows = []
    for agent_id in sorted(grouped):
        observations = grouped[agent_id]
        b4 = statistics.fmean(value[0] for value in observations)
        b5 = statistics.fmean(value[1] for value in observations)
        rows.append({
            "agent_id": agent_id, "observations": len(observations),
            "b4_diversity_at_5": b4, "b5_diversity_at_5": b5,
            "difference_b5_minus_b4": b5 - b4,
        })
    left = [row["b4_diversity_at_5"] for row in rows]
    right = [row["b5_diversity_at_5"] for row in rows]
    differences = np.asarray([row["difference_b5_minus_b4"] for row in rows])
    margin = float(config["diversity_noninferiority"]["absolute_margin"])
    shifted = differences - margin
    test = wilcoxon(shifted, alternative="greater", zero_method="wilcox", method="auto")
    inference = paired_inference(left, right, resamples=10_000, seed=42)
    lower = inference["confidence_interval_95"][0]
    relevance = relevance_result or analyze_official_results(DEFAULT_MANIFEST)[0][
        "paired_difference_b5_minus_b4"
    ]
    diversity_noninferior = lower > margin and float(test.pvalue) < 0.05
    relevance_gain = relevance["mean_difference"] > 0 and relevance["confidence_interval_95"][0] > 0
    return {
        "hypothesis": "H5",
        "metric": "diversity_at_k",
        "k": 5,
        "margin": margin,
        "margin_rationale": config["diversity_noninferiority"]["rationale"],
        "n_agents": len(rows),
        "diversity": {**inference, "p_value_one_sided_noninferiority": float(test.pvalue)},
        "diversity_noninferior": diversity_noninferior,
        "relevance_gain_required": True,
        "relevance_gain_observed": relevance_gain,
        "relevance_difference_b5_minus_b4": relevance["mean_difference"],
        "decision": "supported" if diversity_noninferior and relevance_gain else "not_supported",
        "reason": (
            "H5 exige simultaneamente ganho de relevância e não inferioridade de diversidade; "
            "um resultado favorável em descoberta não compensa perda de relevância."
        ),
    }, rows


def _benchmark_units(cells: Sequence[dict[str, Any]]) -> list[Path]:
    selected: dict[str, str] = {}
    for cell in cells:
        if (
            cell["method"] == "B0" and cell["condition"] == "C5"
            and cell["profile"] == "P5" and cell["k"] == 10
            and cell["run"] == 1 and cell["seed"] == 42
        ):
            selected.setdefault(cell["persona"], cell["comparison_id"])
    if len(selected) != 7:
        raise DiscoveryCostError("Amostra de eficiência deve cobrir as sete personas.")
    return [PROJECT_ROOT / "results/units_v1_1" / f"{selected[key]}.json" for key in sorted(selected)]


def efficiency_analysis(
    cells: Sequence[dict[str, Any]],
    config: Mapping[str, Any],
    *,
    warmup: int | None = None,
    repetitions: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    protocol = config["efficiency"]
    warmup = int(protocol["warmup"] if warmup is None else warmup)
    repetitions = int(protocol["repetitions"] if repetitions is None else repetitions)
    records: dict[str, list[tuple[float, int]]] = defaultdict(list)
    unit_paths = _benchmark_units(cells)
    for path in unit_paths:
        units = load_benchmark_units(
            path,
            config_path=HOLDOUT_CONFIG,
            b2_artifact_path=PROJECT_ROOT / "artifacts/b2_tfidf_v1.json",
            b3_model_path=PROJECT_ROOT / "artifacts/b3_experiment_v1.joblib",
            b3_metadata_path=PROJECT_ROOT / "artifacts/b3_experiment_v1.json",
            methods=tuple(f"B{i}" for i in range(6)),
        )
        unit = units[0]
        for method, recommender in sorted(unit.recommenders.items()):
            request = replace(
                unit.request, method=method, method_version=recommender.method_version,
                candidate_movie_ids=unit.candidates.movie_ids, k=10,
                history=unit.request.history if method == "B5" else (),
            )
            for _ in range(warmup):
                recommender.recommend(request)
            for _ in range(repetitions):
                tracemalloc.start()
                before, _ = tracemalloc.get_traced_memory()
                started = time.perf_counter_ns()
                result = recommender.recommend(request)
                elapsed = (time.perf_counter_ns() - started) / 1_000_000
                _current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                if result.status != "completed":
                    raise DiscoveryCostError(f"Benchmark falhou em {method}.")
                records[method].append((elapsed, max(0, peak - before)))
    artifact_paths = {
        "B0": [], "B1": [], "B2": [PROJECT_ROOT / "artifacts/b2_tfidf_v1.json"],
        "B3": [
            PROJECT_ROOT / "artifacts/b3_experiment_v1.joblib",
            PROJECT_ROOT / "artifacts/b3_experiment_v1.json",
        ],
        "B4": [], "B5": [],
    }
    table = []
    for method in [f"B{i}" for i in range(6)]:
        values = records[method]
        latencies = [row[0] for row in values]
        memories = [row[1] for row in values]
        table.append({
            "method": method,
            "benchmark_rankings": len(values),
            "latency_ms_mean": statistics.fmean(latencies),
            "latency_ms_median": statistics.median(latencies),
            "latency_ms_p95": float(np.percentile(latencies, 95)),
            "peak_memory_bytes_mean": statistics.fmean(memories),
            "peak_memory_bytes_max": max(memories),
            "artifact_size_bytes": sum(path.stat().st_size for path in artifact_paths[method]),
            "training_time_seconds": None,
            "training_time_status": "not_applicable_frozen_artifact",
            "cost_per_ranking_unit": "milliseconds_on_controlled_local_environment",
        })
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or "not_reported",
        "warmup": warmup,
        "repetitions_per_unit": repetitions,
        "units": len(unit_paths),
        "candidate_count": 295,
        "k": 10,
        "clock": protocol["clock"],
        "memory_measure": protocol["memory"],
        "training_time": protocol["training_time"],
        "artifact_scope": "serialized inference artifacts only",
    }
    return table, environment


def pareto_frontier(
    discovery: Sequence[dict[str, Any]], efficiency: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    quality = {row["method"]: row for row in discovery if row["k"] == 5}
    cost = {row["method"]: row for row in efficiency}
    points = []
    for method in sorted(quality):
        row = {
            "method": method,
            "ndcg_at_5": quality[method]["ndcg_at_k_mean"],
            "diversity_at_5": quality[method]["diversity_at_k_mean"],
            "novelty_at_5": quality[method]["novelty_at_k_mean"],
            "coverage_at_5": quality[method]["catalog_coverage_at_k"],
            "latency_ms": cost[method]["latency_ms_median"],
            "peak_memory_bytes": cost[method]["peak_memory_bytes_mean"],
        }
        points.append(row)
    for row in points:
        dominated_by = []
        for other in points:
            if other["method"] == row["method"]:
                continue
            no_worse = (
                other["ndcg_at_5"] >= row["ndcg_at_5"]
                and other["diversity_at_5"] >= row["diversity_at_5"]
                and other["novelty_at_5"] >= row["novelty_at_5"]
                and other["coverage_at_5"] >= row["coverage_at_5"]
                and other["latency_ms"] <= row["latency_ms"]
                and other["peak_memory_bytes"] <= row["peak_memory_bytes"]
            )
            strictly = any((
                other["ndcg_at_5"] > row["ndcg_at_5"],
                other["diversity_at_5"] > row["diversity_at_5"],
                other["novelty_at_5"] > row["novelty_at_5"],
                other["coverage_at_5"] > row["coverage_at_5"],
                other["latency_ms"] < row["latency_ms"],
                other["peak_memory_bytes"] < row["peak_memory_bytes"],
            ))
            if no_worse and strictly:
                dominated_by.append(other["method"])
        row["pareto_optimal"] = not dominated_by
        row["dominated_by"] = ",".join(dominated_by)
    return points


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(discovery: Sequence[dict[str, Any]], pareto: Sequence[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [row for row in discovery if row["k"] == 5]
    figure, axis = plt.subplots(figsize=(8, 5))
    for row in rows:
        axis.scatter(row["ndcg_at_k_mean"], row["diversity_at_k_mean"], s=70)
        axis.annotate(row["method"], (row["ndcg_at_k_mean"], row["diversity_at_k_mean"]))
    axis.set(xlabel="NDCG@5 (maior é melhor)", ylabel="Diversidade@5 (maior é melhor)",
             title="Trade-off relevância × diversidade — agentes sintéticos")
    axis.grid(alpha=.25); figure.tight_layout()
    figure.savefig(output / "relevance_discovery.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    for row in pareto:
        axis.scatter(row["latency_ms"], row["ndcg_at_5"],
                     marker="o" if row["pareto_optimal"] else "x", s=70)
        axis.annotate(row["method"], (row["latency_ms"], row["ndcg_at_5"]))
    axis.set(xlabel="Latência mediana (ms; menor é melhor)", ylabel="NDCG@5",
             title="Qualidade × custo e fronteira de Pareto")
    axis.grid(alpha=.25); figure.tight_layout()
    figure.savefig(output / "quality_cost_pareto.png", dpi=180); plt.close(figure)


def execute(
    manifest_path: Path,
    config_path: Path,
    catalog_path: Path,
    output_dir: Path,
    *,
    warmup: int | None = None,
    repetitions: int | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Saída existente não será sobrescrita: {output_dir}")
    config = _load_config(config_path)
    manifest, cells = _official_cells(manifest_path)
    catalog = _catalog(catalog_path)
    discovery, stability, metadata = discovery_analysis(cells, catalog, config)
    relevance_result = analyze_official_results(manifest_path)[0][
        "paired_difference_b5_minus_b4"
    ]
    noninferiority_result, ni_rows = noninferiority(
        cells, catalog, config, relevance_result=relevance_result
    )
    efficiency, environment = efficiency_analysis(
        cells, config, warmup=warmup, repetitions=repetitions
    )
    pareto = pareto_frontier(discovery, efficiency)
    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "discovery.csv", discovery)
    _write_csv(output_dir / "stability.csv", stability)
    _write_csv(output_dir / "efficiency.csv", efficiency)
    _write_csv(output_dir / "noninferiority_units.csv", ni_rows)
    _write_csv(output_dir / "pareto.csv", pareto)
    report = {
        "schema_version": "1.0",
        "study": config["study"],
        "evidence": manifest["evidence"],
        "matrix_id": manifest["matrix_id"],
        "source_manifest_sha256": _sha256(manifest_path),
        "study_config_sha256": _sha256(config_path),
        "catalog_sha256": _sha256(catalog_path),
        "definitions": metadata,
        "discovery": discovery,
        "stability": stability,
        "noninferiority": noninferiority_result,
        "efficiency": efficiency,
        "benchmark_environment": environment,
        "pareto": pareto,
        "decisions": {
            "RQ3_H5": noninferiority_result["decision"],
            "RQ4": "partially_answered_quality_and_controlled_inference_cost",
        },
        "guardrails": [
            "Nenhum score composto foi criado.",
            "Qualidade e custo são reportados separadamente.",
            "Resultados sintéticos não demonstram preferência humana.",
            "Tempos são específicos do ambiente documentado e não comparam hardware.",
            "Metadados de popularidade pertencem à mesma versão congelada do catálogo.",
        ],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _plots(discovery, pareto, output_dir)
    files = {
        path.name: _sha256(path) for path in sorted(output_dir.iterdir()) if path.is_file()
    }
    (output_dir / "study.manifest.json").write_text(
        json.dumps({
            "schema_version": "1.0", "study": config["study"],
            "matrix_id": manifest["matrix_id"],
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip(),
            "files": files,
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--config", type=Path, default=DEFAULT_STUDY_CONFIG)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--repetitions", type=int)
    args = parser.parse_args()
    config = _load_config(args.config)
    output = args.output_dir or (
        PROJECT_ROOT / "results/derived/specialized_v1/discovery_cost"
        / config["source_matrix_id"]
    )
    report = execute(
        args.manifest, args.config, args.catalog, output,
        warmup=args.warmup, repetitions=args.repetitions,
    )
    print(json.dumps({
        "output_dir": str(output), "decisions": report["decisions"],
        "noninferiority": report["noninferiority"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
