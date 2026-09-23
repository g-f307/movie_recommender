"""Síntese rastreável de cold start, convergência, ablação e robustez."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from scipy.stats import friedmanchisquare, wilcoxon

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.experiments.ablation import load_ablation_variants
from cinebot_ml.experiments.convergence import load_convergence_config
from cinebot_ml.experiments.robustness import load_robustness_config

from .loader import MetricRecord, OfficialResultsError, load_official_records
from .subgroups import holm_adjust


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results/raw/official_v1_1/2942e51456add29e4c999307/experiment.manifest.json"
)
SPECIALIZED_ROOT = PROJECT_ROOT / "results/derived/specialized_v1"


class SynthesisError(ValueError):
    """Indica evidência incompleta ou longitudinalmente incompatível."""


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise SynthesisError("Não é possível resumir um grupo vazio.")
    return sum(values) / len(values)


def _valid(records: Sequence[MetricRecord], *, k: int = 5) -> list[MetricRecord]:
    return [
        row for row in records
        if row.k == k and row.status == "evaluated" and row.metric is not None
    ]


def cold_start_summary(records: Sequence[MetricRecord], *, k: int = 5) -> list[dict[str, Any]]:
    """Resume P0--P5 sem contar medidas repetidas como agentes independentes."""
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in _valid(records, k=k):
        grouped[(row.profile, row.method, row.agent_id)].append(float(row.metric))
    profiles = {f"P{index}" for index in range(6)}
    methods = {f"B{index}" for index in range(6)}
    if {key[0] for key in grouped} != profiles or {key[1] for key in grouped} != methods:
        raise SynthesisError("Cold start não contém exatamente P0–P5 e B0–B5.")
    rows = []
    for profile in sorted(profiles):
        for method in sorted(methods):
            units = [_mean(values) for (p, m, _), values in grouped.items() if (p, m) == (profile, method)]
            rows.append({
                "profile": profile, "method": method, "k": k,
                "unit_of_analysis": "independent_synthetic_agent",
                "n_units": len(units), "mean_ndcg_at_k": _mean(units),
                "minimum_ndcg_at_k": min(units), "maximum_ndcg_at_k": max(units),
            })
    return rows


def convergence_summary(records: Sequence[MetricRecord], *, k: int = 5) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserva o agente ao comparar C0--C5 e condiciona pós-testes ao Friedman."""
    selected = [row for row in _valid(records, k=k) if row.method in {"B4", "B5"}]
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in selected:
        grouped[(row.agent_id, row.method, row.condition)].append(float(row.metric))
    conditions = [f"C{index}" for index in range(6)]
    agents = sorted({row.agent_id for row in selected})
    if any((agent, method, condition) not in grouped for agent in agents for method in ("B4", "B5") for condition in conditions):
        raise SynthesisError("Trajetória incompleta em C0–C5 para B4/B5.")
    unit = {(agent, method, condition): _mean(grouped[(agent, method, condition)]) for agent in agents for method in ("B4", "B5") for condition in conditions}
    rows = []
    previous: dict[str, float] = {}
    for condition in conditions:
        for method in ("B4", "B5"):
            values = [unit[(agent, method, condition)] for agent in agents]
            mean = _mean(values)
            b4 = _mean([unit[(agent, "B4", condition)] for agent in agents])
            rows.append({
                "condition": condition, "method": method, "k": k,
                "unit_of_analysis": "independent_synthetic_agent",
                "n_units": len(values), "mean_ndcg_at_k": mean,
                "gain_vs_c0": mean - _mean([unit[(agent, method, "C0")] for agent in agents]),
                "gain_b5_vs_b4": mean - b4 if method == "B5" else None,
                "marginal_gain": None if method not in previous else mean - previous[method],
            })
            previous[method] = mean
    samples = [[unit[(agent, "B5", condition)] for agent in agents] for condition in conditions]
    statistic, global_p = friedmanchisquare(*samples)
    post_tests = []
    if global_p < 0.05:
        raw = []
        for condition in conditions[1:]:
            result = wilcoxon(samples[0], samples[int(condition[1:])], alternative="two-sided")
            raw.append((condition, float(result.statistic), float(result.pvalue)))
        adjusted = holm_adjust([item[2] for item in raw])
        post_tests = [
            {"comparison": f"{condition}_vs_C0", "statistic": statistic_value,
             "p_value_raw": p_value, "p_value_holm": corrected}
            for (condition, statistic_value, p_value), corrected in zip(raw, adjusted)
        ]
    inference = {
        "test": "Friedman", "method": "B5", "conditions": conditions,
        "n_longitudinal_units": len(agents), "statistic": float(statistic),
        "p_value": float(global_p), "alpha": 0.05,
        "post_tests_performed": bool(post_tests), "post_tests": post_tests,
    }
    return rows, inference


def _specialized_inventory(root: Path) -> dict[str, Any]:
    variants, ablation_hash = load_ablation_variants()
    scenarios, seeds, robustness_hash = load_robustness_config()
    convergence = load_convergence_config()
    ablation_outputs = sorted((root / "ablation").glob("*/ablation.manifest.json"))
    robustness_outputs = sorted((root / "robustness").glob("*/robustness.manifest.json"))
    return {
        "convergence_config_sha256": convergence.source_sha256,
        "ablation": {
            "analysis_type": "exploratory", "config_sha256": ablation_hash,
            "status": "available" if ablation_outputs else "unavailable",
            "reason": None if ablation_outputs else "No valid A0–A6 execution exists; official holdout cells cannot be repurposed as ablations.",
            "variants": [{"variant": item.name, "variant_id": item.variant_id,
                          "removed_components": list(item.removed)} for item in variants],
            "manifests": [str(path) for path in ablation_outputs],
        },
        "robustness": {
            "analysis_type": "exploratory", "config_sha256": robustness_hash,
            "status": "available" if robustness_outputs else "unavailable",
            "reason": None if robustness_outputs else "No valid 12-scenario execution exists; unavailable metrics are not imputed.",
            "seeds": list(seeds),
            "scenarios": [{"scenario": item.name, "kind": item.kind,
                           "intensity": item.intensity, "status": "pending" if not robustness_outputs else "available"}
                          for item in scenarios],
            "manifests": [str(path) for path in robustness_outputs],
        },
    }


def synthesize(manifest_path: Path, *, specialized_root: Path = SPECIALIZED_ROOT) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        manifest, records = load_official_records(manifest_path)
    except OfficialResultsError as exc:
        raise SynthesisError(str(exc)) from exc
    cold = cold_start_summary(records)
    convergence, inference = convergence_summary(records)
    inventory = _specialized_inventory(specialized_root)
    report = {
        "schema_version": "1.0", "analysis": "experiment_synthesis_v1",
        "evidence": manifest["evidence"], "matrix_id": manifest["matrix_id"],
        "official_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "confirmatory": {"primary_analysis": "B5_vs_B4_ndcg_at_5", "source": "statistical_analysis.json"},
        "exploratory": {
            "cold_start": {"status": "available", "profiles": [f"P{i}" for i in range(6)]},
            "convergence": {"status": "available", "inference": inference},
            **inventory,
        },
        "metric_availability": {
            "relevance": "ndcg_at_k available",
            "diversity": "diversity_at_k available in official cells",
            "novelty": "unavailable in official cells",
            "popularity_bias": "unavailable in official cells",
        },
        "guardrails": [
            "Exploratory studies are not confirmatory evidence.",
            "The official holdout is summarized, never reused for model or scenario selection.",
            "Unavailable studies and metrics are reported rather than imputed.",
            "Synthetic-agent findings do not establish human-user generalization.",
        ],
    }
    return report, cold, convergence


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def write_outputs(report: dict[str, Any], cold: Sequence[dict[str, Any]], convergence: Sequence[dict[str, Any]], *, output_dir: Path) -> list[Path]:
    paths = [output_dir / "synthesis.manifest.json", output_dir / "cold_start.csv", output_dir / "convergence.csv"]
    if any(path.exists() for path in paths):
        raise FileExistsError("Saídas de síntese existentes não serão sobrescritas.")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths[0].write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _write_csv(paths[1], cold); _write_csv(paths[2], convergence)
    return paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--specialized-root", type=Path, default=SPECIALIZED_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/reports/experiment_synthesis_v1")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report, cold, convergence = synthesize(args.manifest, specialized_root=args.specialized_root)
    outputs = write_outputs(report, cold, convergence, output_dir=args.output_dir)
    print(json.dumps({"matrix_id": report["matrix_id"], "outputs": [str(path) for path in outputs]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
