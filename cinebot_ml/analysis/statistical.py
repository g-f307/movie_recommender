"""Análise principal B5 × B4 em NDCG@5, agregada por agente."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from .inference import describe, paired_inference
from .loader import MetricRecord, OfficialResultsError, load_official_records


class AnalysisError(ValueError):
    """Contrato estatístico ou pareamento inválido."""


def _agent_pairs(
    records: Sequence[MetricRecord],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected = [
        record
        for record in records
        if record.k == 5
        and record.method in {"B4", "B5"}
        and record.condition != "C0"
    ]
    invalid = sum(
        record.status != "evaluated" or record.metric is None for record in selected
    )
    expected_comparisons = {record.comparison_id for record in selected}
    by_comparison: dict[str, dict[str, MetricRecord]] = defaultdict(dict)
    for record in selected:
        if record.status == "evaluated" and record.metric is not None:
            if record.method in by_comparison[record.comparison_id]:
                raise AnalysisError(f"Método duplicado no par {record.comparison_id}.")
            by_comparison[record.comparison_id][record.method] = record
    incomplete = sum(
        set(by_comparison[comparison]) != {"B4", "B5"}
        for comparison in expected_comparisons
    )
    complete = [pair for pair in by_comparison.values() if set(pair) == {"B4", "B5"}]
    if not complete:
        raise AnalysisError("Nenhum par B5 × B4 válido em NDCG@5 após feedback.")

    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    metadata: dict[str, tuple[str, int]] = {}
    for pair in complete:
        b4, b5 = pair["B4"], pair["B5"]
        identity4 = (
            b4.agent_id, b4.persona, b4.seed, b4.condition, b4.profile, b4.run, b4.k
        )
        identity5 = (
            b5.agent_id, b5.persona, b5.seed, b5.condition, b5.profile, b5.run, b5.k
        )
        if identity4 != identity5:
            raise AnalysisError(f"Pareamento incompatível em {b4.comparison_id}.")
        grouped[b4.agent_id].append((float(b4.metric), float(b5.metric)))
        metadata[b4.agent_id] = (b4.persona, b4.seed)

    rows: list[dict[str, Any]] = []
    for agent_id in sorted(grouped):
        observations = grouped[agent_id]
        b4_mean = sum(value[0] for value in observations) / len(observations)
        b5_mean = sum(value[1] for value in observations) / len(observations)
        persona, seed = metadata[agent_id]
        rows.append({
            "agent_id": agent_id,
            "persona": persona,
            "seed": seed,
            "observations": len(observations),
            "b4_ndcg_at_5": b4_mean,
            "b5_ndcg_at_5": b5_mean,
            "difference_b5_minus_b4": b5_mean - b4_mean,
        })
    exclusions = {
        "invalid_metric_or_status": invalid,
        "incomplete_comparison": incomplete,
        "c0_records_excluded": sum(
            record.k == 5
            and record.method in {"B4", "B5"}
            and record.condition == "C0"
            for record in records
        ),
    }
    return rows, exclusions


def analyze_official_results(
    manifest_path: Path,
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = Path(manifest_path).resolve()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    try:
        manifest, records = load_official_records(manifest_path)
    except OfficialResultsError as exc:
        raise AnalysisError(str(exc)) from exc
    rows, exclusions = _agent_pairs(records)
    b4 = [row["b4_ndcg_at_5"] for row in rows]
    b5 = [row["b5_ndcg_at_5"] for row in rows]
    inference = paired_inference(
        b4, b5, resamples=bootstrap_resamples, seed=bootstrap_seed
    )
    differences = [right - left for left, right in zip(b4, b5)]
    report = {
        "schema_version": "1.0",
        "analysis": "confirmatory_b5_vs_b4_ndcg_at_5",
        "evidence": manifest["evidence"],
        "matrix_id": manifest["matrix_id"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "unit_of_analysis": "independent_synthetic_agent",
        "aggregation": (
            "mean across profiles P0-P5 and post-feedback conditions C1-C5 within agent"
        ),
        "metric": "ndcg_at_k",
        "k": 5,
        "methods": ["B4", "B5"],
        "conditions_included": ["C1", "C2", "C3", "C4", "C5"],
        "conditions_excluded": ["C0"],
        "bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "confidence": 0.95,
            "statistic": "mean paired difference",
        },
        "b4": describe(b4),
        "b5": describe(b5),
        "paired_difference_b5_minus_b4": {
            **describe(differences),
            **inference,
        },
        "accounting": {
            "official_cells": manifest["total_cells"],
            "agent_units": len(rows),
            "minimum_confirmatory_units": 30,
            "confirmatory_sample_size_met": len(rows) >= 30,
            **exclusions,
        },
        "interpretation_guardrails": [
            "Evidence is synthetic and does not establish human preference or satisfaction.",
            "A non-significant result does not prove the null hypothesis.",
            "Subgroup analyses are outside this report and must be labelled exploratory.",
        ],
    }
    return report, rows


def write_outputs(
    report: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    json_output: Path,
    csv_output: Path,
) -> None:
    for path in (json_output, csv_output):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Saída existente não será sobrescrita: {path}")
    json_output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with csv_output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
