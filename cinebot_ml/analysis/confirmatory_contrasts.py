"""Contrastes confirmatórios H3 (P5 × P0) e H4 (B5 × B0)."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from cinebot_ml.config import PROJECT_ROOT

from .inference import describe, paired_inference
from .loader import MetricRecord, OfficialResultsError, load_official_records
from .subgroups import holm_adjust


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "results/raw/official_v1_1/2942e51456add29e4c999307/experiment.manifest.json"
)
PROFILE_ORDER = tuple(f"P{index}" for index in range(6))
POST_FEEDBACK = tuple(f"C{index}" for index in range(1, 6))


class ContrastAnalysisError(ValueError):
    """Indica violação do contrato de pareamento dos contrastes."""


def _valid(record: MetricRecord) -> bool:
    return record.status == "evaluated" and record.metric is not None


def _candidate_guard(left: MetricRecord, right: MetricRecord, label: str) -> None:
    if not left.candidate_set_id or not right.candidate_set_id:
        raise ContrastAnalysisError(f"candidate_set_id ausente em {label}.")
    if left.candidate_set_id != right.candidate_set_id:
        raise ContrastAnalysisError(f"Conjuntos candidatos divergentes em {label}.")


def _pair_records(
    records: Sequence[MetricRecord],
    *,
    selected: Callable[[MetricRecord], bool],
    side: Callable[[MetricRecord], str],
    identity: Callable[[MetricRecord], tuple[Any, ...]],
    expected_sides: tuple[str, str],
) -> tuple[list[tuple[MetricRecord, MetricRecord]], dict[str, int]]:
    chosen = [record for record in records if selected(record)]
    grouped: dict[tuple[Any, ...], dict[str, MetricRecord]] = defaultdict(dict)
    invalid = 0
    for record in chosen:
        if not _valid(record):
            invalid += 1
            continue
        key, label = identity(record), side(record)
        if label in grouped[key]:
            raise ContrastAnalysisError(f"Observação duplicada para {label} em {key}.")
        grouped[key][label] = record

    complete: list[tuple[MetricRecord, MetricRecord]] = []
    incomplete = 0
    left_label, right_label = expected_sides
    for key in sorted(grouped, key=str):
        pair = grouped[key]
        if set(pair) != set(expected_sides):
            incomplete += 1
            continue
        left, right = pair[left_label], pair[right_label]
        _candidate_guard(left, right, str(key))
        complete.append((left, right))
    if not complete:
        raise ContrastAnalysisError(
            f"Nenhum par completo para {left_label} × {right_label}."
        )
    return complete, {
        "selected_records": len(chosen),
        "invalid_metric_or_status": invalid,
        "incomplete_pairs": incomplete,
        "complete_observation_pairs": len(complete),
    }


def _agent_rows(
    pairs: Sequence[tuple[MetricRecord, MetricRecord]],
    *,
    left_name: str,
    right_name: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[MetricRecord, MetricRecord]]] = defaultdict(list)
    for left, right in pairs:
        if (left.agent_id, left.persona, left.seed) != (
            right.agent_id, right.persona, right.seed
        ):
            raise ContrastAnalysisError(f"Agentes incompatíveis em {left.cell_id}.")
        grouped[left.agent_id].append((left, right))

    rows: list[dict[str, Any]] = []
    for agent_id in sorted(grouped):
        observations = grouped[agent_id]
        left_value = sum(float(pair[0].metric) for pair in observations) / len(observations)
        right_value = sum(float(pair[1].metric) for pair in observations) / len(observations)
        exemplar = observations[0][0]
        rows.append({
            "agent_id": agent_id,
            "persona": exemplar.persona,
            "seed": exemplar.seed,
            "observations": len(observations),
            left_name: left_value,
            right_name: right_value,
            "difference": right_value - left_value,
        })
    return rows


def _summary(
    rows: Sequence[dict[str, Any]],
    *,
    left_name: str,
    right_name: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    left = [float(row[left_name]) for row in rows]
    right = [float(row[right_name]) for row in rows]
    differences = [float(row["difference"]) for row in rows]
    return {
        "left": describe(left),
        "right": describe(right),
        "difference": {
            **describe(differences),
            **paired_inference(
                left,
                right,
                resamples=bootstrap_resamples,
                seed=bootstrap_seed,
            ),
        },
    }


def _profile_curve(records: Sequence[MetricRecord], k: int) -> list[dict[str, Any]]:
    selected = [
        record
        for record in records
        if record.method == "B5"
        and record.condition in POST_FEEDBACK
        and record.k == k
        and _valid(record)
    ]
    by_profile_agent: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in selected:
        by_profile_agent[(record.profile, record.agent_id)].append(float(record.metric))
    curve = []
    for profile in PROFILE_ORDER:
        units = [
            sum(values) / len(values)
            for (item_profile, _), values in by_profile_agent.items()
            if item_profile == profile
        ]
        if not units:
            raise ContrastAnalysisError(f"Perfil {profile} sem observações em K={k}.")
        curve.append({"profile": profile, **describe(units)})
    return curve


def _monotonicity(records: Sequence[MetricRecord], k: int) -> dict[str, Any]:
    selected = [
        record
        for record in records
        if record.method == "B5"
        and record.condition in POST_FEEDBACK
        and record.k == k
        and _valid(record)
    ]
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in selected:
        grouped[(record.agent_id, record.profile)].append(float(record.metric))
    complete = 0
    nondecreasing = 0
    adjacent_positive = [0] * 5
    adjacent_total = [0] * 5
    for agent_id in sorted({key[0] for key in grouped}):
        values = []
        for profile in PROFILE_ORDER:
            observations = grouped.get((agent_id, profile), [])
            if not observations:
                values = []
                break
            values.append(sum(observations) / len(observations))
        if not values:
            continue
        complete += 1
        deltas = [values[index + 1] - values[index] for index in range(5)]
        nondecreasing += all(delta >= 0.0 for delta in deltas)
        for index, delta in enumerate(deltas):
            adjacent_positive[index] += delta > 0.0
            adjacent_total[index] += 1
    return {
        "analysis_type": "exploratory",
        "complete_agents": complete,
        "nondecreasing_agents": nondecreasing,
        "nondecreasing_fraction": nondecreasing / complete if complete else 0.0,
        "adjacent_positive_fraction": {
            f"P{index + 1}_minus_P{index}": (
                adjacent_positive[index] / adjacent_total[index]
                if adjacent_total[index] else 0.0
            )
            for index in range(5)
        },
    }


def _h3(
    records: Sequence[MetricRecord],
    k: int,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    pairs, accounting = _pair_records(
        records,
        selected=lambda record: (
            record.method == "B5"
            and record.profile in {"P0", "P5"}
            and record.condition in POST_FEEDBACK
            and record.k == k
        ),
        side=lambda record: record.profile,
        identity=lambda record: (
            record.agent_id, record.persona, record.seed,
            record.condition, record.run, record.k,
        ),
        expected_sides=("P0", "P5"),
    )
    rows = _agent_rows(pairs, left_name="p0_ndcg_at_k", right_name="p5_ndcg_at_k")
    return {
        "hypothesis": "H3",
        "contrast": "P5_minus_P0_under_B5",
        "method": "B5",
        "conditions": list(POST_FEEDBACK),
        "k": k,
        "unit_of_analysis": "independent_synthetic_agent",
        "aggregation": "mean across C1-C5 and repetitions within agent",
        "accounting": {**accounting, "agent_units": len(rows)},
        "summary": _summary(
            rows,
            left_name="p0_ndcg_at_k",
            right_name="p5_ndcg_at_k",
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
        "profile_curve": _profile_curve(records, k),
        "monotonicity": _monotonicity(records, k),
    }, rows


def _h4(
    records: Sequence[MetricRecord],
    k: int,
    phase: str,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conditions = ("C0",) if phase == "no_feedback" else POST_FEEDBACK
    pairs, accounting = _pair_records(
        records,
        selected=lambda record: (
            record.method in {"B0", "B5"}
            and record.condition in conditions
            and record.k == k
        ),
        side=lambda record: record.method,
        identity=lambda record: (
            record.agent_id, record.persona, record.seed, record.condition,
            record.profile, record.run, record.k,
        ),
        expected_sides=("B0", "B5"),
    )
    rows = _agent_rows(pairs, left_name="b0_ndcg_at_k", right_name="b5_ndcg_at_k")
    return {
        "hypothesis": "H4",
        "contrast": "B5_minus_B0",
        "phase": phase,
        "conditions": list(conditions),
        "k": k,
        "unit_of_analysis": "independent_synthetic_agent",
        "aggregation": "mean across conditions, profiles and repetitions within agent",
        "accounting": {**accounting, "agent_units": len(rows)},
        "summary": _summary(
            rows,
            left_name="b0_ndcg_at_k",
            right_name="b5_ndcg_at_k",
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
        ),
    }, rows


def _decision(
    result: dict[str, Any],
    *,
    adjusted_p: float,
    discovery_required: bool = False,
) -> dict[str, Any]:
    effect = result["summary"]["difference"]
    statistical = (
        effect["mean_difference"] > 0.0
        and effect["confidence_interval_95"][0] > 0.0
        and adjusted_p < 0.05
        and result["accounting"]["agent_units"] >= 30
    )
    if discovery_required:
        status = "relevance_supported_discovery_pending" if statistical else "not_supported"
    else:
        status = "supported" if statistical else "not_supported"
    return {
        "status": status,
        "statistical_criteria_met": statistical,
        "p_value_holm": adjusted_p,
        "alpha": 0.05,
        "discovery_evidence_required": discovery_required,
        "reason": (
            "O contraste de relevância atende aos critérios, mas H4 exige também "
            "evidência de descoberta, tratada separadamente."
            if discovery_required and statistical
            else "Critérios direcionais, IC95%, Holm e tamanho amostral avaliados conjuntamente."
        ),
    }


def analyze_confirmatory_contrasts(
    manifest_path: Path,
    *,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    manifest_path = Path(manifest_path).resolve()
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    try:
        manifest, records = load_official_records(manifest_path)
    except OfficialResultsError as exc:
        raise ContrastAnalysisError(str(exc)) from exc

    results: dict[str, Any] = {}
    rows: dict[str, list[dict[str, Any]]] = {}
    for k in (5, 10):
        results[f"h3_k{k}"], rows[f"p5_vs_p0_k{k}"] = _h3(
            records, k, bootstrap_resamples, bootstrap_seed
        )
        for phase in ("no_feedback", "post_feedback"):
            key = f"h4_{phase}_k{k}"
            results[key], rows[f"b5_vs_b0_{phase}_k{k}"] = _h4(
                records, k, phase, bootstrap_resamples, bootstrap_seed
            )

    primary = [results["h3_k5"], results["h4_post_feedback_k5"]]
    adjusted = holm_adjust([
        item["summary"]["difference"]["p_value_two_sided"] for item in primary
    ])
    decisions = {
        "H3": _decision(results["h3_k5"], adjusted_p=adjusted[0]),
        "H4": _decision(
            results["h4_post_feedback_k5"],
            adjusted_p=adjusted[1],
            discovery_required=True,
        ),
    }
    report = {
        "schema_version": "1.0",
        "analysis": "confirmatory_h3_h4_contrasts",
        "evidence": manifest["evidence"],
        "matrix_id": manifest["matrix_id"],
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha256,
        "metric": "ndcg_at_k",
        "primary_family": ["H3:P5_minus_P0:k5", "H4:B5_minus_B0:post_feedback:k5"],
        "multiplicity": "Holm step-down, alpha=0.05",
        "results": results,
        "decisions": decisions,
        "guardrails": [
            "Resultados descrevem agentes sintéticos e não demonstram satisfação humana.",
            "K=10 é análise de sensibilidade; K=5 é o desfecho principal.",
            "C0 e C1-C5 são relatados separadamente em H4.",
            "Ausências não são imputadas; pares incompletos são contabilizados.",
            "H4 só pode ser encerrada após avaliar descoberta e cobertura.",
        ],
    }
    return report, rows


def write_outputs(
    report: dict[str, Any],
    rows: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Saída existente não será sobrescrita: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / "contrasts.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "p5_vs_p0.json").write_text(
        json.dumps(
            {key: value for key, value in report["results"].items() if key.startswith("h3_")},
            ensure_ascii=False, sort_keys=True, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    (output_dir / "b5_vs_b0.json").write_text(
        json.dumps(
            {key: value for key, value in report["results"].items() if key.startswith("h4_")},
            ensure_ascii=False, sort_keys=True, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    for name, values in rows.items():
        path = output_dir / f"{name}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    manifest = {
        "schema_version": "1.0",
        "analysis": report["analysis"],
        "matrix_id": report["matrix_id"],
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output_dir.iterdir())
            if path.is_file()
        },
    }
    (output_dir / "analysis.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args()
    report, rows = analyze_confirmatory_contrasts(
        args.manifest,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT / "results/derived/specialized_v1/contrasts" / report["matrix_id"]
    )
    write_outputs(report, rows, output_dir)
    print(json.dumps({
        "matrix_id": report["matrix_id"],
        "output_dir": str(output_dir),
        "decisions": report["decisions"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
