"""Análise rastreável da execução complementar v2, sem combinar evidências v1.1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cinebot_ml.experiments.policy_sensitivity import (
    DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH,
    DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH,
    load_frozen_validation_selection,
    load_policy_sensitivity_config,
)

from .inference import describe, paired_inference
from .statistical import analyze_official_results
from .subgroups import holm_adjust


class V2AnalysisError(ValueError):
    """Manifesto, célula ou pareamento incompatível com o estudo v2."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise V2AnalysisError(f"JSON v2 inválido em {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise V2AnalysisError(f"Objeto JSON esperado em {path}.")
    return value


@dataclass(frozen=True)
class V2MetricRecord:
    cell_id: str
    comparison_id: str
    execution_id: str
    execution_config_sha256: str
    method: str
    condition: str
    profile: str
    persona: str
    seed: int
    run: int
    k: int
    agent_id: str
    candidate_set_id: str
    relevance_id: str
    partition: str
    scenario_id: str
    scenario_hash: str
    profile_noise_level: str
    exposure_policy: str
    trajectory_status: str
    trajectory_sha256: str | None
    status: str
    metric: float | None


_CELL_FIELDS = (
    "method", "condition", "profile", "persona", "seed", "run", "k",
)
_PAIR_FIELDS = (
    "execution_id", "execution_config_sha256", "partition", "agent_id", "persona",
    "seed", "condition", "profile", "run", "k", "candidate_set_id", "relevance_id", "scenario_id",
    "scenario_hash", "profile_noise_level", "exposure_policy", "trajectory_status",
    "trajectory_sha256",
)
_DIMENSIONS = (
    "scenario_id", "condition", "profile", "persona", "seed", "profile_noise_level",
)


def load_v2_records(manifest_path: Path) -> tuple[dict[str, Any], list[V2MetricRecord]]:
    """Carrega uma execução v2 selada e valida sua identidade antes da análise."""
    manifest_path = Path(manifest_path).resolve()
    manifest = _read_json(manifest_path)
    execution = manifest.get("execution")
    if manifest.get("status") != "complete" or not isinstance(execution, Mapping):
        raise V2AnalysisError("Manifesto não representa uma execução v2 completa.")
    execution_id = execution.get("id")
    config_sha = execution.get("config_sha256")
    if not isinstance(execution_id, str) or execution_id not in {"execution-v2", "execution-v2.1"}:
        raise V2AnalysisError("Manifesto não possui identidade de execução v2.")
    if not isinstance(config_sha, str) or len(config_sha) != 64:
        raise V2AnalysisError("Manifesto não possui hash de configuração v2 válido.")
    exposure_policy = execution.get("exposure_policy")
    if exposure_policy not in {"closed-loop-v2", "shared-seeded-v2.1"}:
        raise V2AnalysisError("A análise v2 exige uma política de exposição reconhecida.")

    matrix_info = manifest.get("matrix")
    if not isinstance(matrix_info, Mapping) or not isinstance(matrix_info.get("matrix_id"), str):
        raise V2AnalysisError("Manifesto v2 não referencia uma matriz válida.")
    root = manifest_path.parent
    matrix_path = root / "matrix.manifest.json"
    if root.name != matrix_info["matrix_id"] or not matrix_path.is_file():
        raise V2AnalysisError("Diretório ou manifesto da matriz v2 diverge da execução.")
    if _sha256(matrix_path) != matrix_info.get("manifest_sha256"):
        raise V2AnalysisError("Hash do manifesto da matriz v2 diverge da execução.")
    matrix = _read_json(matrix_path)
    if (
        matrix.get("matrix_id") != matrix_info["matrix_id"]
        or matrix.get("total_cells") != matrix_info.get("total_cells")
        or matrix.get("execution_id") != execution_id
        or matrix.get("execution_config_sha256") != config_sha
    ):
        raise V2AnalysisError("Identidade da matriz v2 é incompatível com a execução.")

    expected_hashes = manifest.get("cells_sha256")
    if not isinstance(expected_hashes, Mapping) or len(expected_hashes) != matrix_info["total_cells"]:
        raise V2AnalysisError("Índice de hashes das células v2 está ausente ou incompleto.")
    cells_dir = root / "cells"
    actual_ids = {path.stem for path in cells_dir.glob("*.json")}
    if actual_ids != set(expected_hashes):
        raise V2AnalysisError("Células presentes não correspondem ao manifesto v2.")
    matrix_cells = {
        str(cell.get("cell_id")): cell for cell in matrix.get("cells", [])
        if isinstance(cell, Mapping)
    }
    if set(matrix_cells) != set(expected_hashes):
        raise V2AnalysisError("A matriz v2 não cobre exatamente as células seladas.")

    records: list[V2MetricRecord] = []
    for cell_id in sorted(expected_hashes):
        path = cells_dir / f"{cell_id}.json"
        if _sha256(path) != expected_hashes[cell_id]:
            raise V2AnalysisError(f"Hash divergente para a célula v2 {cell_id}.")
        payload = _read_json(path)
        cell = payload.get("cell")
        individual = payload.get("result", {}).get("individual")
        if payload.get("status") != "completed" or not isinstance(cell, Mapping) or not isinstance(individual, Mapping):
            raise V2AnalysisError(f"Célula v2 {cell_id} não está completa.")
        if cell.get("cell_id") != cell_id or dict(cell) != matrix_cells[cell_id]:
            raise V2AnalysisError(f"Célula v2 {cell_id} diverge do manifesto da matriz.")
        if cell.get("execution_id") != execution_id or cell.get("execution_config_sha256") != config_sha:
            raise V2AnalysisError(f"Célula v2 {cell_id} pertence a outra execução.")
        if any(individual.get(field) != cell.get(field) for field in _CELL_FIELDS):
            raise V2AnalysisError(f"Metadados da célula v2 {cell_id} divergem do resultado.")
        partition = individual.get("partition")
        if partition not in {"validation", "holdout"}:
            raise V2AnalysisError(f"Célula v2 {cell_id} não declara uma partição válida.")
        scenario_id = individual.get("policy_scenario_id")
        scenario_hash = individual.get("policy_scenario_hash")
        if not isinstance(scenario_id, str) or not scenario_id or not isinstance(scenario_hash, str) or not scenario_hash:
            raise V2AnalysisError(f"Célula v2 {cell_id} não declara o cenário de política.")
        raw_metric = individual.get("metrics", {}).get("ndcg_at_k")
        metric = float(raw_metric) if isinstance(raw_metric, (int, float)) else None
        trajectory_sha256 = payload.get("result", {}).get("trajectory_sha256")
        if exposure_policy == "shared-seeded-v2.1" and (
            not isinstance(trajectory_sha256, str) or len(trajectory_sha256) != 64
        ):
            raise V2AnalysisError(f"Célula v2.1 {cell_id} não declara hash de trajetória válido.")
        records.append(V2MetricRecord(
            cell_id=cell_id,
            comparison_id=str(cell.get("comparison_id", "")),
            execution_id=execution_id,
            execution_config_sha256=config_sha,
            method=str(cell.get("method", "")),
            condition=str(cell.get("condition", "")),
            profile=str(cell.get("profile", "")),
            persona=str(cell.get("persona", "")),
            seed=int(cell.get("seed")),
            run=int(cell.get("run")),
            k=int(cell.get("k")),
            agent_id=str(individual.get("agent_id", "")),
            candidate_set_id=str(individual.get("candidate_set_id", "")),
            relevance_id=str(individual.get("relevance_id", "")),
            partition=str(partition),
            scenario_id=scenario_id,
            scenario_hash=scenario_hash,
            profile_noise_level=str(individual.get("profile_noise_level", "")),
            exposure_policy=str(individual.get("exposure_policy", "")),
            trajectory_status=str(individual.get("trajectory_status", "")),
            trajectory_sha256=str(trajectory_sha256) if trajectory_sha256 is not None else None,
            status=str(individual.get("evaluation_status", "")),
            metric=metric,
        ))
    return manifest, records


def _paired_rows(
    records: Sequence[V2MetricRecord], *, selected_scenario_id: str | None,
    selected_scenario_hash: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected = [record for record in records if record.method in {"B4", "B5"} and record.k == 5]
    grouped: dict[tuple[str, str], dict[str, V2MetricRecord]] = defaultdict(dict)
    invalid = 0
    for record in selected:
        if record.status != "evaluated" or record.metric is None:
            invalid += 1
            continue
        key = (record.partition, record.comparison_id)
        if record.method in grouped[key]:
            raise V2AnalysisError(f"Método duplicado no par v2 {record.comparison_id}.")
        grouped[key][record.method] = record

    rows: list[dict[str, Any]] = []
    incomplete = 0
    for (partition, comparison_id), pair in sorted(grouped.items()):
        if set(pair) != {"B4", "B5"}:
            incomplete += 1
            continue
        b4, b5 = pair["B4"], pair["B5"]
        if tuple(getattr(b4, field) for field in _PAIR_FIELDS) != tuple(getattr(b5, field) for field in _PAIR_FIELDS):
            raise V2AnalysisError(f"Pareamento incompatível em {comparison_id}.")
        if b4.exposure_policy not in {"closed-loop-v2", "shared-seeded-v2.1"} or b4.trajectory_status not in {"completed", "exhausted"}:
            raise V2AnalysisError(f"Trajetória incompatível em {comparison_id}.")
        if partition == "holdout":
            if not selected_scenario_id:
                raise V2AnalysisError("Holdout v2 exige uma seleção de validação congelada.")
            if b4.scenario_id != selected_scenario_id:
                raise V2AnalysisError("O holdout v2 contém cenário diferente da seleção congelada.")
            if selected_scenario_hash and b4.scenario_hash != selected_scenario_hash:
                raise V2AnalysisError("O holdout v2 contém hash diferente da seleção congelada.")
        rows.append({
            "comparison_id": comparison_id,
            "partition": partition,
            "scenario_id": b4.scenario_id,
            "scenario_hash": b4.scenario_hash,
            "agent_id": b4.agent_id,
            "persona": b4.persona,
            "seed": b4.seed,
            "condition": b4.condition,
            "profile": b4.profile,
            "run": b4.run,
            "k": b4.k,
            "candidate_set_id": b4.candidate_set_id,
            "relevance_id": b4.relevance_id,
            "profile_noise_level": b4.profile_noise_level,
            "exposure_policy": b4.exposure_policy,
            "trajectory_status": b4.trajectory_status,
            "trajectory_sha256": b4.trajectory_sha256,
            "b4_ndcg_at_5": float(b4.metric),
            "b5_ndcg_at_5": float(b5.metric),
            "difference_b5_minus_b4": float(b5.metric) - float(b4.metric),
        })
    if not rows:
        raise V2AnalysisError("Nenhum par B5 × B4 válido foi encontrado na execução v2.")
    return rows, {
        "selected_records": len(selected),
        "valid_pairs": len(rows),
        "invalid_metric_or_status": invalid,
        "incomplete_pairs": incomplete,
    }


def _subgroup_summaries(
    pairs: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int, bootstrap_seed: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for partition in ("validation", "holdout"):
        partition_rows = [row for row in pairs if row["partition"] == partition]
        for dimension in _DIMENSIONS:
            family: list[dict[str, Any]] = []
            categories = sorted({str(row[dimension]) for row in partition_rows})
            for category in categories:
                observations = [row for row in partition_rows if str(row[dimension]) == category]
                by_agent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
                for row in observations:
                    by_agent[str(row["agent_id"])].append(row)
                b4 = [sum(float(item["b4_ndcg_at_5"]) for item in values) / len(values) for _, values in sorted(by_agent.items())]
                b5 = [sum(float(item["b5_ndcg_at_5"]) for item in values) / len(values) for _, values in sorted(by_agent.items())]
                inference = paired_inference(b4, b5, resamples=bootstrap_resamples, seed=bootstrap_seed)
                family.append({
                    "partition": partition,
                    "dimension": dimension,
                    "category": category,
                    "analysis_status": "exploratory",
                    "unit_of_analysis": "independent_synthetic_agent",
                    "n_agents": len(b4),
                    "n_pairs": len(observations),
                    "b4_mean": describe(b4)["mean"],
                    "b5_mean": describe(b5)["mean"],
                    "difference_mean": inference["mean_difference"],
                    "difference_median": describe([right - left for left, right in zip(b4, b5)])["median"],
                    "ci95_low": inference["confidence_interval_95"][0],
                    "ci95_high": inference["confidence_interval_95"][1],
                    "wilcoxon_statistic": inference["wilcoxon_statistic"],
                    "p_value_raw": inference["p_value_two_sided"],
                    "rank_biserial_correlation": inference["rank_biserial_correlation"],
                    "positive": inference["positive"],
                    "negative": inference["negative"],
                    "ties": inference["ties"],
                })
            adjusted = holm_adjust([float(row["p_value_raw"]) for row in family])
            for row, corrected in zip(family, adjusted):
                row["p_value_holm"] = corrected
                row["holm_family"] = f"v2:{partition}:{dimension}"
            summaries.extend(family)
    return summaries


def _partition_overview(pairs: Sequence[Mapping[str, Any]], *, bootstrap_resamples: int, bootstrap_seed: int) -> list[dict[str, Any]]:
    overview = []
    for partition in ("validation", "holdout"):
        rows = [row for row in pairs if row["partition"] == partition]
        if not rows:
            continue
        by_agent: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_agent[str(row["agent_id"])].append(row)
        b4 = [sum(float(item["b4_ndcg_at_5"]) for item in values) / len(values) for _, values in sorted(by_agent.items())]
        b5 = [sum(float(item["b5_ndcg_at_5"]) for item in values) / len(values) for _, values in sorted(by_agent.items())]
        inference = paired_inference(b4, b5, resamples=bootstrap_resamples, seed=bootstrap_seed)
        overview.append({
            "partition": partition,
            "analysis_status": "exploratory_selection" if partition == "validation" else "exploratory_complementary_holdout",
            "n_agents": len(b4),
            "n_pairs": len(rows),
            "b4": describe(b4),
            "b5": describe(b5),
            "difference_b5_minus_b4": inference,
        })
    return overview


def _design_comparison(v1_report: Mapping[str, Any] | None, v2_overview: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    v2_holdout = next((row for row in v2_overview if row["partition"] == "holdout"), None)
    if v1_report is None:
        return {
            "status": "v1_source_not_supplied",
            "pooling": "prohibited",
            "reason": "A execução oficial v1.1 deve ser fornecida explicitamente e permanece separada da v2.",
            "v2_holdout": v2_holdout,
        }
    v1_effect = v1_report["paired_difference_b5_minus_b4"]
    return {
        "status": "reported_separately",
        "pooling": "prohibited",
        "design_difference": {
            "v1_1": "neutral exposure with profile aligned to latent preference",
            "v2": "closed-loop exposure with noisy declared profile and validation-only policy selection",
        },
        "v1_1_primary": {
            "matrix_id": v1_report["matrix_id"],
            "n_agents": v1_report["accounting"]["agent_units"],
            "difference_b5_minus_b4": v1_effect,
        },
        "v2_holdout": v2_holdout,
    }


def analyze_v2_execution(
    manifest_path: Path,
    *,
    selected_scenario_id: str | None = None,
    selection_path: Path | None = None,
    v1_manifest_path: Path | None = None,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 42,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Analisa B5 × B4 por partição e pressuposto, mantendo v1.1 separado."""
    manifest_path = Path(manifest_path).resolve()
    manifest, records = load_v2_records(manifest_path)
    selected_scenario_hash = None
    locked_selection = False
    if selection_path is not None:
        config = (
            load_policy_sensitivity_config(
                DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH,
                schema_path=DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH,
            )
            if manifest["execution"].get("protocol_version") == "protocol-v2.1"
            else load_policy_sensitivity_config()
        )
        try:
            selection = load_frozen_validation_selection(config, Path(selection_path))
        except Exception as exc:
            raise V2AnalysisError(f"Seleção de validação congelada inválida: {exc}") from exc
        frozen_scenario_id = str(selection["selected_scenario_id"])
        if selected_scenario_id and selected_scenario_id != frozen_scenario_id:
            raise V2AnalysisError("Cenário informado diverge da seleção de validação congelada.")
        selected_scenario_id = frozen_scenario_id
        selected_scenario_hash = str(selection["selected_scenario_hash"])
        locked_selection = True
    pairs, accounting = _paired_rows(
        records,
        selected_scenario_id=selected_scenario_id,
        selected_scenario_hash=selected_scenario_hash,
    )
    summaries = _subgroup_summaries(
        pairs, bootstrap_resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed,
    )
    overview = _partition_overview(
        pairs, bootstrap_resamples=bootstrap_resamples, bootstrap_seed=bootstrap_seed,
    )
    v1_report = None
    if v1_manifest_path is not None:
        try:
            v1_report, _ = analyze_official_results(Path(v1_manifest_path))
        except Exception as exc:
            raise V2AnalysisError(f"Não foi possível carregar a fonte v1.1 separada: {exc}") from exc
    report = {
        "schema_version": "1.0",
        "analysis": "v2_ecological_robustness_b5_vs_b4",
        "execution": dict(manifest["execution"]),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "matrix_id": manifest["matrix"]["matrix_id"],
        "metric": "ndcg_at_k",
        "k": 5,
        "unit_of_analysis": "independent_synthetic_agent",
        "selection": {
            "selected_scenario_id": selected_scenario_id,
            "selected_scenario_hash": selected_scenario_hash,
            "locked_validation_selection": locked_selection,
            "validation_and_holdout_separated": True,
        },
        "accounting": accounting,
        "partition_overview": overview,
        "multiplicity": {
            "method": "Holm step-down",
            "families": "one family per v2 partition and analysis dimension",
            "alpha": 0.05,
        },
        "design_comparison": _design_comparison(v1_report, overview),
        "guardrails": [
            "v1.1 and v2 are reported separately",
            "v2 estimates describe synthetic agents rather than human participants",
            "validation is used for policy selection; holdout is evaluated only after selection",
            "negative, null, invalid, and incomplete outcomes remain accounted for",
            "the v1.1 and v2 effects are not pooled or treated as a meta-analysis",
        ],
    }
    return report, pairs, summaries


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise V2AnalysisError(f"Não há linhas para {path.name}.")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_figure(path: Path, draw) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.0, 4.2))
    draw(axis)
    figure.tight_layout()
    outputs = []
    for suffix in (".pdf", ".png"):
        target = path.with_suffix(suffix)
        figure.savefig(target, dpi=220 if suffix == ".png" else None, bbox_inches="tight")
        outputs.append(target)
    plt.close(figure)
    return outputs


def _write_figures(pairs: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]], directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    effects = [row for row in summaries if row["dimension"] == "scenario_id"]
    def policy(axis):
        labels = [f"{row['partition']}:{row['category']}" for row in effects]
        values = [float(row["difference_mean"]) for row in effects]
        lower = [value - float(row["ci95_low"]) for value, row in zip(values, effects)]
        upper = [float(row["ci95_high"]) - value for value, row in zip(values, effects)]
        positions = list(range(len(effects)))
        axis.errorbar(values, positions, xerr=[lower, upper], fmt="o", color="#0072B2", capsize=3)
        axis.axvline(0.0, color="#666666", linestyle="--", linewidth=1)
        axis.set_yticks(positions, labels)
        axis.set_xlabel("Paired B5 − B4 difference in NDCG@5")
        axis.set_title("Policy sensitivity by partition")
    outputs.extend(_write_figure(directory / "policy_sensitivity", policy))

    trajectories = [row for row in summaries if row["dimension"] == "condition"]
    def temporal(axis):
        for partition, marker in (("validation", "o"), ("holdout", "s")):
            rows = [row for row in trajectories if row["partition"] == partition]
            rows.sort(key=lambda row: row["category"])
            if rows:
                axis.plot([row["category"] for row in rows], [row["difference_mean"] for row in rows], marker=marker, label=partition)
        axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
        axis.set_xlabel("Feedback condition")
        axis.set_ylabel("Mean paired B5 − B4 NDCG@5")
        axis.set_title("Closed-loop temporal effect")
        axis.legend(frameon=False)
    outputs.extend(_write_figure(directory / "closed_loop_trajectory", temporal))

    observed = [row for row in summaries if row["dimension"] == "profile_noise_level"]
    def assumptions(axis):
        labels = [f"{row['partition']}:{row['category']}" for row in observed]
        values = [float(row["difference_mean"]) for row in observed]
        axis.bar(labels, values, color="#009E73")
        axis.axhline(0.0, color="#666666", linestyle="--", linewidth=1)
        axis.set_ylabel("Mean paired B5 − B4 NDCG@5")
        axis.set_title("Effect under observed-profile noise")
        axis.tick_params(axis="x", rotation=25)
    outputs.extend(_write_figure(directory / "observed_profile_effect", assumptions))
    return outputs


def _file_metadata(path: Path) -> dict[str, Any]:
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def write_v2_outputs(
    report: Mapping[str, Any],
    pairs: Sequence[Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> list[Path]:
    """Escreve relatório, fontes tabulares e figuras numa nova pasta atômica."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Saída existente não será sobrescrita: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "v2-analysis"
        figures = stage / "figures"
        stage.mkdir()
        report_path = stage / "v2_analysis.json"
        pairs_path = stage / "v2_paired_rows.csv"
        summaries_path = stage / "v2_subgroups.csv"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_csv(pairs_path, pairs)
        _write_csv(summaries_path, summaries)
        figure_paths = _write_figures(pairs, summaries, figures)
        generated = sorted(path for path in stage.rglob("*") if path.is_file())
        manifest_path = stage / "analysis.manifest.json"
        manifest_path.write_text(json.dumps({
            "schema_version": "1.0",
            "analysis": report["analysis"],
            "source_manifest": report["source_manifest"],
            "source_manifest_sha256": report["source_manifest_sha256"],
            "outputs": {path.relative_to(stage).as_posix(): _file_metadata(path) for path in generated},
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        shutil.move(str(stage), output_dir)
    return [output_dir / "v2_analysis.json", output_dir / "v2_paired_rows.csv", output_dir / "v2_subgroups.csv", *[output_dir / path.relative_to(figures.parent) for path in figure_paths], output_dir / "analysis.manifest.json"]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="execution.manifest.json da execução v2")
    selection_group = parser.add_mutually_exclusive_group(required=True)
    selection_group.add_argument("--selection", type=Path, help="Arquivo de seleção congelada da validação v2")
    selection_group.add_argument("--selected-scenario", help="Cenário já validado; preferir --selection em execuções científicas")
    parser.add_argument("--v1-manifest", type=Path, help="Manifesto oficial v1.1, mantido em relatório separado")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report, pairs, summaries = analyze_v2_execution(
        args.manifest,
        selected_scenario_id=args.selected_scenario,
        selection_path=args.selection,
        v1_manifest_path=args.v1_manifest,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    outputs = write_v2_outputs(report, pairs, summaries, args.output_dir)
    print(json.dumps({"matrix_id": report["matrix_id"], "outputs": [str(path) for path in outputs]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
