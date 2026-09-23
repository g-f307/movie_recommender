"""Geração determinística dos artefatos científicos usados no artigo."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from cinebot_ml.config import PROJECT_ROOT

from .loader import MetricRecord, load_official_records
from .statistical import analyze_official_results
from .subgroups import analyze_subgroups
from .synthesis import DEFAULT_MANIFEST, SPECIALIZED_ROOT, synthesize


class ReportingError(ValueError):
    """Indica entrada ou destino incompatível com um relatório rastreável."""


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ReportingError("Grupo sem valores válidos.")
    return sum(values) / len(values)


def general_method_table(records: Sequence[MetricRecord]) -> list[dict[str, Any]]:
    """Tabela B0--B5 em NDCG@5, agregada primeiro dentro do agente."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in records:
        if row.k == 5 and row.status == "evaluated" and row.metric is not None:
            grouped[(row.method, row.agent_id)].append(float(row.metric))
    rows = []
    for method in [f"B{index}" for index in range(6)]:
        units = [_mean(values) for (candidate, _), values in grouped.items() if candidate == method]
        if not units:
            raise ReportingError(f"Método ausente na tabela geral: {method}.")
        rows.append({
            "method": method, "metric": "ndcg_at_k", "k": 5,
            "relevance_source": "synthetic_user", "n_independent_agents": len(units),
            "mean": _mean(units), "minimum": min(units), "maximum": max(units),
        })
    return rows


def inference_table(confirmatory: dict[str, Any], synthesis: dict[str, Any]) -> list[dict[str, Any]]:
    paired = confirmatory["paired_difference_b5_minus_b4"]
    convergence = synthesis["exploratory"]["convergence"]["inference"]
    return [{
        "analysis": "confirmatory_b5_vs_b4", "metric": "ndcg_at_k", "k": 5,
        "test": "Wilcoxon paired two-sided", "n_units": confirmatory["accounting"]["agent_units"],
        "estimate": paired["mean_difference"],
        "ci95_low": paired["confidence_interval_95"][0],
        "ci95_high": paired["confidence_interval_95"][1],
        "p_value": paired["p_value_two_sided"],
        "effect_size": paired["rank_biserial_correlation"],
        "effect_size_name": "paired_rank_biserial_correlation",
    }, {
        "analysis": "exploratory_b5_convergence", "metric": "ndcg_at_k", "k": 5,
        "test": "Friedman", "n_units": convergence["n_longitudinal_units"],
        "estimate": convergence["statistic"], "ci95_low": None, "ci95_high": None,
        "p_value": convergence["p_value"], "effect_size": None,
        "effect_size_name": None,
    }]


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ReportingError(f"Tabela vazia: {path.name}.")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _plot_line(rows: Sequence[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(8, 5))
    for method in ("B4", "B5"):
        selected = [row for row in rows if row["method"] == method]
        axis.plot([row["condition"] for row in selected],
                  [row["mean_ndcg_at_k"] for row in selected], marker="o", label=method)
    axis.set(title="Convergência por feedback — agentes sintéticos",
             xlabel="Condição (C0–C5)", ylabel="NDCG@5")
    axis.legend(title="Método"); axis.grid(alpha=.25); figure.tight_layout()
    figure.savefig(output, dpi=180); plt.close(figure)


def _plot_cold(rows: Sequence[dict[str, Any]], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(9, 5.5))
    profiles = [f"P{index}" for index in range(6)]
    for method in [f"B{index}" for index in range(6)]:
        values = [next(row["mean_ndcg_at_k"] for row in rows
                       if row["method"] == method and row["profile"] == profile)
                  for profile in profiles]
        axis.plot(profiles, values, marker="o", label=method)
    axis.set(title="Cold start por informação disponível — agentes sintéticos",
             xlabel="Perfil P0–P5", ylabel="NDCG@5")
    axis.legend(title="Método", ncol=3); axis.grid(alpha=.25); figure.tight_layout()
    figure.savefig(output, dpi=180); plt.close(figure)


def _plot_unavailable(title: str, reason: str, labels: Sequence[str], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(10, 4.8))
    axis.axis("off")
    axis.set_title(f"{title} — agentes sintéticos", pad=20)
    axis.text(.5, .62, "Resultados indisponíveis (N/D)", ha="center", va="center",
              fontsize=16, weight="bold", transform=axis.transAxes)
    axis.text(.5, .43, reason, ha="center", va="center", wrap=True,
              transform=axis.transAxes)
    axis.text(.5, .18, "Configuração prevista: " + ", ".join(labels), ha="center",
              va="center", wrap=True, fontsize=8, transform=axis.transAxes)
    figure.tight_layout(); figure.savefig(output, dpi=180); plt.close(figure)


def _git_commit() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def generate_report(manifest_path: Path, output_root: Path,
                    *, specialized_root: Path = SPECIALIZED_ROOT) -> list[Path]:
    """Gera o pacote completo em staging e só então publica no destino vazio."""
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Destino não está vazio e não será sobrescrito: {output_root}")
    manifest, records = load_official_records(manifest_path)
    confirmatory, pairs = analyze_official_results(manifest_path)
    subgroup_report, subgroup_rows, _ = analyze_subgroups(manifest_path)
    synthesis, cold_rows, convergence_rows = synthesize(
        manifest_path, specialized_root=specialized_root
    )
    general = general_method_table(records)
    tests = inference_table(confirmatory, synthesis)
    with tempfile.TemporaryDirectory(prefix="cinebot-report-") as temporary:
        stage = Path(temporary); tables = stage / "tables"; figures = stage / "figures"; reports = stage / "reports"
        for directory in (tables, figures, reports): directory.mkdir()
        _write_csv(tables / "methods_b0_b5.csv", general)
        _write_csv(tables / "b5_vs_b4_primary.csv", pairs)
        _write_csv(tables / "subgroups.csv", subgroup_rows)
        _write_csv(tables / "statistical_tests.csv", tests)
        _write_csv(tables / "cold_start.csv", cold_rows)
        _write_csv(tables / "convergence.csv", convergence_rows)
        _plot_line(convergence_rows, figures / "convergence.png")
        _plot_cold(cold_rows, figures / "cold_start.png")
        ablation = synthesis["exploratory"]["ablation"]
        robustness = synthesis["exploratory"]["robustness"]
        _plot_unavailable("Ablação A0–A6", ablation["reason"],
                          [row["variant"] for row in ablation["variants"]], figures / "ablation.png")
        _plot_unavailable("Robustez", robustness["reason"],
                          [row["scenario"] for row in robustness["scenarios"]], figures / "robustness.png")
        metadata = {
            "schema_version": "1.0", "report": "scientific_artifacts_v1",
            "evidence": "exploratory_synthetic", "matrix_id": manifest["matrix_id"],
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "git_commit": _git_commit(), "metric": "ndcg_at_k", "primary_k": 5,
            "confirmatory": confirmatory, "subgroups": subgroup_report,
            "synthesis": synthesis,
            "outputs": sorted(str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file()),
        }
        (reports / "scientific_report.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        metadata["outputs"] = sorted(str(path.relative_to(stage)) for path in stage.rglob("*") if path.is_file())
        (reports / "scientific_report.json").write_text(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        output_root.mkdir(parents=True, exist_ok=True)
        for source in stage.iterdir(): shutil.copytree(source, output_root / source.name)
    return sorted(path for path in output_root.rglob("*") if path.is_file())


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--specialized-root", type=Path, default=SPECIALIZED_ROOT)
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "results/scientific_report_v1")
    args = parser.parse_args(list(argv) if argv is not None else None)
    outputs = generate_report(args.manifest, args.output_root,
                              specialized_root=args.specialized_root)
    print(json.dumps({"outputs": [str(path) for path in outputs]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
