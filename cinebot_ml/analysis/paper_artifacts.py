"""Publication-ready figures and LaTeX tables backed by frozen evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Iterable

from cinebot_ml.config import PROJECT_ROOT
from .statistical import analyze_official_results
from .synthesis import DEFAULT_MANIFEST, SPECIALIZED_ROOT, synthesize


class PaperArtifactError(ValueError):
    """Raised when the evidence contract is incomplete or inconsistent."""


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"

EDITORIAL_LABELS = {
    "Agente": "Agent",
    "Média e IC95%": "Mean and 95% CI",
    "Sem histórico": "Without history",
    "Sem gênero": "Without genre",
    "Contraditório": "Contradictory",
    "Sintético": "Synthetic",
    "Popularidade": "Popularity",
    "Aleatório": "Random",
    "Classificador supervisionado": "Supervised classifier",
    "TF-IDF textual": "Textual TF-IDF",
    "Perfil estático": "Static profile",
    "Perfil incremental": "Incremental profile",
    "não": "no",
    "sim": "yes",
    "não sustentada": "not supported",
    "sustentada no simulador": "supported in the simulator",
    "B5-B4 sintético": "B5-B4 synthetic",
    "method": "Method",
    "description": "Description",
    "feedback": "Feedback",
    "contrast": "Contrast",
    "estimate": "Estimate",
    "ci95": "95% CI",
    "p_value": "p-value",
    "n": "N",
    "rq": "RQ",
    "hypothesis": "Hypothesis",
    "decision": "Decision",
}


def _editorial_label(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    translated = EDITORIAL_LABELS.get(value, value)
    return translated.replace("_", " ").capitalize() if "_" in translated else translated


def _validate(evidence: dict[str, Any]) -> None:
    if not str(evidence.get("matrix_id", "")).strip():
        raise PaperArtifactError("matrix_id is required")
    if not evidence.get("sources"):
        raise PaperArtifactError("sources are required")


def _plot(path: Path, draw: Callable[[Any, Any], None]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    height = 7.5 if path.name == "mechanisms_forest" else 3.8
    figure, axis = plt.subplots(figsize=(6.4, height))
    draw(figure, axis)
    figure.tight_layout()
    metadata = {"Creator": "movie_recommender", "Producer": "movie_recommender",
                "CreationDate": None, "ModDate": None}
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", metadata=metadata)
    plt.close(figure)


def _primary(evidence: dict[str, Any], axis: Any) -> None:
    values = evidence["primary"]["agent_differences"]
    axis.scatter(values, range(1, len(values) + 1), color=BLUE, marker="o", label="Agent")
    mean = evidence["primary"]["mean_difference"]
    low, high = evidence["primary"]["confidence_interval_95"]
    axis.errorbar(mean, 0.35, xerr=[[mean - low], [high - mean]], color=ORANGE,
                  marker="D", capsize=4, label="Mean and 95% CI")
    axis.axvline(0, color=GRAY, linewidth=1, linestyle="--")
    axis.set(xlabel="Paired B5 − B4 difference in NDCG@5", ylabel="Agent")
    axis.legend(frameon=False)


def _trajectory(evidence: dict[str, Any], axis: Any) -> None:
    rows = evidence["convergence"]
    labels = [row["condition"] for row in rows]
    axis.plot(labels, [row["b4_mean"] for row in rows], color=BLUE, marker="o", label="B4")
    axis.plot(labels, [row["b5_mean"] for row in rows], color=ORANGE, marker="s", label="B5")
    axis.set(xlabel="Feedback condition", ylabel="Mean NDCG@5")
    axis.legend(frameon=False)


def _forest(rows: list[dict[str, Any]], axis: Any, xlabel: str) -> None:
    positions = list(range(len(rows), 0, -1))
    for position, row in zip(positions, rows):
        low, high = row["ci95"]
        value = row["estimate"]
        axis.errorbar(value, position, xerr=[[value - low], [high - value]], color=BLUE,
                      marker="o", capsize=4)
    axis.axvline(0, color=GRAY, linewidth=1, linestyle="--")
    axis.set_yticks(positions, [
        _editorial_label(row.get("label", row.get("source"))) for row in rows
    ])
    axis.set_xlabel(xlabel)


def _escape(value: Any) -> str:
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("_", r"\_"), ("#", r"\#")):
        text = text.replace(old, new)
    return text


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, list):
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    return _escape(value)


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise PaperArtifactError(f"table {path.stem} has no rows")
    columns = list(rows[0])
    lines = [r"\begin{tabular}{" + "l" * len(columns) + "}", r"\toprule",
             " & ".join(_escape(_editorial_label(column)) for column in columns) + r" \\", r"\midrule"]
    lines.extend(" & ".join(
        _format_value(_editorial_label(row.get(column, ""))) for column in columns
    ) + r" \\"
                 for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _digest(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def render_paper_artifacts(evidence: dict[str, Any], output_root: Path) -> list[Path]:
    """Render a complete, traceable artifact bundle without overwriting outputs."""
    _validate(evidence)
    output_root = Path(output_root)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_root.parent) as temporary:
        stage = Path(temporary) / "paper"
        figures = stage / "figures"
        tables = stage / "tables"
        figures.mkdir(parents=True)
        tables.mkdir()

        _plot(figures / "primary_paired", lambda _, axis: _primary(evidence, axis))
        _plot(figures / "feedback_trajectory", lambda _, axis: _trajectory(evidence, axis))
        mechanisms = [*evidence["ablation"], *evidence["robustness"]]
        _plot(figures / "mechanisms_forest",
              lambda _, axis: _forest(mechanisms, axis, "Effect on NDCG@5"))
        _plot(figures / "transportability",
              lambda _, axis: _forest(evidence["transportability"], axis,
                                      "B5 − B4 difference in NDCG@5"))

        table_names = {
            "design": "design.tex",
            "hypotheses": "hypothesis_decisions.tex",
            "contrasts": "contrasts.tex",
        }
        for key, filename in table_names.items():
            _write_table(tables / filename, evidence["tables"][key])

        generated = sorted(path for path in stage.rglob("*") if path.is_file())
        outputs = {path.relative_to(stage).as_posix(): _digest(path) for path in generated}
        manifest = {
            "schema_version": "1.0",
            "evidence_schema_version": evidence.get("schema_version"),
            "matrix_id": evidence["matrix_id"],
            "sources": evidence["sources"],
            "outputs": outputs,
        }
        (stage / "paper_artifacts.manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_root.exists():
            output_root.rmdir()
        shutil.move(str(stage), output_root)

    return sorted(path for path in output_root.rglob("*") if path.is_file())


_METHODS = [
    {"method": "B0", "description": "Popularidade", "feedback": "não"},
    {"method": "B1", "description": "Aleatório", "feedback": "não"},
    {"method": "B2", "description": "Classificador supervisionado", "feedback": "não"},
    {"method": "B3", "description": "TF-IDF textual", "feedback": "não"},
    {"method": "B4", "description": "Perfil estático", "feedback": "não"},
    {"method": "B5", "description": "Perfil incremental", "feedback": "sim"},
]


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise PaperArtifactError(f"missing source: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), **_digest(path)}


def build_paper_evidence(manifest_path: Path = DEFAULT_MANIFEST,
                         specialized_root: Path = SPECIALIZED_ROOT) -> dict[str, Any]:
    """Load only frozen structured results and normalize their editorial contract."""
    manifest_path = Path(manifest_path)
    manifest = _read_json(manifest_path)
    matrix_id = manifest.get("matrix_id")
    confirmatory, pairs = analyze_official_results(manifest_path)
    synthesis, _, convergence_rows = synthesize(manifest_path, specialized_root=specialized_root)
    if confirmatory.get("matrix_id") != matrix_id or synthesis.get("matrix_id") != matrix_id:
        raise PaperArtifactError("incompatible matrix identity")

    roots = Path(specialized_root)
    contrast_path = roots / "contrasts" / matrix_id / "contrasts.json"
    ablation_path = roots / "ablation" / "538fae9a59ad7976c95c6ef8" / "comparisons.json"
    ablation_manifest = ablation_path.with_name("ablation.manifest.json")
    robustness_path = roots / "robustness" / "56e31e9e624c565ed60a3a64" / "scenario_report.json"
    robustness_manifest = robustness_path.with_name("robustness.manifest.json")
    external_root = roots / "external_validation" / "external-validation-movielens-100k-v1"
    transport_path = external_root / "transportability.json"
    external_path = external_root / "external_report.json"
    contrasts = _read_json(contrast_path)
    if contrasts.get("matrix_id") != matrix_id:
        raise PaperArtifactError("incompatible contrast matrix identity")
    ablation = _read_json(ablation_path)
    robustness = _read_json(robustness_path)
    transport = _read_json(transport_path)
    external = _read_json(external_path)

    primary = confirmatory["paired_difference_b5_minus_b4"]
    convergence = []
    for condition in [f"C{i}" for i in range(6)]:
        selected = {row["method"]: row for row in convergence_rows if row["condition"] == condition}
        convergence.append({"condition": condition, "b4_mean": selected["B4"]["mean_ndcg_at_k"],
                            "b5_mean": selected["B5"]["mean_ndcg_at_k"]})
    ablation_rows = [{"label": row["variant"] + ": " + ", ".join(row["removed_components"]),
                      "estimate": row["mean_difference"], "ci95": row["confidence_interval_95"]}
                     for row in ablation]
    robustness_rows = [{"label": row["scenario"], "estimate": row["mean_difference"],
                        "ci95": row["confidence_interval_95"]} for row in robustness]
    transport_rows = [
        {"source": "Sintético", "estimate": transport["synthetic"]["mean_difference"],
         "ci95": transport["synthetic"]["confidence_interval_95"], "n": transport["synthetic"]["n"]},
        {"source": "MovieLens 100K", "estimate": transport["public"]["mean_difference"],
         "ci95": transport["public"]["confidence_interval_95"], "n": transport["public"]["n"]},
    ]
    h3 = contrasts["results"]["h3_k5"]["summary"]["difference"]
    h4 = contrasts["results"]["h4_post_feedback_k5"]["summary"]["difference"]
    hypotheses = [
        {"rq": "RQ1", "hypothesis": "H1", "estimate": primary["mean_difference"],
         "ci95": primary["confidence_interval_95"], "decision": "não sustentada"},
        {"rq": "RQ4", "hypothesis": "H3", "estimate": h3["mean_difference"],
         "ci95": h3["confidence_interval_95"], "decision": "sustentada no simulador"},
        {"rq": "RQ4", "hypothesis": "H4", "estimate": h4["mean_difference"],
         "ci95": h4["confidence_interval_95"], "decision": "sustentada no simulador"},
    ]
    contrast_rows = [
        {"contrast": "B5-B4 sintético", "estimate": primary["mean_difference"],
         "ci95": primary["confidence_interval_95"], "p_value": primary["p_value_two_sided"], "n": primary["n"]},
        {"contrast": "P5-P0", "estimate": h3["mean_difference"], "ci95": h3["confidence_interval_95"],
         "p_value": h3["p_value_two_sided"], "n": h3["n"]},
        {"contrast": "B5-B0", "estimate": h4["mean_difference"], "ci95": h4["confidence_interval_95"],
         "p_value": h4["p_value_two_sided"], "n": h4["n"]},
        {"contrast": "B5-B4 MovieLens", "estimate": external["contrasts"]["B5_minus_B4"]["mean_difference"],
         "ci95": external["contrasts"]["B5_minus_B4"]["confidence_interval_95"],
         "p_value": external["contrasts"]["B5_minus_B4"]["p_value_two_sided"], "n": external["eligible_users"]},
    ]
    paths = {"official_manifest": manifest_path, "contrasts": contrast_path,
             "ablation": ablation_path, "ablation_manifest": ablation_manifest,
             "robustness": robustness_path, "robustness_manifest": robustness_manifest,
             "transportability": transport_path, "external_report": external_path}
    return {"schema_version": "1.0", "matrix_id": matrix_id,
            "primary": {"agent_differences": [row["difference_b5_minus_b4"] for row in pairs],
                        "mean_difference": primary["mean_difference"],
                        "confidence_interval_95": primary["confidence_interval_95"],
                        "p_value": primary["p_value_two_sided"],
                        "rank_biserial": primary["rank_biserial_correlation"], "n": primary["n"]},
            "convergence": convergence, "ablation": ablation_rows, "robustness": robustness_rows,
            "transportability": transport_rows,
            "tables": {"design": _METHODS, "hypotheses": hypotheses, "contrasts": contrast_rows},
            "sources": {name: _source(path) for name, path in paths.items()}}


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--specialized-root", type=Path, default=SPECIALIZED_ROOT)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results/paper_artifacts_v1")
    args = parser.parse_args(list(argv) if argv is not None else None)
    evidence = build_paper_evidence(args.manifest, args.specialized_root)
    outputs = render_paper_artifacts(evidence, args.output_root)
    print(json.dumps({"matrix_id": evidence["matrix_id"], "outputs": [str(p) for p in outputs]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
