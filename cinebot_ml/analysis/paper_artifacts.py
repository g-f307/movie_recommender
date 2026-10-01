"""Publication-ready figures and LaTeX tables backed by frozen evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable


class PaperArtifactError(ValueError):
    """Raised when the evidence contract is incomplete or inconsistent."""


BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
GRAY = "#666666"


def _validate(evidence: dict[str, Any]) -> None:
    if not str(evidence.get("matrix_id", "")).strip():
        raise PaperArtifactError("matrix_id is required")
    if not evidence.get("sources"):
        raise PaperArtifactError("sources are required")


def _plot(path: Path, draw: Callable[[Any, Any], None]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.4, 3.8))
    draw(figure, axis)
    figure.tight_layout()
    metadata = {"Creator": "movie_recommender", "Producer": "movie_recommender"}
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata=metadata)
    figure.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", metadata=metadata)
    plt.close(figure)


def _primary(evidence: dict[str, Any], axis: Any) -> None:
    values = evidence["primary"]["agent_differences"]
    axis.scatter(values, range(1, len(values) + 1), color=BLUE, marker="o", label="Agente")
    mean = evidence["primary"]["mean_difference"]
    low, high = evidence["primary"]["confidence_interval_95"]
    axis.errorbar(mean, 0.35, xerr=[[mean - low], [high - mean]], color=ORANGE,
                  marker="D", capsize=4, label="Média e IC95%")
    axis.axvline(0, color=GRAY, linewidth=1, linestyle="--")
    axis.set(xlabel="Diferença pareada B5 − B4 em NDCG@5", ylabel="Agente")
    axis.legend(frameon=False)


def _trajectory(evidence: dict[str, Any], axis: Any) -> None:
    rows = evidence["convergence"]
    labels = [row["condition"] for row in rows]
    axis.plot(labels, [row["b4_mean"] for row in rows], color=BLUE, marker="o", label="B4")
    axis.plot(labels, [row["b5_mean"] for row in rows], color=ORANGE, marker="s", label="B5")
    axis.set(xlabel="Condição de feedback", ylabel="NDCG@5 médio")
    axis.legend(frameon=False)


def _forest(rows: list[dict[str, Any]], axis: Any, xlabel: str) -> None:
    positions = list(range(len(rows), 0, -1))
    for position, row in zip(positions, rows):
        low, high = row["ci95"]
        value = row["estimate"]
        axis.errorbar(value, position, xerr=[[value - low], [high - value]], color=BLUE,
                      marker="o", capsize=4)
    axis.axvline(0, color=GRAY, linewidth=1, linestyle="--")
    axis.set_yticks(positions, [row.get("label", row.get("source")) for row in rows])
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
             " & ".join(_escape(column) for column in columns) + r" \\", r"\midrule"]
    lines.extend(" & ".join(_format_value(row.get(column, "")) for column in columns) + r" \\"
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
              lambda _, axis: _forest(mechanisms, axis, "Efeito em NDCG@5"))
        _plot(figures / "transportability",
              lambda _, axis: _forest(evidence["transportability"], axis,
                                      "Diferença B5 − B4 em NDCG@5"))

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
