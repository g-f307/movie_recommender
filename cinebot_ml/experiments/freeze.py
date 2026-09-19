"""Pré-voo da issue #46; não congela nem executa resultados oficiais."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.experiments.matrix import ExperimentMatrix, load_experiment_matrix
from cinebot_ml.experimental_splits import reconstruct_assignments
from cinebot_ml.ranking.supervised import (
    SupervisedConfigError, load_supervised_config, validate_supervised_metadata,
)
from cinebot_ml.ranking.tfidf import TfidfArtifact


PILOT_METHODS = ("B0", "B1", "B2", "B3", "B4", "B5")
PILOT_CONDITIONS = ("C0", "C2", "C4")
PILOT_PROFILES = ("P0", "P4", "P5")
PILOT_SEEDS = (42, 137)


def pilot_matrix() -> ExperimentMatrix:
    return load_experiment_matrix(
        methods=PILOT_METHODS,
        conditions=PILOT_CONDITIONS,
        profiles=PILOT_PROFILES,
        personas=("consistent", "noisy"),
        seeds=PILOT_SEEDS,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def preflight(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Emite evidência de prontidão sem inferir aprovações humanas."""
    matrix = pilot_matrix()
    checks: dict[str, dict[str, Any]] = {}
    for label, relative in {
        "catalog": "data/filmes.json",
        "dataset": "datasets/movie_preferences.csv",
        "feedback": "datasets/user_feedback.csv",
        "split_manifest": "results/manifests/splits/movie_id_split.json",
        "b2_artifact": "artifacts/b2_tfidf_v1.json",
        "b3_model": "artifacts/production_model.joblib",
        "b3_metadata": "artifacts/model_metadata.json",
        "experimental_b3_model": "artifacts/b3_experiment_v1.joblib",
        "experimental_b3_metadata": "artifacts/b3_experiment_v1.json",
    }.items():
        path = root / relative
        checks[label] = ({"status": "present", "path": relative,
                          "bytes": path.stat().st_size, "sha256": _sha256(path)}
                         if path.is_file() else {"status": "missing", "path": relative})
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    blockers = [label for label, value in checks.items() if value["status"] == "missing"]
    split_path = root / "results/manifests/splits/movie_id_split.json"
    if split_path.is_file():
        try:
            split = json.loads(split_path.read_text(encoding="utf-8"))
            assignments = reconstruct_assignments(split)
            if split.get("source_sha256") != checks["dataset"].get("sha256"):
                raise ValueError("Hash da fonte do split diverge do dataset atual.")
            checks["split_manifest"]["contract"] = "valid"
            checks["split_manifest"]["assigned_movies"] = len(assignments)
        except (OSError, ValueError, KeyError) as exc:
            checks["split_manifest"]["contract"] = "invalid"
            checks["split_manifest"]["reason"] = str(exc)
            blockers.append("split_manifest_invalid")
    b2_path = root / "artifacts/b2_tfidf_v1.json"
    if b2_path.is_file():
        try:
            artifact = TfidfArtifact.load(b2_path)
            checks["b2_artifact"]["contract"] = "valid"
            checks["b2_artifact"]["artifact_id"] = artifact.artifact_id
        except (OSError, ValueError) as exc:
            checks["b2_artifact"]["contract"] = "invalid"
            checks["b2_artifact"]["reason"] = str(exc)
            blockers.append("b2_artifact_invalid")
    metadata_path = root / "artifacts/model_metadata.json"
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            validate_supervised_metadata(metadata, load_supervised_config())
            checks["b3_metadata"]["contract"] = "valid"
        except (OSError, ValueError, SupervisedConfigError) as exc:
            checks["b3_metadata"]["contract"] = "invalid"
            checks["b3_metadata"]["reason"] = str(exc)
            blockers.append("b3_metadata_invalid")
    # Escopo científico v1 definido pelo roadmap; revisão externa não é presumida.
    # O contrato de metadados não prova que os IDs de treino coincidem com
    # o manifesto atual; essa evidência precisa acompanhar o modelo novo.
    blockers.append("b3_split_provenance_unverified")
    units_root = root / "results/units"
    missing_units = sum(not (units_root / f"{cell.comparison_id}.json").is_file()
                        for cell in matrix.cells)
    if missing_units:
        blockers.append("paired_units_missing")
    return {
        "status": "blocked", "purpose": "preflight_only", "commit": commit,
        "method_scope": {"included": list(PILOT_METHODS), "b6": "deferred_no_distinct_method",
                         "basis": "roadmap_and_issue_47_before_holdout"},
        "review_status": "advisor_unavailable_not_approved",
        "pilot_matrix": matrix.to_manifest(), "inputs": checks,
        "missing_paired_unit_files": missing_units,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = preflight()
    encoded = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    else:
        print(encoded, end="")
    return 2 if report["blockers"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
