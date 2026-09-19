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
        "b2_artifact": "artifacts/b2_tfidf_v1.json",
        "b3_model": "artifacts/production_model.joblib",
        "b3_metadata": "artifacts/model_metadata.json",
    }.items():
        path = root / relative
        checks[label] = ({"status": "present", "path": relative,
                          "bytes": path.stat().st_size, "sha256": _sha256(path)}
                         if path.is_file() else {"status": "missing", "path": relative})
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    blockers = [label for label, value in checks.items() if value["status"] == "missing"]
    # Estes estados não podem ser deduzidos de arquivos ou de métricas piloto.
    blockers.extend(("advisor_protocol_approval", "b6_decision", "frozen_b3_artifact",
                     "official_cell_runner"))
    return {
        "status": "blocked", "purpose": "preflight_only", "commit": commit,
        "pilot_matrix": matrix.to_manifest(), "inputs": checks,
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
