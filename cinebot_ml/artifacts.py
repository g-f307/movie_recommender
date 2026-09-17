"""Preparação reproduzível dos artefatos experimentais B2 e B3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

from cinebot_ml.config import DEFAULT_DATA_PATH, PROJECT_ROOT
from cinebot_ml.dataset import load_catalog
from cinebot_ml.experimental_splits import reconstruct_assignments
from cinebot_ml.ranking import TfidfArtifact, fit_tfidf_artifact


DEFAULT_B2_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "b2_tfidf_v1.json"


class ArtifactPreparationError(ValueError):
    """Dados ou split incompatíveis com a preparação de artefatos."""


def build_b2_artifact(
    catalog: Sequence[Mapping[str, object]],
    assignments: Mapping[str, str],
    output_path: Path = DEFAULT_B2_ARTIFACT_PATH,
) -> TfidfArtifact:
    """Ajusta B2 exclusivamente nos filmes atribuídos à partição de treino."""
    training = [
        movie
        for movie in catalog
        if assignments.get(str(movie.get("id"))) == "train"
    ]
    if not training:
        raise ArtifactPreparationError("O split não contém filmes de treino presentes no catálogo.")
    non_train = sorted({value for value in assignments.values() if value not in {"train", "validation", "test"}})
    if non_train:
        raise ArtifactPreparationError(f"Partição desconhecida no split: {non_train[0]}")
    artifact = fit_tfidf_artifact(training, partition="train")
    artifact.save(output_path)
    return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build-b2",))
    parser.add_argument("--catalog", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_B2_ARTIFACT_PATH)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        assignments = reconstruct_assignments(manifest)
        artifact = build_b2_artifact(load_catalog(args.catalog), assignments, args.output)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Erro na preparação de artefatos: {exc}") from exc
    print(json.dumps({"method": "B2", "artifact_id": artifact.artifact_id, "output": args.output.as_posix()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
