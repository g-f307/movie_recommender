"""Treino experimental B3 sem consultar holdout nem sobrescrever produção."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd

from cinebot_ml.config import DATASET_PATH, PROJECT_ROOT
from cinebot_ml.experimental_splits import reconstruct_assignments, sha256_file
from cinebot_ml.modeling import (build_model_candidates, build_pipeline, create_model,
                                 select_best_threshold)
from cinebot_ml.ranking.supervised import load_supervised_config, validate_supervised_metadata
from cinebot_ml.schema import FEATURE_COLUMNS


SAMPLE_MODULUS = 32  # amostragem determinística independente de rótulo e partição
SAMPLE_SEED = 2027
DEFAULT_SPLIT = PROJECT_ROOT / "results/manifests/splits/movie_id_split.json"
DEFAULT_MODEL = PROJECT_ROOT / "artifacts/b3_experiment_v1.joblib"
DEFAULT_METADATA = PROJECT_ROOT / "artifacts/b3_experiment_v1.json"


def _selected(row_number: int) -> bool:
    digest = hashlib.sha256(f"{SAMPLE_SEED}:{row_number}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % SAMPLE_MODULUS == 0


def train_b3(dataset: Path = DATASET_PATH, split_path: Path = DEFAULT_SPLIT,
             model_path: Path = DEFAULT_MODEL, metadata_path: Path = DEFAULT_METADATA) -> dict:
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError("Artefato B3 experimental já existe; não será sobrescrito.")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    if split["source_sha256"] != sha256_file(dataset):
        raise ValueError("Dataset diverge da fonte do split B3.")
    assignments = reconstruct_assignments(split)
    selected = {"train": [], "validation": []}
    offset = 0
    columns = ["movie_id", "relevante", *FEATURE_COLUMNS]
    for chunk in pd.read_csv(dataset, usecols=columns, chunksize=10_000, low_memory=False):
        mask = [_selected(index) for index in range(offset, offset + len(chunk))]
        offset += len(chunk)
        sample = chunk.loc[mask].copy()
        sample["partition"] = sample["movie_id"].astype(str).map(assignments)
        if sample["partition"].isna().any():
            raise ValueError("Dataset contém movie_id ausente do split.")
        for partition in selected:
            part = sample.loc[sample["partition"] == partition, columns]
            if not part.empty:
                selected[partition].append(part)
    frames = {key: pd.concat(parts, ignore_index=True) for key, parts in selected.items()}
    for partition, frame in frames.items():
        if frame.empty or frame["relevante"].nunique() < 2:
            raise ValueError(f"A amostra {partition} não contém duas classes.")
    train, validation = frames["train"], frames["validation"]
    if set(train.movie_id).intersection(validation.movie_id):
        raise ValueError("Leakage de movie_id entre treino e validação B3.")
    candidates = []
    for family, parameter_sets in build_model_candidates().items():
        for params in parameter_sets:
            model = build_pipeline(create_model(family, params))
            model.fit(train[FEATURE_COLUMNS], train["relevante"])
            positive = list(model.classes_).index(1)
            scores = model.predict_proba(validation[FEATURE_COLUMNS])[:, positive]
            threshold, metrics = select_best_threshold(validation["relevante"], scores)
            candidates.append((metrics["f1"], metrics["precision"], metrics["average_precision"],
                               family, params, threshold, metrics))
            del model, scores
            gc.collect()
    winner = max(candidates, key=lambda row: row[:3])
    _, _, _, family, params, threshold, metrics = winner
    model = build_pipeline(create_model(family, params))
    model.fit(train[FEATURE_COLUMNS], train["relevante"])
    config = load_supervised_config()
    metadata = {
        "method_version": config.version, "winner_model": family,
        "winner_params": params, "winner_threshold": float(threshold),
        "feature_columns": list(FEATURE_COLUMNS), "positive_class": config.positive_class,
        "label_source": config.label_source, "fit_partition": "train",
        "selection_partitions": ["train", "validation"],
        "frozen_before_holdout": True,
        "split_manifest_sha256": sha256_file(split_path),
        "dataset_sha256": split["source_sha256"],
        "sample_policy": {"seed": SAMPLE_SEED, "modulus": SAMPLE_MODULUS,
                          "key": "zero_based_csv_row_number_sha256"},
        "sample_rows": {key: len(frame) for key, frame in frames.items()},
        "validation_metrics": metrics,
        "selection_policy": "max_validation_f1_then_precision_then_average_precision",
        "holdout_accessed": False,
    }
    validate_supervised_metadata(metadata, config)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    result = train_b3(args.dataset, args.split, args.model, args.metadata)
    print(json.dumps({"model": str(args.model), "selection": result["winner_model"],
                      "sample_rows": result["sample_rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
