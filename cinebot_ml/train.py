import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # pragma: no cover
    StratifiedGroupKFold = None

from cinebot_ml.config import (
    DATASET_PATH,
    DEFAULT_DATA_PATH,
    MODEL_METADATA_PATH,
    MODEL_NAME,
    MODEL_PATH,
    MLFLOW_TRACKING_URI,
    REFERENCE_DATA_PATH,
)
from cinebot_ml.dataset import build_dataset
from cinebot_ml.dataset import summarize_supervision_sources
from cinebot_ml.modeling import (
    build_model_candidates,
    build_pipeline,
    create_model,
    evaluate_probabilities,
    select_best_threshold,
)
from cinebot_ml.schema import FEATURE_COLUMNS


def build_group_cv(n_splits: int = 5):
    if StratifiedGroupKFold is not None:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return GroupKFold(n_splits=n_splits)


def summarize_cv_metrics(metrics_per_fold: list[dict[str, float]]) -> tuple[dict[str, float], dict[str, float]]:
    metric_names = metrics_per_fold[0].keys()
    means = {name: float(np.mean([fold[name] for fold in metrics_per_fold])) for name in metric_names}
    stds = {name: float(np.std([fold[name] for fold in metrics_per_fold])) for name in metric_names}
    return means, stds


def maybe_log_mlflow(run_name: str, model, metrics: dict, input_example: pd.DataFrame) -> str | None:
    try:
        import mlflow
        from mlflow.tracking import MlflowClient
    except ImportError:
        return None

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("cinebot-ranking")

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({"model_family": run_name, "features": ",".join(FEATURE_COLUMNS)})
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example.head(3),
            registered_model_name=MODEL_NAME,
        )
        return run.info.run_id


def promote_model_alias(run_id: str | None) -> None:
    if not run_id:
        return

    try:
        from mlflow.tracking import MlflowClient
    except ImportError:
        return

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    latest_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    for version in latest_versions:
        if version.run_id == run_id:
            client.set_registered_model_alias(MODEL_NAME, "production", version.version)
            break


def main() -> None:
    dataset = build_dataset(data_path=DEFAULT_DATA_PATH, output_path=DATASET_PATH)
    if dataset.empty:
        raise ValueError("Não foi possível gerar dataset de treino a partir do catálogo atual.")
    supervision_summary = summarize_supervision_sources(dataset)

    outer_split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(outer_split.split(dataset, dataset["relevante"], groups=dataset["movie_id"]))
    train_df = dataset.iloc[train_idx].reset_index(drop=True)
    test_df = dataset.iloc[test_idx].reset_index(drop=True)

    x_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["relevante"]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["relevante"]
    group_cv = build_group_cv(n_splits=5)

    results = []
    best_model = None
    best_result = None

    for model_name, candidates in build_model_candidates().items():
        best_candidate = None

        for params in candidates:
            fold_metrics = []
            fold_thresholds = []

            for cv_train_idx, cv_val_idx in group_cv.split(
                train_df,
                train_df["relevante"],
                groups=train_df["movie_id"],
            ):
                cv_train_df = train_df.iloc[cv_train_idx].reset_index(drop=True)
                cv_val_df = train_df.iloc[cv_val_idx].reset_index(drop=True)
                pipeline = build_pipeline(create_model(model_name, params))
                pipeline.fit(cv_train_df[FEATURE_COLUMNS], cv_train_df["relevante"])
                val_probabilities = pipeline.predict_proba(cv_val_df[FEATURE_COLUMNS])[:, 1]
                threshold, metrics = select_best_threshold(cv_val_df["relevante"], val_probabilities)
                fold_metrics.append(metrics)
                fold_thresholds.append(threshold)

            cv_mean_metrics, cv_std_metrics = summarize_cv_metrics(fold_metrics)
            selected_threshold = float(np.median(fold_thresholds))

            candidate_result = {
                "params": params,
                "threshold": selected_threshold,
                "metrics": cv_mean_metrics,
                "cv_metrics_std": cv_std_metrics,
                "cv_thresholds": fold_thresholds,
            }
            if best_candidate is None or (
                cv_mean_metrics["f1"],
                cv_mean_metrics["precision"],
                cv_mean_metrics.get("average_precision", 0.0),
                cv_mean_metrics["roc_auc"],
            ) > (
                best_candidate["metrics"]["f1"],
                best_candidate["metrics"]["precision"],
                best_candidate["metrics"].get("average_precision", 0.0),
                best_candidate["metrics"]["roc_auc"],
            ):
                best_candidate = candidate_result

        if best_candidate is None:
            continue

        pipeline = build_pipeline(create_model(model_name, best_candidate["params"]))
        pipeline.fit(x_train, y_train)
        test_probabilities = pipeline.predict_proba(x_test)[:, 1]
        metrics = evaluate_probabilities(y_test, test_probabilities, best_candidate["threshold"])
        predictions = (test_probabilities >= best_candidate["threshold"]).astype(int)
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1]).tolist()
        run_id = maybe_log_mlflow(model_name, pipeline, metrics, x_train)
        result = {
            "model_name": model_name,
            "metrics": metrics,
            "run_id": run_id,
            "confusion_matrix": matrix,
            "best_params": best_candidate["params"],
            "threshold": best_candidate["threshold"],
            "cv_metrics_mean": best_candidate["metrics"],
            "cv_metrics_std": best_candidate["cv_metrics_std"],
            "cv_thresholds": best_candidate["cv_thresholds"],
        }
        results.append(result)

        if best_result is None or (
            metrics["f1"],
            metrics["precision"],
            metrics.get("average_precision", 0.0),
            metrics.get("roc_auc", 0.0),
        ) > (
            best_result["metrics"]["f1"],
            best_result["metrics"]["precision"],
            best_result["metrics"].get("average_precision", 0.0),
            best_result["metrics"].get("roc_auc", 0.0),
        ):
            best_model = pipeline
            best_result = result

    if best_model is None or best_result is None:
        raise RuntimeError("Nenhum modelo foi treinado com sucesso.")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    Path("artifacts").mkdir(parents=True, exist_ok=True)

    reference_columns = [
        "sinopse",
        "genero",
        "generos_texto",
        "pref_1",
        "pref_2",
        "pref_3",
        "decade_pref",
        "popularity_pref",
        "ano",
        "nota",
        "votos",
        "duracao",
        "streaming_count",
        "sinopse_word_count",
        "keyword_count",
        "secondary_genre_count",
        "has_streaming",
        "has_poster",
        "primary_match_pref_1",
        "secondary_match_pref_2",
        "secondary_match_pref_3",
        "ranked_match_count",
        "genre_overlap_count",
        "matches_decade_pref",
        "popular_score",
        "hidden_gem_score",
        "matches_popularity_pref",
        "popularity_match_score",
    ]
    train_df[reference_columns].to_csv(REFERENCE_DATA_PATH, index=False)
    promote_model_alias(best_result["run_id"])

    metadata = {
        "trained_at": datetime.now().isoformat(),
        "model_name": MODEL_NAME,
        "winner_model": best_result["model_name"],
        "winner_metrics": best_result["metrics"],
        "winner_confusion_matrix": best_result["confusion_matrix"],
        "winner_params": best_result["best_params"],
        "winner_threshold": best_result["threshold"],
        "tracking_uri": MLFLOW_TRACKING_URI,
        "dataset_path": str(DATASET_PATH),
        "catalog_source": str(DEFAULT_DATA_PATH),
        "task_framing": {
            "supervised_target": "aderencia_ao_perfil_declarado",
            "label_description": "Classe 1 representa contexto aderente ao perfil informado; classe 0 representa contexto nao aderente.",
            "feedback_role": "Feedback real sobrescreve rotulos quando existe evidencia explicita do usuario e complementa a personalizacao em inferencia.",
        },
        "dataset_supervision_summary": supervision_summary,
        "split_strategy": "GroupShuffleSplit por movie_id no teste final + validação cruzada por grupos no treino",
        "cross_validation": {
            "strategy": "StratifiedGroupKFold" if StratifiedGroupKFold is not None else "GroupKFold",
            "n_splits": 5,
            "group_column": "movie_id",
        },
        "all_results": results,
        "challenge_mapping": {
            "student": "Gabriel de Sá",
            "project": "CineBot, curadoria de filmes",
            "suggested_ml": "Recomendação content-based com 3 gêneros ranqueados, década e popularidade escolhidas pelo usuário",
        },
    }
    with open(MODEL_METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
