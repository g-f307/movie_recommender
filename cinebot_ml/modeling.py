from collections.abc import Mapping

import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.svm import LinearSVC

from cinebot_ml.schema import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS


TEXT_COLUMNS = {
    "sinopse": ("sinopse", TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words=None)),
    "perfil": ("ranked_profile_text", TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words=None)),
    "generos": ("generos_texto", TfidfVectorizer(max_features=250, ngram_range=(1, 2), stop_words=None)),
}


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("sinopse_tfidf", TEXT_COLUMNS["sinopse"][1], TEXT_COLUMNS["sinopse"][0]),
            ("perfil_tfidf", TEXT_COLUMNS["perfil"][1], TEXT_COLUMNS["perfil"][0]),
            ("generos_tfidf", TEXT_COLUMNS["generos"][1], TEXT_COLUMNS["generos"][0]),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("numeric", MinMaxScaler(), NUMERIC_COLUMNS),
        ],
        remainder="drop",
    )


def build_models() -> Mapping[str, object]:
    return {
        "logistic_regression": LogisticRegression(max_iter=4000, class_weight="balanced"),
        "linear_svc": CalibratedClassifierCV(
            estimator=LinearSVC(class_weight="balanced", random_state=42),
            cv=3,
        ),
        "complement_nb": ComplementNB(alpha=0.4),
    }


def build_model_candidates() -> Mapping[str, list[dict]]:
    return {
        "logistic_regression": [
            {"C": 0.5},
            {"C": 1.0},
            {"C": 2.0},
        ],
        "linear_svc": [
            {"C": 0.5},
            {"C": 1.0},
            {"C": 2.0},
        ],
        "complement_nb": [
            {"alpha": 0.2},
            {"alpha": 0.4},
            {"alpha": 0.8},
        ],
    }


def create_model(model_name: str, params: dict | None = None) -> object:
    params = params or {}
    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=4000,
            class_weight="balanced",
            C=params.get("C", 1.0),
        )
    if model_name == "linear_svc":
        return CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                random_state=42,
                C=params.get("C", 1.0),
            ),
            cv=3,
        )
    if model_name == "complement_nb":
        return ComplementNB(alpha=params.get("alpha", 0.4))
    raise ValueError(f"Modelo desconhecido: {model_name}")


def build_pipeline(model: object) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def evaluate_model(model: Pipeline, features, target) -> dict[str, float]:
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
    }

    if probabilities is not None:
        metrics["roc_auc"] = roc_auc_score(target, probabilities)
        metrics["average_precision"] = average_precision_score(target, probabilities)

    return metrics


def evaluate_probabilities(target, probabilities, threshold: float) -> dict[str, float]:
    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(target, predictions),
        "precision": precision_score(target, predictions, zero_division=0),
        "recall": recall_score(target, predictions, zero_division=0),
        "f1": f1_score(target, predictions, zero_division=0),
        "roc_auc": roc_auc_score(target, probabilities),
        "average_precision": average_precision_score(target, probabilities),
    }
    return metrics


def build_precision_recall_points(target, probabilities) -> dict[str, list[float]]:
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
    }


def build_roc_points(target, probabilities) -> dict[str, list[float]]:
    fpr, tpr, thresholds = roc_curve(target, probabilities)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thresholds.tolist(),
    }


def select_best_threshold(target, probabilities) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_metrics = evaluate_probabilities(target, probabilities, best_threshold)

    for threshold in np.arange(0.3, 0.71, 0.05):
        metrics = evaluate_probabilities(target, probabilities, float(threshold))
        if (
            metrics["f1"],
            metrics["precision"],
            metrics["recall"],
            metrics["roc_auc"],
        ) > (
            best_metrics["f1"],
            best_metrics["precision"],
            best_metrics["recall"],
            best_metrics["roc_auc"],
        ):
            best_threshold = float(threshold)
            best_metrics = metrics

    return best_threshold, best_metrics
