import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "filmes.json"
DATASET_PATH = PROJECT_ROOT / "datasets" / "movie_preferences.csv"
FEEDBACK_PATH = PROJECT_ROOT / "datasets" / "user_feedback.csv"
REFERENCE_DATA_PATH = PROJECT_ROOT / "datasets" / "reference_features.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "production_model.joblib"
MODEL_METADATA_PATH = PROJECT_ROOT / "artifacts" / "model_metadata.json"
DRIFT_REPORT_PATH = PROJECT_ROOT / "reports" / "relatorio_drift.html"
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"

MODEL_NAME = "cinebot_relevance"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB_PATH}")
DEFAULT_TOP_N = 5
ENABLE_DRIFT_ON_PREDICT = os.getenv("CINEBOT_ENABLE_DRIFT_ON_PREDICT", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "sim",
}

GENRE_LABELS = {
    "acao": "Ação",
    "comedia": "Comédia",
    "drama": "Drama",
    "terror": "Terror",
    "scifi": "Ficção Científica",
    "romance": "Romance",
    "suspense": "Suspense",
}

DECADE_LABELS = {
    "moderno": "Moderno",
    "anos_2000": "Anos 2000",
    "antes_2000": "Antes dos anos 2000",
}

POPULARITY_LABELS = {
    "popular": "Popular",
    "joia_escondida": "Joia escondida",
}

RANK_WEIGHTS = (0.50, 0.20, 0.10)
DECADE_WEIGHT = 0.10
POPULARITY_WEIGHT = 0.10
RELEVANCE_THRESHOLD = 0.6
HIGH_RATING_BONUS = 0.05
MULTI_STREAMING_BONUS = 0.05
KEYWORD_DENSITY_BONUS = 0.03
