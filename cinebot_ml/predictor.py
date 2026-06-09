import json
from pathlib import Path

import pandas as pd

from cinebot_ml.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_TOP_N,
    DRIFT_REPORT_PATH,
    FEEDBACK_PATH,
    MODEL_METADATA_PATH,
    MODEL_NAME,
    MODEL_PATH,
    MLFLOW_TRACKING_URI,
    REFERENCE_DATA_PATH,
)
from cinebot_ml.dataset import (
    build_catalog_context,
    build_inference_frame,
    canonicalize_genre,
    canonicalize_decade_preference,
    canonicalize_popularity_preference,
    genre_overlap,
    label_for_genre,
    label_for_decade_preference,
    label_for_popularity_preference,
    load_catalog,
    load_feedback,
    movie_release_period,
    movie_popularity_bucket,
    movie_genres,
    primary_and_secondary_genres,
)
from cinebot_ml.monitoring import generate_drift_report
from cinebot_ml.schema import FEATURE_COLUMNS


class CinebotRecommender:
    def __init__(self) -> None:
        self._model = None
        self._metadata = {}

    @property
    def metadata(self) -> dict:
        if self._metadata:
            return self._metadata

        if MODEL_METADATA_PATH.exists():
            with open(MODEL_METADATA_PATH, "r", encoding="utf-8") as handle:
                self._metadata = json.load(handle)
        return self._metadata

    def load(self):
        if self._model is not None:
            return self._model

        try:
            import mlflow

            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            self._model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@production")
            return self._model
        except Exception:
            pass

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Modelo não encontrado em {MODEL_PATH}")

        import joblib

        self._model = joblib.load(MODEL_PATH)
        return self._model

    def healthcheck(self) -> dict:
        try:
            self.load()
            return {
                "status": "ok",
                "model_name": self.metadata.get("model_name", MODEL_NAME),
                "winner": self.metadata.get("winner_model"),
            }
        except Exception as exc:
            return {"status": "erro", "detail": str(exc)}

    def recommend(
        self,
        ranked_genres: list[str],
        decade_preference: str,
        popularity_preference: str,
        data_path: str | Path | None = None,
        top_n: int = DEFAULT_TOP_N,
        user_id: int | str | None = None,
    ) -> dict:
        model = self.load()
        catalog = load_catalog(Path(data_path) if data_path else DEFAULT_DATA_PATH)
        if not catalog:
            raise ValueError("Catálogo vazio ou indisponível para recomendação.")

        canonical_ranked = [canonicalize_genre(genre) for genre in ranked_genres[:3]]
        normalized_decade = canonicalize_decade_preference(decade_preference)
        normalized_popularity = canonicalize_popularity_preference(popularity_preference)
        if normalized_decade is None or normalized_popularity is None:
            raise ValueError("Década e popularidade precisam ser informadas para recomendar.")

        inference_df = build_inference_frame(
            catalog,
            canonical_ranked,
            normalized_decade,
            normalized_popularity,
        )
        probabilities = model.predict_proba(inference_df[FEATURE_COLUMNS])[:, 1]
        movie_lookup = {str(movie.get("id")): movie for movie in catalog}
        catalog_context = build_catalog_context(catalog)
        user_profile = self._build_user_profile(user_id=user_id, catalog=catalog)
        scored = inference_df.copy()
        scored["ml_score"] = probabilities
        scored["user_score"] = scored["movie_id"].map(
            lambda movie_id: self._personalization_score(movie_lookup.get(str(movie_id), {}), user_profile)
        )
        scored["matches_decade_filter"] = scored["movie_id"].map(
            lambda movie_id: int(
                movie_release_period(movie_lookup.get(str(movie_id), {})) == normalized_decade
            )
        )
        scored["matches_popularity_filter"] = scored["movie_id"].map(
            lambda movie_id: int(
                movie_popularity_bucket(
                    movie_lookup.get(str(movie_id), {}),
                    catalog_context,
                )
                == normalized_popularity
            )
        )
        scored["genre_overlap_count"] = scored["movie_id"].map(
            lambda movie_id: genre_overlap(canonical_ranked, movie_lookup.get(str(movie_id), {}))[1]
        )
        if user_profile.get("seen_movie_ids"):
            scored = scored[~scored["movie_id"].astype(str).isin(user_profile["seen_movie_ids"])].copy()
        scored = scored[
            (scored["matches_decade_filter"] == 1)
            & (scored["matches_popularity_filter"] == 1)
        ].copy()
        if scored.empty:
            raise ValueError("Não encontrei filmes que combinem com os filtros de gêneros, década e popularidade.")
        best_overlap = int(scored["genre_overlap_count"].max())
        if best_overlap <= 0:
            raise ValueError("Não encontrei filmes que combinem com os gêneros selecionados dentro dos filtros escolhidos.")
        scored = scored[scored["genre_overlap_count"] == best_overlap].copy()
        scored["final_score"] = (scored["ml_score"] + scored["user_score"]).clip(lower=0.0, upper=1.0)
        scored = scored.sort_values(
            by=["genre_overlap_count", "final_score", "ml_score", "nota", "votos"],
            ascending=[False, False, False, False, False],
        )
        scored = scored.drop_duplicates(subset=["movie_id"], keep="first")

        recommendations = []
        for _, row in scored.head(top_n).iterrows():
            movie = movie_lookup.get(str(row["movie_id"]), {})
            genres = [label_for_genre(genre) for genre in movie_genres(movie)]
            recommendations.append(
                {
                    "movie_id": row["movie_id"],
                    "titulo": row["titulo"],
                    "ano": int(row["ano"]),
                    "sinopse": movie.get("sinopse", ""),
                    "diretor": row["diretor"],
                    "nota": float(row["nota"]),
                    "votos": int(row["votos"]),
                    "duracao": int(row["duracao"]),
                    "streaming": movie.get("streaming", []),
                    "poster": row["poster"],
                    "url": row["url"],
                    "perfil": movie.get("perfil", ""),
                    "generos": genres,
                    "score": round(float(row["final_score"]), 4),
                    "score_percentual": round(float(row["final_score"]) * 100, 2),
                    "justificativa": self._build_reason(
                        movie,
                        canonical_ranked,
                        normalized_decade,
                        normalized_popularity,
                        float(row["final_score"]),
                        user_profile,
                    ),
                }
            )

        drift_report = self._generate_drift(inference_df)
        return {
            "ranked_genres": [label_for_genre(item) for item in canonical_ranked],
            "decade_preference": label_for_decade_preference(normalized_decade),
            "popularity_preference": label_for_popularity_preference(normalized_popularity),
            "winner_model": self.metadata.get("winner_model"),
            "drift_report": drift_report,
            "recommendations": recommendations,
        }

    def _build_reason(
        self,
        movie: dict,
        ranked_genres: list[str],
        decade_preference: str,
        popularity_preference: str,
        score: float,
        user_profile: dict,
    ) -> str:
        genres = set(movie_genres(movie))
        matches = [label_for_genre(genre) for genre in ranked_genres if genre in genres]
        release_period = label_for_decade_preference(decade_preference)
        popularity_label = label_for_popularity_preference(popularity_preference)
        director = str(movie.get("diretor", "") or "").strip()
        if director and director in user_profile.get("liked_directors", set()):
            return f"Combina com seus gêneros, {release_period.lower()} e diretores que você costuma gostar. Score {score:.2f}."
        if matches:
            return (
                f"Alinhado aos seus gêneros priorizados: {', '.join(matches)}, "
                f"com perfil {release_period.lower()} e {popularity_label.lower()}. Score {score:.2f}."
            )
        return f"Selecionado por similaridade de sinopse, metadados e histórico recente. Score {score:.2f}."

    def _build_user_profile(self, user_id: int | str | None, catalog: list[dict]) -> dict:
        if user_id is None:
            return {}

        feedback = load_feedback(FEEDBACK_PATH)
        if feedback.empty or "user_id" not in feedback.columns:
            return {}

        normalized_user_id = str(user_id)
        user_feedback = feedback[feedback["user_id"].fillna("").astype(str) == normalized_user_id]
        if user_feedback.empty:
            return {}

        movie_lookup = {str(movie.get("id")): movie for movie in catalog}
        profile = {
            "seen_movie_ids": set(),
            "liked_directors": set(),
            "director_scores": {},
            "primary_genre_scores": {},
            "secondary_genre_scores": {},
            "keyword_scores": {},
        }

        for row in user_feedback.itertuples(index=False):
            movie = movie_lookup.get(str(row.movie_id))
            if not movie:
                continue

            weight = 1 if int(row.label) == 1 else -1
            profile["seen_movie_ids"].add(str(row.movie_id))

            primary_genre, secondary_genres = primary_and_secondary_genres(movie)
            profile["primary_genre_scores"][primary_genre] = (
                profile["primary_genre_scores"].get(primary_genre, 0.0) + 0.18 * weight
            )

            for genre in secondary_genres:
                profile["secondary_genre_scores"][genre] = (
                    profile["secondary_genre_scores"].get(genre, 0.0) + 0.08 * weight
                )

            director = str(movie.get("diretor", "") or "").strip()
            if director:
                profile["director_scores"][director] = profile["director_scores"].get(director, 0.0) + 0.12 * weight
                if weight > 0:
                    profile["liked_directors"].add(director)

            for keyword in (movie.get("palavras_chave", []) or [])[:8]:
                normalized_keyword = str(keyword).strip().lower()
                if normalized_keyword:
                    profile["keyword_scores"][normalized_keyword] = (
                        profile["keyword_scores"].get(normalized_keyword, 0.0) + 0.03 * weight
                    )

        return profile

    def _personalization_score(self, movie: dict, user_profile: dict) -> float:
        if not movie or not user_profile:
            return 0.0

        primary_genre, secondary_genres = primary_and_secondary_genres(movie)
        score = 0.0
        score += user_profile["primary_genre_scores"].get(primary_genre, 0.0)
        score += user_profile["director_scores"].get(str(movie.get("diretor", "") or "").strip(), 0.0)

        for genre in secondary_genres:
            score += user_profile["secondary_genre_scores"].get(genre, 0.0)

        for keyword in (movie.get("palavras_chave", []) or [])[:8]:
            normalized_keyword = str(keyword).strip().lower()
            if normalized_keyword:
                score += user_profile["keyword_scores"].get(normalized_keyword, 0.0)

        return max(-0.35, min(score, 0.35))

    def _generate_drift(self, current_data: pd.DataFrame) -> str | None:
        if not REFERENCE_DATA_PATH.exists():
            return None

        reference_data = pd.read_csv(REFERENCE_DATA_PATH)
        subset = current_data[
            [
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
        ]
        return generate_drift_report(reference_data=reference_data, current_data=subset, output_path=DRIFT_REPORT_PATH)
