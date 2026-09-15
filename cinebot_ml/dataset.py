import csv
import json
import math
import unicodedata
from datetime import datetime, timezone
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd

from cinebot_ml.config import (
    DATASET_PATH,
    DECADE_LABELS,
    DECADE_WEIGHT,
    DEFAULT_DATA_PATH,
    FEEDBACK_PATH,
    GENRE_LABELS,
    HIGH_RATING_BONUS,
    KEYWORD_DENSITY_BONUS,
    MULTI_STREAMING_BONUS,
    POPULARITY_LABELS,
    POPULARITY_WEIGHT,
    RELEVANCE_THRESHOLD,
    RANK_WEIGHTS,
)


FEEDBACK_COLUMNS = [
    "user_id",
    "movie_id",
    "pref_1",
    "pref_2",
    "pref_3",
    "decade_pref",
    "popularity_pref",
    "feedback",
    "label",
    "timestamp",
]


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_only = normalized.encode("ascii", "ignore").decode("ascii")
    return ascii_only.lower().strip().replace("-", " ").replace("_", " ")


GENRE_ALIASES = {
    "acao": "acao",
    "action": "acao",
    "ação": "acao",
    "comedia": "comedia",
    "comedy": "comedia",
    "comédia": "comedia",
    "drama": "drama",
    "terror": "terror",
    "horror": "terror",
    "ficcao cientifica": "scifi",
    "ficção científica": "scifi",
    "science fiction": "scifi",
    "sci fi": "scifi",
    "scifi": "scifi",
    "romance": "romance",
    "suspense": "suspense",
    "thriller": "suspense",
}

DECADE_ALIASES = {
    "moderno": "moderno",
    "anos 2000": "anos_2000",
    "anos2000": "anos_2000",
    "2000s": "anos_2000",
    "before 2000": "antes_2000",
    "antes dos anos 2000": "antes_2000",
    "antes de 2000": "antes_2000",
    "antes disso": "antes_2000",
}

POPULARITY_ALIASES = {
    "popular": "popular",
    "filme popular": "popular",
    "joia escondida": "joia_escondida",
    "joiaescondida": "joia_escondida",
    "hidden gem": "joia_escondida",
    "filme b": "joia_escondida",
}


def canonicalize_genre(value: str) -> str:
    slug = slugify(value)
    return GENRE_ALIASES.get(slug, slug.replace(" ", ""))


def label_for_genre(value: str) -> str:
    return GENRE_LABELS.get(canonicalize_genre(value), value)


def canonicalize_decade_preference(value: str | None) -> str | None:
    if value is None:
        return None
    slug = slugify(value)
    canonical = DECADE_ALIASES.get(slug, slug.replace(" ", "_"))
    return canonical if canonical in DECADE_LABELS else None


def canonicalize_popularity_preference(value: str | None) -> str | None:
    if value is None:
        return None
    slug = slugify(value)
    canonical = POPULARITY_ALIASES.get(slug, slug.replace(" ", "_"))
    return canonical if canonical in POPULARITY_LABELS else None


def label_for_decade_preference(value: str) -> str:
    canonical = canonicalize_decade_preference(value)
    return DECADE_LABELS.get(canonical, value)


def label_for_popularity_preference(value: str) -> str:
    canonical = canonicalize_popularity_preference(value)
    return POPULARITY_LABELS.get(canonical, value)


def load_catalog(path: Path | None = None) -> list[dict]:
    data_path = Path(path or DEFAULT_DATA_PATH)
    if not data_path.exists():
        return []

    with open(data_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    catalog = []
    for profile, movies in payload.get("perfis", {}).items():
        for movie in movies:
            enriched = dict(movie)
            enriched.setdefault("perfil", profile)
            catalog.append(enriched)
    return catalog


def movie_genres(movie: dict) -> list[str]:
    genres = set()

    for raw in [movie.get("perfil"), movie.get("genero")]:
        if raw:
            genres.add(canonicalize_genre(raw))

    for raw in movie.get("generos_secundarios", []) or []:
        if raw:
            genres.add(canonicalize_genre(raw))

    return sorted(g for g in genres if g in GENRE_LABELS)


def primary_and_secondary_genres(movie: dict) -> tuple[str, set[str]]:
    primary_genre = canonicalize_genre(movie.get("perfil") or movie.get("genero", ""))
    all_genres = set(movie_genres(movie))
    secondary_genres = {genre for genre in all_genres if genre != primary_genre}
    return primary_genre, secondary_genres


def genre_overlap(ranked_genres: tuple[str, str, str] | list[str], movie: dict) -> tuple[list[str], int]:
    ranked_set = {canonicalize_genre(genre) for genre in ranked_genres[:3]}
    overlaps = [genre for genre in movie_genres(movie) if genre in ranked_set]
    return overlaps, len(overlaps)


def build_ranked_profiles() -> list[tuple[str, str, str]]:
    return list(permutations(GENRE_LABELS.keys(), 3))


def build_catalog_context(catalog: list[dict]) -> dict[str, float]:
    max_votes = max((int(movie.get("votos", 0) or 0) for movie in catalog), default=1)
    period_vote_values: dict[str, list[int]] = {period: [] for period in DECADE_LABELS}
    period_max_votes: dict[str, float] = {}

    for movie in catalog:
        period = movie_release_period(movie)
        period_vote_values.setdefault(period, []).append(max(int(movie.get("votos", 0) or 0), 0))

    for period, values in period_vote_values.items():
        sorted_values = sorted(values)
        period_vote_values[period] = sorted_values
        period_max_votes[period] = float(max(sorted_values, default=1))

    return {
        "max_votes": float(max(max_votes, 1)),
        "period_vote_values": period_vote_values,
        "period_max_votes": period_max_votes,
    }


def movie_release_period(movie: dict) -> str:
    raw_year = movie.get("ano", 0)
    year = int(raw_year or 0) if str(raw_year).isdigit() else 0
    if year >= 2010:
        return "moderno"
    if year >= 2000:
        return "anos_2000"
    return "antes_2000"


def vote_scale(movie: dict, catalog_context: dict[str, float]) -> float:
    votes = max(int(movie.get("votos", 0) or 0), 0)
    period = movie_release_period(movie)
    period_max_votes = catalog_context.get("period_max_votes", {})
    max_votes = max(float(period_max_votes.get(period, catalog_context.get("max_votes", 1.0))), 1.0)
    return float(np.log1p(votes) / np.log1p(max_votes))


def vote_percentile(movie: dict, catalog_context: dict[str, float]) -> float:
    period = movie_release_period(movie)
    votes = max(int(movie.get("votos", 0) or 0), 0)
    period_vote_values = catalog_context.get("period_vote_values", {}).get(period, [])
    if not period_vote_values:
        return 1.0

    values = np.asarray(period_vote_values)
    rank = int(np.searchsorted(values, votes, side="right"))
    return float(rank / len(values))


def popularity_scores(movie: dict, catalog_context: dict[str, float]) -> tuple[float, float]:
    note = min(max(float(movie.get("nota", 0) or 0) / 10.0, 0.0), 1.0)
    votes = vote_scale(movie, catalog_context)
    percentile = vote_percentile(movie, catalog_context)
    popular_score = min(1.0, 0.45 * votes + 0.25 * percentile + 0.30 * note)
    hidden_gem_score = min(1.0, 0.45 * note + 0.30 * (1.0 - votes) + 0.25 * (1.0 - percentile))
    return popular_score, hidden_gem_score


def movie_popularity_bucket(movie: dict, catalog_context: dict[str, float]) -> str:
    popular_score, hidden_gem_score = popularity_scores(movie, catalog_context)
    note = min(max(float(movie.get("nota", 0) or 0) / 10.0, 0.0), 1.0)
    percentile = vote_percentile(movie, catalog_context)

    if note >= 0.74 and percentile <= 0.45:
        return "joia_escondida"
    if percentile >= 0.60:
        return "popular"
    return "popular" if popular_score >= hidden_gem_score else "joia_escondida"


def weighted_affinity(
    movie: dict,
    ranked_genres: tuple[str, str, str],
    decade_preference: str,
    popularity_preference: str,
    catalog_context: dict[str, float],
) -> float:
    preference_features = build_preference_features(
        movie,
        ranked_genres,
        decade_preference,
        popularity_preference,
        catalog_context,
    )
    score = 0.0

    score += RANK_WEIGHTS[0] * preference_features["primary_match_pref_1"]
    score += RANK_WEIGHTS[1] * preference_features["secondary_match_pref_2"]
    score += RANK_WEIGHTS[2] * preference_features["secondary_match_pref_3"]
    score += DECADE_WEIGHT * preference_features["matches_decade_pref"]
    score += POPULARITY_WEIGHT * preference_features["popularity_match_score"]

    note = float(movie.get("nota", 0) or 0)
    if note >= 8.0:
        score += HIGH_RATING_BONUS

    if len(movie.get("streaming", []) or []) >= 2:
        score += MULTI_STREAMING_BONUS

    if len(movie.get("palavras_chave", []) or []) >= 5:
        score += KEYWORD_DENSITY_BONUS

    if not movie.get("sinopse"):
        score -= 0.05

    return max(0.0, min(score, 1.0))


def build_profile_text(
    ranked_genres: tuple[str, str, str],
    decade_preference: str | None = None,
    popularity_preference: str | None = None,
) -> str:
    pieces = []
    for genre, weight in zip(ranked_genres, RANK_WEIGHTS):
        repeats = max(1, math.ceil(weight * 10))
        pieces.extend([label_for_genre(genre)] * repeats)

    if decade_preference:
        pieces.extend([label_for_decade_preference(decade_preference)] * 2)

    if popularity_preference:
        pieces.extend([label_for_popularity_preference(popularity_preference)] * 2)

    return " ".join(pieces)


def build_movie_text(movie: dict) -> str:
    return " ".join(
        [
            movie.get("titulo", ""),
            movie.get("sinopse", ""),
            movie.get("diretor", ""),
            " ".join(label_for_genre(g) for g in movie_genres(movie)),
            " ".join(movie.get("palavras_chave", []) or []),
            movie.get("nacionalidade", ""),
        ]
    ).strip()


def build_metadata_features(movie: dict) -> dict:
    sinopse = str(movie.get("sinopse", "") or "")
    palavras_chave = [item for item in (movie.get("palavras_chave", []) or []) if item]
    generos_secundarios = [item for item in (movie.get("generos_secundarios", []) or []) if item]
    streaming = movie.get("streaming", []) or []
    return {
        "sinopse_word_count": len(sinopse.split()),
        "keyword_count": len(palavras_chave),
        "secondary_genre_count": len(generos_secundarios),
        "has_streaming": int(bool(streaming)),
        "has_poster": int(bool(movie.get("poster"))),
    }


def build_preference_features(
    movie: dict,
    ranked_genres: tuple[str, str, str],
    decade_preference: str,
    popularity_preference: str,
    catalog_context: dict[str, float],
) -> dict:
    primary_genre, secondary_genres = primary_and_secondary_genres(movie)
    all_genres = set(movie_genres(movie))
    overlaps, overlap_count = genre_overlap(ranked_genres, movie)
    primary_match_pref_1 = int(primary_genre == ranked_genres[0])
    # Despite the legacy field names, preference 2/3 should count if the genre is present
    # either as primary or secondary.
    secondary_match_pref_2 = int(ranked_genres[1] in all_genres)
    secondary_match_pref_3 = int(ranked_genres[2] in all_genres)
    release_period = movie_release_period(movie)
    popular_score, hidden_gem_score = popularity_scores(movie, catalog_context)
    popularity_bucket = movie_popularity_bucket(movie, catalog_context)
    popularity_match_score = popular_score if popularity_preference == "popular" else hidden_gem_score

    return {
        "primary_match_pref_1": primary_match_pref_1,
        "secondary_match_pref_2": secondary_match_pref_2,
        "secondary_match_pref_3": secondary_match_pref_3,
        "ranked_match_count": primary_match_pref_1 + secondary_match_pref_2 + secondary_match_pref_3,
        "genre_overlap_count": overlap_count,
        "release_period": release_period,
        "matches_decade_pref": int(release_period == decade_preference),
        "popular_score": round(popular_score, 4),
        "hidden_gem_score": round(hidden_gem_score, 4),
        "matches_popularity_pref": int(popularity_bucket == popularity_preference),
        "popularity_match_score": round(popularity_match_score, 4),
    }


def build_training_rows(catalog: list[dict]) -> list[dict]:
    rows = []
    ranked_profiles = build_ranked_profiles()
    catalog_context = build_catalog_context(catalog)

    for movie in catalog:
        genres_text = " ".join(label_for_genre(g) for g in movie_genres(movie))
        metadata_features = build_metadata_features(movie)
        base_row = {
            "movie_id": movie.get("id"),
            "titulo": movie.get("titulo", ""),
            "sinopse": build_movie_text(movie),
            "genero": canonicalize_genre(movie.get("perfil") or movie.get("genero", "")),
            "generos_texto": genres_text,
            "diretor": movie.get("diretor", ""),
            "ano": int(movie.get("ano", 0) or 0) if str(movie.get("ano", "")).isdigit() else 0,
            "nota": float(movie.get("nota", 0) or 0),
            "votos": int(movie.get("votos", 0) or 0),
            "duracao": int(movie.get("duracao", 0) or 0),
            "streaming_count": len(movie.get("streaming", []) or []),
            "release_period": movie_release_period(movie),
            "poster": movie.get("poster", ""),
            "url": movie.get("url", ""),
            **metadata_features,
        }

        for ranked_genres in ranked_profiles:
            for decade_preference in DECADE_LABELS:
                for popularity_preference in POPULARITY_LABELS:
                    score = weighted_affinity(
                        movie,
                        ranked_genres,
                        decade_preference,
                        popularity_preference,
                        catalog_context,
                    )
                    preference_features = build_preference_features(
                        movie,
                        ranked_genres,
                        decade_preference,
                        popularity_preference,
                        catalog_context,
                    )
                    rows.append(
                        {
                            **base_row,
                            "pref_1": ranked_genres[0],
                            "pref_2": ranked_genres[1],
                            "pref_3": ranked_genres[2],
                            "decade_pref": decade_preference,
                            "popularity_pref": popularity_preference,
                            "ranked_profile_text": build_profile_text(
                                ranked_genres,
                                decade_preference,
                                popularity_preference,
                            ),
                            **preference_features,
                            "score_heuristico": round(score, 4),
                            "relevante": int(score >= RELEVANCE_THRESHOLD),
                            "label_source": "heuristic_proxy",
                            "feedback_applied": 0,
                            "feedback_scope": "none",
                            "feedback_votes": 0,
                        }
                    )

    return rows


def load_feedback(feedback_path: Path = FEEDBACK_PATH) -> pd.DataFrame:
    if not feedback_path.exists():
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    feedback = pd.read_csv(feedback_path)
    if feedback.empty:
        return pd.DataFrame(columns=FEEDBACK_COLUMNS)

    for column in FEEDBACK_COLUMNS:
        if column not in feedback.columns:
            feedback[column] = pd.NA

    feedback = feedback[FEEDBACK_COLUMNS].copy()
    feedback["movie_id"] = pd.to_numeric(feedback["movie_id"], errors="coerce").astype("Int64")
    feedback["pref_1"] = feedback["pref_1"].map(lambda value: canonicalize_genre(str(value)) if pd.notna(value) else value)
    feedback["pref_2"] = feedback["pref_2"].map(lambda value: canonicalize_genre(str(value)) if pd.notna(value) else value)
    feedback["pref_3"] = feedback["pref_3"].map(lambda value: canonicalize_genre(str(value)) if pd.notna(value) else value)
    feedback["decade_pref"] = feedback["decade_pref"].map(
        lambda value: canonicalize_decade_preference(str(value)) if pd.notna(value) and str(value).strip() else pd.NA
    )
    feedback["popularity_pref"] = feedback["popularity_pref"].map(
        lambda value: canonicalize_popularity_preference(str(value)) if pd.notna(value) and str(value).strip() else pd.NA
    )
    feedback["feedback"] = feedback["feedback"].fillna("").astype(str).str.strip().str.lower()
    feedback["label"] = pd.to_numeric(feedback["label"], errors="coerce").fillna(0).astype(int)
    feedback["user_id"] = feedback["user_id"].astype("string")
    feedback["timestamp"] = feedback["timestamp"].astype("string")
    feedback = feedback.dropna(subset=["movie_id", "pref_1", "pref_2", "pref_3"])
    return feedback


def override_labels_with_feedback(dataframe: pd.DataFrame, feedback_path: Path = FEEDBACK_PATH) -> pd.DataFrame:
    feedback = load_feedback(feedback_path)
    if feedback.empty:
        return dataframe

    specific_feedback = feedback.dropna(subset=["decade_pref", "popularity_pref"])
    legacy_feedback = feedback[feedback["decade_pref"].isna() | feedback["popularity_pref"].isna()]

    def aggregate_feedback(feedback_frame: pd.DataFrame, group_columns: list[str], label_column: str) -> pd.DataFrame:
        aggregated = (
            feedback_frame.groupby(group_columns, as_index=False)
            .agg(label_mean=("label", "mean"), feedback_count=("label", "size"))
        )
        aggregated[label_column] = (aggregated["label_mean"] >= 0.5).astype(int)
        return aggregated[group_columns + [label_column]]

    merged = dataframe.copy()

    if not specific_feedback.empty:
        specific_aggregated = aggregate_feedback(
            specific_feedback,
            ["movie_id", "pref_1", "pref_2", "pref_3", "decade_pref", "popularity_pref"],
            "label_specific",
        )
        specific_counts = (
            specific_feedback.groupby(
                ["movie_id", "pref_1", "pref_2", "pref_3", "decade_pref", "popularity_pref"],
                as_index=False,
            )
            .size()
            .rename(columns={"size": "feedback_votes_specific"})
        )
        specific_aggregated = specific_aggregated.merge(
            specific_counts,
            on=["movie_id", "pref_1", "pref_2", "pref_3", "decade_pref", "popularity_pref"],
            how="left",
        )
        merged = merged.merge(
            specific_aggregated,
            on=["movie_id", "pref_1", "pref_2", "pref_3", "decade_pref", "popularity_pref"],
            how="left",
        )
    else:
        merged["label_specific"] = pd.NA
        merged["feedback_votes_specific"] = pd.NA

    if not legacy_feedback.empty:
        legacy_aggregated = aggregate_feedback(
            legacy_feedback,
            ["movie_id", "pref_1", "pref_2", "pref_3"],
            "label_legacy",
        )
        legacy_counts = (
            legacy_feedback.groupby(["movie_id", "pref_1", "pref_2", "pref_3"], as_index=False)
            .size()
            .rename(columns={"size": "feedback_votes_legacy"})
        )
        legacy_aggregated = legacy_aggregated.merge(
            legacy_counts,
            on=["movie_id", "pref_1", "pref_2", "pref_3"],
            how="left",
        )
        merged = merged.merge(
            legacy_aggregated,
            on=["movie_id", "pref_1", "pref_2", "pref_3"],
            how="left",
        )
    else:
        merged["label_legacy"] = pd.NA
        merged["feedback_votes_legacy"] = pd.NA

    resolved_label = pd.to_numeric(
        merged["label_specific"].combine_first(merged["label_legacy"]),
        errors="coerce",
    )
    resolved_label_filled = resolved_label.fillna(0).astype(int)
    merged["relevante"] = np.where(
        resolved_label.notna(),
        resolved_label_filled,
        merged["relevante"].astype(int),
    )
    has_specific = merged["label_specific"].notna()
    has_legacy = merged["label_legacy"].notna()
    merged["feedback_applied"] = (has_specific | has_legacy).astype(int)
    merged["feedback_scope"] = np.select(
        [has_specific, has_legacy],
        ["exact_context_feedback", "profile_feedback"],
        default="none",
    )
    merged["label_source"] = np.select(
        [has_specific, has_legacy],
        ["feedback_exact_context", "feedback_profile"],
        default="heuristic_proxy",
    )
    specific_votes = pd.to_numeric(merged["feedback_votes_specific"], errors="coerce").fillna(0).astype(int)
    legacy_votes = pd.to_numeric(merged["feedback_votes_legacy"], errors="coerce").fillna(0).astype(int)
    merged["feedback_votes"] = np.where(has_specific, specific_votes, np.where(has_legacy, legacy_votes, 0))
    return merged.drop(
        columns=[
            "label_specific",
            "label_legacy",
            "feedback_votes_specific",
            "feedback_votes_legacy",
        ]
    )


def summarize_supervision_sources(dataframe: pd.DataFrame) -> dict:
    total_rows = int(len(dataframe))
    unique_movies = int(dataframe["movie_id"].nunique()) if "movie_id" in dataframe.columns else 0
    class_distribution = {}
    if "relevante" in dataframe.columns and total_rows:
        counts = dataframe["relevante"].value_counts(dropna=False).sort_index()
        class_distribution = {
            str(int(label)): {
                "rows": int(count),
                "share": round(float(count / total_rows), 4),
            }
            for label, count in counts.items()
        }

    if "label_source" in dataframe.columns and total_rows:
        source_counts = dataframe["label_source"].value_counts(dropna=False)
        label_sources = {
            str(source): {
                "rows": int(count),
                "share": round(float(count / total_rows), 4),
            }
            for source, count in source_counts.items()
        }
    else:
        label_sources = {}

    feedback_rows = int(dataframe["feedback_applied"].sum()) if "feedback_applied" in dataframe.columns else 0
    feedback_events = int(dataframe["feedback_votes"].sum()) if "feedback_votes" in dataframe.columns else 0

    return {
        "total_rows": total_rows,
        "unique_movies": unique_movies,
        "class_distribution": class_distribution,
        "label_sources": label_sources,
        "feedback_rows": feedback_rows,
        "feedback_rows_share": round(float(feedback_rows / total_rows), 4) if total_rows else 0.0,
        "feedback_events": feedback_events,
    }


def build_dataset(
    data_path: Path | None = None,
    output_path: Path = DATASET_PATH,
    feedback_path: Path = FEEDBACK_PATH,
) -> pd.DataFrame:
    catalog = load_catalog(data_path)
    rows = build_training_rows(catalog)
    dataframe = pd.DataFrame(rows)
    dataframe = override_labels_with_feedback(dataframe, feedback_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    return dataframe


def build_inference_frame(
    catalog: list[dict],
    ranked_genres: list[str],
    decade_preference: str | None,
    popularity_preference: str | None,
    *,
    allow_missing_preferences: bool = False,
) -> pd.DataFrame:
    normalized_genres = [canonicalize_genre(genre) for genre in ranked_genres[:3]]
    if allow_missing_preferences:
        normalized_genres.extend(["__missing__"] * (3 - len(normalized_genres)))
    if len(normalized_genres) < 3:
        raise ValueError("Três gêneros precisam ser informados para inferência.")
    ranked_tuple = tuple(normalized_genres[:3])
    normalized_decade = canonicalize_decade_preference(decade_preference)
    normalized_popularity = canonicalize_popularity_preference(popularity_preference)
    if allow_missing_preferences:
        normalized_decade = normalized_decade or "__missing__"
        normalized_popularity = normalized_popularity or "__missing__"
    else:
        normalized_decade = normalized_decade or "moderno"
        normalized_popularity = normalized_popularity or "popular"
    catalog_context = build_catalog_context(catalog)
    rows = []

    for movie in catalog:
        metadata_features = build_metadata_features(movie)
        preference_features = build_preference_features(
            movie,
            ranked_tuple,
            normalized_decade,
            normalized_popularity,
            catalog_context,
        )
        rows.append(
            {
                "movie_id": movie.get("id"),
                "titulo": movie.get("titulo", ""),
                "sinopse": build_movie_text(movie),
                "genero": canonicalize_genre(movie.get("perfil") or movie.get("genero", "")),
                "generos_texto": " ".join(label_for_genre(g) for g in movie_genres(movie)),
                "diretor": movie.get("diretor", ""),
                "ano": int(movie.get("ano", 0) or 0) if str(movie.get("ano", "")).isdigit() else 0,
                "nota": float(movie.get("nota", 0) or 0),
                "votos": int(movie.get("votos", 0) or 0),
                "duracao": int(movie.get("duracao", 0) or 0),
                "streaming_count": len(movie.get("streaming", []) or []),
                "release_period": movie_release_period(movie),
                "poster": movie.get("poster", ""),
                "url": movie.get("url", ""),
                "pref_1": ranked_tuple[0],
                "pref_2": ranked_tuple[1],
                "pref_3": ranked_tuple[2],
                "decade_pref": normalized_decade,
                "popularity_pref": normalized_popularity,
                "ranked_profile_text": build_profile_text(
                    ranked_tuple,
                    normalized_decade,
                    normalized_popularity,
                ),
                **metadata_features,
                **preference_features,
            }
        )

    return pd.DataFrame(rows)


def append_feedback(
    user_id: int | str | None,
    movie_id: int,
    ranked_genres: list[str],
    decade_preference: str | None,
    popularity_preference: str | None,
    feedback_value: str,
    feedback_path: Path = FEEDBACK_PATH,
) -> None:
    label = 1 if feedback_value == "like" else 0
    feedback_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "user_id": str(user_id) if user_id is not None else pd.NA,
        "movie_id": movie_id,
        "pref_1": canonicalize_genre(ranked_genres[0]),
        "pref_2": canonicalize_genre(ranked_genres[1]),
        "pref_3": canonicalize_genre(ranked_genres[2]),
        "decade_pref": canonicalize_decade_preference(decade_preference) if decade_preference else pd.NA,
        "popularity_pref": (
            canonicalize_popularity_preference(popularity_preference) if popularity_preference else pd.NA
        ),
        "feedback": feedback_value,
        "label": label,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    feedback = load_feedback(feedback_path)
    updated = pd.concat([feedback, pd.DataFrame([row])], ignore_index=True)
    updated = updated[FEEDBACK_COLUMNS]
    updated.to_csv(feedback_path, index=False)


def main() -> None:
    dataframe = build_dataset()
    print(f"Dataset gerado com {len(dataframe)} linhas em {DATASET_PATH}")


if __name__ == "__main__":
    main()
