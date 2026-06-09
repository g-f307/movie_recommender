TEXT_COLUMNS = ["sinopse", "ranked_profile_text", "generos_texto"]
CATEGORICAL_COLUMNS = [
    "genero",
    "pref_1",
    "pref_2",
    "pref_3",
    "decade_pref",
    "popularity_pref",
    "release_period",
    "diretor",
]
NUMERIC_COLUMNS = [
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
FEATURE_COLUMNS = TEXT_COLUMNS + CATEGORICAL_COLUMNS + NUMERIC_COLUMNS
