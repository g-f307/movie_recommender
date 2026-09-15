"""Contratos e implementações do pipeline experimental de ranking."""

from cinebot_ml.ranking.candidates import (
    CandidateSet,
    CandidateSetValidationError,
    EligibilityPolicy,
    build_candidate_set,
    build_candidate_set_from_config,
)
from cinebot_ml.ranking.contracts import (
    ContractValidationError,
    RankingResult,
    RecommendationItem,
    RecommendationRequest,
    Recommender,
    rank_scored_candidates,
)
from cinebot_ml.ranking.content import (
    ContentConfig,
    ContentConfigError,
    ContentProfile,
    ContentRecommender,
    ItemContent,
    build_content_profile,
    build_item_content,
    content_similarity,
    load_content_config,
)
from cinebot_ml.ranking.popularity import (
    PopularityConfig,
    PopularityConfigError,
    PopularityRecommender,
    bayesian_popularity_score,
    load_popularity_config,
)

__all__ = [
    "CandidateSet",
    "CandidateSetValidationError",
    "ContractValidationError",
    "ContentConfig",
    "ContentConfigError",
    "ContentProfile",
    "ContentRecommender",
    "EligibilityPolicy",
    "PopularityConfig",
    "PopularityConfigError",
    "PopularityRecommender",
    "ItemContent",
    "RankingResult",
    "RecommendationItem",
    "RecommendationRequest",
    "Recommender",
    "build_candidate_set",
    "build_candidate_set_from_config",
    "build_content_profile",
    "build_item_content",
    "bayesian_popularity_score",
    "load_popularity_config",
    "content_similarity",
    "load_content_config",
    "rank_scored_candidates",
]
