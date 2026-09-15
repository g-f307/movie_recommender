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
    "EligibilityPolicy",
    "PopularityConfig",
    "PopularityConfigError",
    "PopularityRecommender",
    "RankingResult",
    "RecommendationItem",
    "RecommendationRequest",
    "Recommender",
    "build_candidate_set",
    "build_candidate_set_from_config",
    "bayesian_popularity_score",
    "load_popularity_config",
    "rank_scored_candidates",
]
