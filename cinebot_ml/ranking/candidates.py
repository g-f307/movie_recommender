"""Construção única e determinística do conjunto candidato experimental."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from cinebot_ml.dataset import (
    build_catalog_context,
    canonicalize_decade_preference,
    canonicalize_genre,
    canonicalize_popularity_preference,
    load_catalog,
    movie_genres,
    movie_popularity_bucket,
    movie_release_period,
)
from cinebot_ml.experiment_config import (
    DEFAULT_CONFIG_PATH,
    load_config,
    resolve_project_path,
    sha256_file,
)
from cinebot_ml.ranking.contracts import MovieId, RecommendationRequest


CANDIDATE_SET_VERSION = "1.0"
FILTER_KEYS = {"genres", "decade", "popularity"}


class CandidateSetValidationError(ValueError):
    """Catálogo, filtros ou histórico incompatíveis com o conjunto candidato."""


def _movie_id_key(movie_id: MovieId) -> tuple[int, Any]:
    return (0, movie_id) if isinstance(movie_id, int) else (1, str(movie_id))


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EligibilityPolicy:
    """Critérios mínimos aplicados igualmente antes de qualquer baseline."""

    version: str = "1.0"
    require_title: bool = True
    require_year: bool = True
    require_genre: bool = True
    exclude_history_items: bool = True

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise CandidateSetValidationError("A política de elegibilidade deve possuir versão.")


@dataclass(frozen=True)
class CandidateSet:
    """Snapshot comum de filmes elegíveis entregue a todos os métodos."""

    candidate_set_id: str
    version: str
    catalog_path: str
    catalog_sha256: str
    policy: EligibilityPolicy
    filters: Mapping[str, Any]
    movies: tuple[Mapping[str, Any], ...]
    total_catalog_items: int
    excluded_counts: Mapping[str, int] = field(default_factory=dict)

    @property
    def movie_ids(self) -> tuple[MovieId, ...]:
        return tuple(movie["id"] for movie in self.movies)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "candidate_set_version": self.version,
            "candidate_set_id": self.candidate_set_id,
            "catalog": {"path": self.catalog_path, "sha256": self.catalog_sha256},
            "policy": asdict(self.policy),
            "filters": dict(self.filters),
            "total_catalog_items": self.total_catalog_items,
            "candidate_count": len(self.movies),
            "candidate_movie_ids": list(self.movie_ids),
            "excluded_counts": dict(sorted(self.excluded_counts.items())),
        }

    def attach_to_request(self, request: RecommendationRequest) -> RecommendationRequest:
        """Produz a requisição final sem alterar o contexto experimental."""
        if request.candidate_movie_ids and request.candidate_movie_ids != self.movie_ids:
            raise CandidateSetValidationError(
                "A requisição já contém candidatos diferentes do CandidateSet."
            )
        return replace(request, candidate_movie_ids=self.movie_ids)


def _validated_year(movie: Mapping[str, Any]) -> bool:
    value = movie.get("ano")
    try:
        year = int(value)
    except (TypeError, ValueError):
        return False
    return 1888 <= year <= 2200


def _eligibility_reason(movie: Mapping[str, Any], policy: EligibilityPolicy) -> str | None:
    movie_id = movie.get("id")
    if isinstance(movie_id, bool) or not isinstance(movie_id, (int, str)):
        return "invalid_id"
    if isinstance(movie_id, int) and movie_id < 0:
        return "invalid_id"
    if isinstance(movie_id, str) and not movie_id.strip():
        return "invalid_id"
    if policy.require_title and not str(movie.get("titulo") or "").strip():
        return "missing_title"
    if policy.require_year and not _validated_year(movie):
        return "invalid_year"
    if policy.require_genre and not movie_genres(dict(movie)):
        return "missing_genre"
    return None


def _has_popularity_metadata(movie: Mapping[str, Any]) -> bool:
    try:
        note = float(movie.get("nota"))
        votes = int(movie.get("votos"))
    except (TypeError, ValueError):
        return False
    return 0.0 <= note <= 10.0 and votes >= 0


def _normalize_filters(request: RecommendationRequest) -> dict[str, Any]:
    raw = request.profile_data.get("candidate_filters", {})
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise CandidateSetValidationError("profile_data.candidate_filters deve ser um objeto.")
    unknown = sorted(set(raw) - FILTER_KEYS)
    if unknown:
        raise CandidateSetValidationError(f"Filtro candidato desconhecido: {unknown[0]}")
    normalized: dict[str, Any] = {}
    if "genres" in raw:
        genres = raw["genres"]
        if not isinstance(genres, Sequence) or isinstance(genres, (str, bytes)) or not genres:
            raise CandidateSetValidationError("candidate_filters.genres deve ser uma lista não vazia.")
        values = sorted({canonicalize_genre(str(genre)) for genre in genres})
        if any(not value for value in values):
            raise CandidateSetValidationError("candidate_filters.genres contém valor vazio.")
        normalized["genres"] = values
    if "decade" in raw:
        decade = canonicalize_decade_preference(raw["decade"])
        if decade is None:
            raise CandidateSetValidationError("candidate_filters.decade é inválido.")
        normalized["decade"] = decade
    if "popularity" in raw:
        popularity = canonicalize_popularity_preference(raw["popularity"])
        if popularity is None:
            raise CandidateSetValidationError("candidate_filters.popularity é inválido.")
        normalized["popularity"] = popularity
    return normalized


def _history_movie_ids(
    request: RecommendationRequest,
    catalog_ids: set[MovieId],
) -> set[MovieId]:
    consumed: set[MovieId] = set()
    for index, event in enumerate(request.history):
        if "movie_id" not in event:
            raise CandidateSetValidationError(f"history[{index}].movie_id é obrigatório.")
        movie_id = event["movie_id"]
        if movie_id not in catalog_ids:
            raise CandidateSetValidationError(
                f"history[{index}] referencia filme inexistente no catálogo: {movie_id}"
            )
        consumed.add(movie_id)
    return consumed


def build_candidate_set(
    catalog: Sequence[Mapping[str, Any]],
    request: RecommendationRequest,
    *,
    catalog_path: str,
    catalog_sha256: str,
    policy: EligibilityPolicy | None = None,
) -> CandidateSet:
    """Aplica elegibilidade, histórico e filtros explícitos em ordem estável."""
    if not catalog:
        raise CandidateSetValidationError("O catálogo está vazio.")
    policy = policy or EligibilityPolicy()
    filters = _normalize_filters(request)
    seen_ids: set[MovieId] = set()
    for index, movie in enumerate(catalog):
        if not isinstance(movie, Mapping):
            raise CandidateSetValidationError(f"Item {index} do catálogo não é um objeto.")
        movie_id = movie.get("id")
        if isinstance(movie_id, bool) or not isinstance(movie_id, (int, str)):
            continue
        if movie_id in seen_ids:
            raise CandidateSetValidationError(f"movie_id duplicado no catálogo: {movie_id}")
        seen_ids.add(movie_id)

    consumed = _history_movie_ids(request, seen_ids)
    context: Mapping[str, Any] = {}
    if "popularity" in filters:
        popularity_catalog = [
            dict(movie)
            for movie in catalog
            if _eligibility_reason(movie, policy) is None and _has_popularity_metadata(movie)
        ]
        context = build_catalog_context(popularity_catalog)
    accepted: list[Mapping[str, Any]] = []
    excluded: dict[str, int] = {}

    def reject(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for movie in catalog:
        reason = _eligibility_reason(movie, policy)
        if reason:
            reject(reason)
            continue
        movie_id = movie["id"]
        if policy.exclude_history_items and movie_id in consumed:
            reject("already_presented_or_rated")
            continue
        if "genres" in filters and not set(filters["genres"]).intersection(movie_genres(dict(movie))):
            reject("genre_filter")
            continue
        if "decade" in filters and movie_release_period(dict(movie)) != filters["decade"]:
            reject("decade_filter")
            continue
        if "popularity" in filters and not _has_popularity_metadata(movie):
            reject("invalid_popularity_metadata")
            continue
        if (
            "popularity" in filters
            and movie_popularity_bucket(dict(movie), context) != filters["popularity"]
        ):
            reject("popularity_filter")
            continue
        accepted.append(MappingProxyType(dict(movie)))

    accepted.sort(key=lambda movie: _movie_id_key(movie["id"]))
    accepted_ids = tuple(movie["id"] for movie in accepted)
    if request.candidate_movie_ids and request.candidate_movie_ids != accepted_ids:
        raise CandidateSetValidationError(
            "candidate_movie_ids da requisição diverge do conjunto candidato calculado."
        )
    identity = {
        "version": CANDIDATE_SET_VERSION,
        "catalog_sha256": catalog_sha256,
        "policy": asdict(policy),
        "filters": filters,
        "consumed_movie_ids": sorted(consumed, key=_movie_id_key),
        "candidate_movie_ids": list(accepted_ids),
    }
    return CandidateSet(
        candidate_set_id=_canonical_hash(identity)[:20],
        version=CANDIDATE_SET_VERSION,
        catalog_path=catalog_path,
        catalog_sha256=catalog_sha256,
        policy=policy,
        filters=filters,
        movies=tuple(accepted),
        total_catalog_items=len(catalog),
        excluded_counts=excluded,
    )


def build_candidate_set_from_config(
    request: RecommendationRequest,
    config_path: Path = DEFAULT_CONFIG_PATH,
    *,
    project_root: Path | None = None,
    policy: EligibilityPolicy | None = None,
) -> CandidateSet:
    """Carrega o catálogo pelo caminho relativo da configuração experimental."""
    config = load_config(config_path)
    relative_path = config["paths"]["catalog"]
    root = project_root or config_path.resolve().parent.parent
    catalog_path = resolve_project_path(relative_path, root)
    if not catalog_path.is_file():
        raise CandidateSetValidationError(f"Catálogo não encontrado: {relative_path}")
    try:
        catalog = load_catalog(catalog_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateSetValidationError(f"Não foi possível carregar o catálogo: {exc}") from exc
    return build_candidate_set(
        catalog,
        request,
        catalog_path=relative_path,
        catalog_sha256=sha256_file(catalog_path),
        policy=policy,
    )
