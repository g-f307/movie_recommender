"""Baseline B2 com TF-IDF treinável somente na partição de treino."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import sklearn
import yaml
import numpy as np
from jsonschema import Draft202012Validator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.dataset import (
    canonicalize_genre,
    movie_genres,
)
from cinebot_ml.ranking.candidates import CandidateSet
from cinebot_ml.ranking.contracts import (
    ContractValidationError,
    RankingResult,
    RecommendationRequest,
    rank_scored_candidates,
)


DEFAULT_B2_CONFIG_PATH = PROJECT_ROOT / "configs" / "methods" / "b2_tfidf_v1.yaml"
DEFAULT_B2_SCHEMA_PATH = PROJECT_ROOT / "configs" / "methods" / "b2_tfidf_v1.schema.json"
PROFILE_GENRE_LIMIT = {"P0": 0, "P1": 1, "P2": 3, "P3": 3, "P4": 3, "P5": 3}


class TfidfConfigError(ValueError):
    """Configuração, artefato ou partição incompatível com B2."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class TfidfConfig:
    version: str
    vectorizer: Mapping[str, Any]
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]


@dataclass(frozen=True)
class TfidfArtifact:
    method_version: str
    config_sha256: str
    vocabulary: Mapping[str, int]
    idf: tuple[float, ...]
    training_corpus_sha256: str
    sklearn_version: str
    artifact_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "B2",
            "method_version": self.method_version,
            "config_sha256": self.config_sha256,
            "vocabulary": dict(sorted(self.vocabulary.items())),
            "idf": list(self.idf),
            "training_corpus_sha256": self.training_corpus_sha256,
            "sklearn_version": self.sklearn_version,
            "artifact_id": self.artifact_id,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TfidfArtifact":
        if value.get("method") != "B2":
            raise TfidfConfigError("Artefato TF-IDF não pertence ao método B2.")
        try:
            artifact = cls(
                method_version=str(value["method_version"]),
                config_sha256=str(value["config_sha256"]),
                vocabulary={str(term): int(index) for term, index in value["vocabulary"].items()},
                idf=tuple(float(number) for number in value["idf"]),
                training_corpus_sha256=str(value["training_corpus_sha256"]),
                sklearn_version=str(value["sklearn_version"]),
                artifact_id=str(value["artifact_id"]),
            )
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            raise TfidfConfigError("Artefato TF-IDF inválido.") from exc
        if sorted(artifact.vocabulary.values()) != list(range(len(artifact.vocabulary))):
            raise TfidfConfigError("Vocabulário TF-IDF possui índices inválidos.")
        if len(artifact.idf) != len(artifact.vocabulary) or any(
            not math.isfinite(value) or value <= 0 for value in artifact.idf
        ):
            raise TfidfConfigError("IDF incompatível com o vocabulário TF-IDF.")
        expected = _artifact_id(artifact.to_dict(), include_id=False)
        if artifact.artifact_id != expected:
            raise TfidfConfigError("Identidade do artefato TF-IDF é inválida.")
        return artifact

    @classmethod
    def load(cls, path: Path) -> "TfidfArtifact":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TfidfConfigError(f"Não foi possível carregar o artefato TF-IDF: {exc}") from exc
        if not isinstance(value, Mapping):
            raise TfidfConfigError("Artefato TF-IDF deve ser um objeto.")
        return cls.from_dict(value)


def _artifact_id(value: Mapping[str, Any], *, include_id: bool) -> str:
    payload = dict(value)
    if not include_id:
        payload.pop("artifact_id", None)
    return _sha256_bytes(_canonical_json(payload).encode("utf-8"))[:20]


def load_tfidf_config(path: Path = DEFAULT_B2_CONFIG_PATH) -> TfidfConfig:
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_B2_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise TfidfConfigError(f"Não foi possível carregar a configuração B2: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise TfidfConfigError("A configuração B2 deve ser um objeto.")
    Draft202012Validator.check_schema(schema)
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        location = ".".join(map(str, error.absolute_path)) or "config"
        raise TfidfConfigError(f"Configuração B2 inválida em {location}: {error.message}")
    lock_path = path.with_suffix(".lock.json")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TfidfConfigError(f"Lock B2 ausente ou inválido: {lock_path}") from exc
    digest = _sha256_bytes(path.read_bytes())
    if lock.get("sha256") != digest:
        raise TfidfConfigError("A configuração B2 congelada foi alterada sem nova versão.")
    return TfidfConfig(
        version=str(payload["version"]),
        vectorizer=dict(payload["vectorizer"]),
        source_path=path,
        source_sha256=digest,
        snapshot=dict(payload),
    )


def normalize_text(value: Any) -> str:
    """Normalização estável; acentos e caixa são tratados pelo vectorizer."""

    if value is None:
        return ""
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(str(item).strip() for item in value if str(item).strip())
    return " ".join(str(value).split())


def build_movie_document(movie: Mapping[str, Any]) -> str:
    genres = " ".join(movie_genres(dict(movie)))
    return normalize_text(
        [
            normalize_text(movie.get("sinopse")),
            genres,
            normalize_text(movie.get("palavras_chave")),
        ]
    )


def build_profile_document(request: RecommendationRequest) -> str:
    """Produz texto somente com atributos autorizados pelo perfil."""

    data = request.profile_data
    raw_genres = data.get("ranked_genres", [])
    if not isinstance(raw_genres, (list, tuple)):
        raw_genres = []
    genre_limit = PROFILE_GENRE_LIMIT[request.profile]
    genre_weights = (3, 2, 1)
    terms: list[str] = []
    for raw, repeats in zip(raw_genres[:genre_limit], genre_weights):
        genre = canonicalize_genre(str(raw))
        if genre:
            terms.extend([genre] * repeats)
    return normalize_text(terms)


def _vectorizer(config: TfidfConfig, vocabulary: Mapping[str, int] | None = None) -> TfidfVectorizer:
    values = config.vectorizer
    return TfidfVectorizer(
        lowercase=bool(values["lowercase"]),
        strip_accents=values["strip_accents"],
        stop_words=values["stop_words"],
        ngram_range=tuple(values["ngram_range"]),
        max_features=int(values["max_features"]),
        min_df=int(values["min_df"]),
        sublinear_tf=bool(values["sublinear_tf"]),
        norm=values["norm"],
        vocabulary=dict(vocabulary) if vocabulary is not None else None,
    )


def fit_tfidf_artifact(
    training_movies: Sequence[Mapping[str, Any]],
    *,
    partition: str,
    config_path: Path = DEFAULT_B2_CONFIG_PATH,
) -> TfidfArtifact:
    if partition.strip().lower() != "train":
        raise TfidfConfigError("TF-IDF só pode ser ajustado com partition=train; holdout é proibido.")
    config = load_tfidf_config(config_path)
    documents = [build_movie_document(movie) for movie in training_movies]
    if not documents or not any(document.strip() for document in documents):
        raise TfidfConfigError("Corpus de treino TF-IDF está vazio.")
    vectorizer = _vectorizer(config)
    try:
        vectorizer.fit(documents)
    except ValueError as exc:
        raise TfidfConfigError(f"Não foi possível ajustar o TF-IDF: {exc}") from exc
    corpus_hash = _sha256_bytes(_canonical_json(documents).encode("utf-8"))
    raw = {
        "method": "B2",
        "method_version": config.version,
        "config_sha256": config.source_sha256,
        "vocabulary": {
            term: int(index) for term, index in sorted(vectorizer.vocabulary_.items())
        },
        "idf": [float(value) for value in vectorizer.idf_],
        "training_corpus_sha256": corpus_hash,
        "sklearn_version": sklearn.__version__,
    }
    return TfidfArtifact(
        method_version=config.version,
        config_sha256=config.source_sha256,
        vocabulary=raw["vocabulary"],
        idf=tuple(raw["idf"]),
        training_corpus_sha256=corpus_hash,
        sklearn_version=sklearn.__version__,
        artifact_id=_artifact_id(raw, include_id=False),
    )


def _restore_vectorizer(config: TfidfConfig, artifact: TfidfArtifact) -> TfidfVectorizer:
    vectorizer = _vectorizer(config, artifact.vocabulary)
    vectorizer.fit(["placeholder"])
    vectorizer._tfidf.idf_ = np.asarray(artifact.idf, dtype=float)
    return vectorizer


class TfidfRecommender:
    """B2 textual, inicializado com artefato ajustado exclusivamente em treino."""

    method = "B2"

    def __init__(
        self,
        candidates: CandidateSet,
        artifact: TfidfArtifact,
        config_path: Path = DEFAULT_B2_CONFIG_PATH,
    ) -> None:
        self.candidates = candidates
        self.config = load_tfidf_config(config_path)
        if artifact.method_version != self.config.version:
            raise TfidfConfigError("Versão do artefato diverge da configuração B2.")
        if artifact.config_sha256 != self.config.source_sha256:
            raise TfidfConfigError("Hash da configuração diverge do artefato B2.")
        self.artifact = artifact
        self.method_version = self.config.version
        self._vectorizer = _restore_vectorizer(self.config, artifact)
        self._item_matrix = self._vectorizer.transform(
            [build_movie_document(movie) for movie in candidates.movies]
        )

    def recommend(self, request: RecommendationRequest) -> RankingResult:
        started = time.perf_counter()
        if request.method != self.method:
            raise ContractValidationError("TfidfRecommender aceita somente method=B2.")
        if request.method_version != self.method_version:
            raise ContractValidationError("A versão B2 da requisição diverge da configuração.")
        if request.candidate_movie_ids != self.candidates.movie_ids:
            raise ContractValidationError("A requisição diverge do CandidateSet de B2.")
        profile_document = build_profile_document(request)
        profile_vector = self._vectorizer.transform([profile_document])
        scores = cosine_similarity(profile_vector, self._item_matrix).ravel()
        scored = []
        for movie, score in zip(self.candidates.movies, scores):
            value = float(score)
            if not math.isfinite(value):
                value = 0.0
            metadata: dict[str, Any] = {
                "score_components": {
                    "formula": "tfidf_cosine_similarity",
                    "cosine_similarity": value,
                    "profile_document_empty": not bool(profile_document),
                }
            }
            title = str(movie.get("titulo") or "").strip()
            if title:
                metadata["title"] = title
            scored.append((movie["id"], value, metadata))
        result = RankingResult(
            experiment_id=request.experiment_id,
            method=self.method,
            method_version=self.method_version,
            condition=request.condition,
            profile=request.profile,
            seed=request.seed,
            logical_timestamp=request.logical_timestamp,
            status="completed",
            latency_ms=(time.perf_counter() - started) * 1000.0,
            ranked_items=rank_scored_candidates(scored, request.k),
            metadata={
                "candidate_set_id": self.candidates.candidate_set_id,
                "model_id": self.artifact.artifact_id,
            },
        )
        result.validate_against(request)
        return result

    def method_manifest(self) -> dict[str, Any]:
        try:
            config_path = self.config.source_path.relative_to(PROJECT_ROOT)
        except ValueError:
            config_path = Path(self.config.source_path.name)
        return {
            "method": self.method,
            "method_version": self.method_version,
            "config_path": config_path.as_posix(),
            "config_sha256": self.config.source_sha256,
            "config_snapshot": dict(self.config.snapshot),
            "artifact_id": self.artifact.artifact_id,
            "training_corpus_sha256": self.artifact.training_corpus_sha256,
            "vocabulary_size": len(self.artifact.vocabulary),
            "sklearn_version": self.artifact.sklearn_version,
            "fit_partition": "train",
            "holdout_used_for_fit": False,
            "candidate_set_id": self.candidates.candidate_set_id,
        }
