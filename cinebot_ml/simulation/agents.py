"""Agentes sintéticos reproduzíveis e independentes dos recomendadores."""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.dataset import movie_genres, slugify

DEFAULT_AGENT_CONFIG_PATH = PROJECT_ROOT / "configs/agents/personas_v1.yaml"
DEFAULT_AGENT_SCHEMA_PATH = PROJECT_ROOT / "configs/agents/personas_v1.schema.json"

class AgentConfigError(ValueError):
    """Configuração ou item incompatível com os agentes sintéticos."""

def _hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()

@dataclass(frozen=True)
class AgentConfig:
    version: str
    personas: Mapping[str, Mapping[str, float | int]]
    cohort: Mapping[str, int]
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]

def load_agent_config(path: Path = DEFAULT_AGENT_CONFIG_PATH) -> AgentConfig:
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(DEFAULT_AGENT_SCHEMA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise AgentConfigError(f"Não foi possível carregar personas: {exc}") from exc
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        raise AgentConfigError(f"Configuração de personas inválida: {error.message}")
    if set(payload["cohort"]) != set(payload["personas"]):
        raise AgentConfigError("A distribuição do cohort deve cobrir exatamente as personas.")
    numbers = [value for persona in payload["personas"].values() for value in persona.values()]
    if any(not math.isfinite(float(value)) for value in numbers):
        raise AgentConfigError("Parâmetros das personas devem ser finitos.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AgentConfigError("Lock das personas ausente ou inválido.") from exc
    if lock.get("sha256") != digest:
        raise AgentConfigError("A configuração de personas foi alterada sem nova versão.")
    return AgentConfig(str(payload["version"]), payload["personas"], payload["cohort"], path, digest, dict(payload))

@dataclass(frozen=True)
class LatentPreference:
    genres: tuple[tuple[str, float], ...]
    directors: tuple[str, ...]
    keywords: tuple[str, ...]
    decade: str
    popularity_weight: float
    preference_id: str

@dataclass(frozen=True)
class SyntheticJudgment:
    source: str
    agent_id: str
    movie_id: int | str
    utility: float
    like_probability: float
    feedback: str
    graded_relevance: int
    interaction: int

def _vocabulary(catalog: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    values: set[str] = set()
    for movie in catalog:
        raw = movie.get(field)
        raw_values = [raw] if isinstance(raw, str) else (raw or [])
        values.update(normalized for value in raw_values if (normalized := slugify(str(value))))
    return sorted(values)

def build_agent(persona: str, seed: int, catalog: Sequence[Mapping[str, Any]], config: AgentConfig | None = None, index: int = 0) -> "SyntheticAgent":
    config = config or load_agent_config()
    if persona not in config.personas:
        raise AgentConfigError(f"Persona desconhecida: {persona}")
    params = config.personas[persona]
    genres = sorted({genre for movie in catalog for genre in movie_genres(dict(movie))})
    if not genres:
        raise AgentConfigError("Catálogo não possui gêneros para gerar preferências.")
    rng = random.Random(f"{config.version}:{persona}:{seed}:{index}")
    count = min(int(params["genre_count"]), len(genres))
    selected = rng.sample(genres, count)
    concentration = float(params["concentration"])
    weighted_genres = tuple((genre, round(concentration * (0.82 ** rank), 8)) for rank, genre in enumerate(selected))
    directors, keywords = _vocabulary(catalog, "diretor"), _vocabulary(catalog, "palavras_chave")
    latent_payload = {"genres": weighted_genres, "directors": rng.sample(directors, min(2, len(directors))), "keywords": rng.sample(keywords, min(3, len(keywords))), "decade": rng.choice(["antes_2000", "anos_2000", "moderno"]), "popularity_weight": params["popularity_weight"]}
    preference = LatentPreference(weighted_genres, tuple(latent_payload["directors"]), tuple(latent_payload["keywords"]), str(latent_payload["decade"]), float(params["popularity_weight"]), _hash(latent_payload)[:24])
    agent_id = f"synthetic-{persona}-{seed}-{index:04d}"
    return SyntheticAgent(agent_id, persona, seed, preference, params, config)

class SyntheticAgent:
    """Avalia um item sem receber ranking, método, posição ou score externo."""
    def __init__(self, agent_id: str, persona: str, seed: int, preference: LatentPreference, parameters: Mapping[str, float | int], config: AgentConfig):
        self.agent_id, self.persona, self.seed = agent_id, persona, seed
        self.preference, self.parameters, self.config = preference, parameters, config

    @property
    def preference_id(self) -> str:
        """Retorna o identificador estável da preferência latente do agente."""
        return self.preference.preference_id

    def _rng(self, movie_id: int | str, interaction: int, channel: str) -> random.Random:
        return random.Random(_hash([self.agent_id, movie_id, interaction, channel]))

    def utility(self, movie: Mapping[str, Any], interaction: int) -> float:
        movie_id = movie.get("id")
        if isinstance(movie_id, bool) or not isinstance(movie_id, (int, str)):
            raise AgentConfigError("Item sintético deve possuir movie_id válido.")
        genre_weights = dict(self.preference.genres)
        genre = max((genre_weights.get(value, 0.0) for value in movie_genres(dict(movie))), default=0.0)
        director = float(slugify(str(movie.get("diretor") or "")) in self.preference.directors)
        keywords = set(_vocabulary([movie], "palavras_chave"))
        keyword = len(keywords.intersection(self.preference.keywords)) / max(len(self.preference.keywords), 1)
        year = int(movie.get("ano", 0) or 0) if str(movie.get("ano", "")).isdigit() else 0
        decade = "moderno" if year >= 2010 else "anos_2000" if year >= 2000 else "antes_2000"
        decade_match = float(decade == self.preference.decade)
        votes = max(float(movie.get("votos", 0) or 0), 0.0)
        popularity = min(math.log1p(votes) / math.log1p(1_000_000), 1.0)
        stable_taste = self._rng(movie_id, 0, "latent-item").uniform(-0.08, 0.08)
        raw = 0.12 + 0.48 * genre + 0.12 * director + 0.10 * keyword + 0.10 * decade_match + self.preference.popularity_weight * popularity + stable_taste
        noise = self._rng(movie_id, interaction, "noise").gauss(0, float(self.parameters["noise"]))
        return round(min(1.0, max(0.0, raw + noise)), 8)

    def evaluate(self, movie: Mapping[str, Any], interaction: int) -> SyntheticJudgment:
        if interaction < 0:
            raise AgentConfigError("interaction deve ser não negativa.")
        utility = self.utility(movie, interaction)
        temperature = float(self.parameters["temperature"])
        probability = 1.0 / (1.0 + math.exp(-(utility - 0.5) / temperature))
        exploration = float(self.parameters["exploration"])
        probability = (1.0 - exploration) * probability + exploration * 0.5
        if self._rng(movie["id"], interaction, "contradiction").random() < float(self.parameters["contradiction"]):
            probability = 1.0 - probability
        probability = min(1.0, max(0.0, probability))
        liked = self._rng(movie["id"], interaction, "response").random() < probability
        grade = min(3, max(0, int(math.floor(utility * 4))))
        return SyntheticJudgment("synthetic_user", self.agent_id, movie["id"], utility, round(probability, 8), "like" if liked else "dislike", grade, interaction)

    def manifest(self) -> dict[str, Any]:
        return {"agent_id": self.agent_id, "persona": self.persona, "seed": self.seed, "version": self.config.version, "preference_id": self.preference.preference_id, "source": "synthetic_user", "parameters": dict(self.parameters), "config_sha256": self.config.source_sha256}

def build_agent_cohort(catalog: Sequence[Mapping[str, Any]], base_seed: int, config: AgentConfig | None = None) -> tuple[SyntheticAgent, ...]:
    config = config or load_agent_config()
    return tuple(build_agent(persona, base_seed, catalog, config, index) for persona in sorted(config.cohort) for index in range(config.cohort[persona]))
