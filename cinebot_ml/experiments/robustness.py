"""Perturbações exploratórias versionadas sem alterar recomendadores."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.dataset import movie_genres

DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/robustness_v1.yaml"
DEFAULT_SCHEMA = PROJECT_ROOT / "configs/experiments/robustness_v1.schema.json"


class RobustnessError(ValueError):
    """Cenário ou resultado incompatível com o protocolo."""


@dataclass(frozen=True)
class Scenario:
    name: str
    kind: str
    intensity: float
    version: str


@dataclass(frozen=True)
class RobustnessCase:
    scenario: Scenario
    seed: int
    agent_id: str
    catalog: tuple[Mapping[str, Any], ...]
    profile: Mapping[str, Any]
    feedback: tuple[Mapping[str, Any], ...]
    source: str = "synthetic_user"

    @property
    def case_id(self) -> str:
        return _digest({"scenario": self.scenario.__dict__, "seed": self.seed,
                        "agent_id": self.agent_id, "catalog": self.catalog,
                        "profile": self.profile, "feedback": self.feedback})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=list).encode()).hexdigest()[:24]


def _genres(movie: Mapping[str, Any]) -> list[str]:
    # O catálogo oficial usa perfil/genero/generos_secundarios; o campo
    # generos é aceito apenas para fixtures e catálogos legados.
    return sorted(set(movie_genres(dict(movie))) | set(movie.get("generos", []) or []))


def load_robustness_config(path: Path = DEFAULT_CONFIG) -> tuple[tuple[Scenario, ...], tuple[int, ...], str]:
    try:
        raw = path.read_bytes()
        payload = yaml.safe_load(raw)
        schema = json.loads(DEFAULT_SCHEMA.read_text(encoding="utf-8"))
        error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
        if error:
            raise RobustnessError(error.message)
        digest = hashlib.sha256(raw).hexdigest()
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
        if lock.get("sha256") != digest:
            raise RobustnessError("Configuração de robustez alterada sem lock.")
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise RobustnessError(f"Configuração de robustez inválida: {exc}") from exc
    expected = {"nominal", "random_feedback", "contradictory_feedback", "extreme_negative",
                "specific_preference", "broad_preference", "few_candidates", "rare_genre",
                "popularity_concentration", "long_tail", "missing_metadata", "catalog_shift"}
    if set(payload["scenarios"]) != expected or any(name != spec["kind"] for name, spec in payload["scenarios"].items()):
        raise RobustnessError("Cenários de robustez divergem do protocolo.")
    return tuple(Scenario(name, spec["kind"], float(spec["intensity"]), str(payload["version"]))
                 for name, spec in sorted(payload["scenarios"].items())), tuple(payload["seeds"]), digest


def make_case(scenario: Scenario, seed: int, agent_id: str,
              catalog: Sequence[Mapping[str, Any]], profile: Mapping[str, Any],
              feedback: Sequence[Mapping[str, Any]]) -> RobustnessCase:
    rng = random.Random(f"{scenario.version}:{scenario.name}:{seed}:{agent_id}")
    movies = [dict(movie) for movie in catalog]
    visible = dict(profile)
    events = [dict(event) for event in feedback]
    kind, intensity = scenario.kind, scenario.intensity
    if kind == "random_feedback":
        for event in events:
            event["feedback"] = rng.choice(("like", "dislike"))
    elif kind == "contradictory_feedback":
        for event in events:
            event["feedback"] = "dislike" if event.get("feedback") == "like" else "like"
    elif kind == "extreme_negative":
        for event in events:
            event["feedback"] = "dislike"
    elif kind == "specific_preference":
        visible["ranked_genres"] = list(visible.get("ranked_genres", []))[:1]
    elif kind == "broad_preference":
        visible["ranked_genres"] = sorted({genre for movie in movies for genre in _genres(movie)})
    elif kind == "few_candidates":
        movies = movies[:int(intensity)]
    elif kind == "rare_genre":
        counts: dict[str, int] = {}
        for movie in movies:
            for genre in _genres(movie):
                counts[genre] = counts.get(genre, 0) + 1
        if counts:
            visible["ranked_genres"] = [min(counts, key=lambda genre: (counts[genre], genre))]
    elif kind in {"popularity_concentration", "long_tail"}:
        ordered = sorted(movies, key=lambda movie: (int(movie.get("votos") or 0), str(movie.get("id"))))
        count = max(1, round(len(ordered) * intensity))
        movies = ordered[-count:] if kind == "popularity_concentration" else ordered[:count]
    elif kind == "missing_metadata":
        for movie in rng.sample(movies, min(len(movies), round(len(movies) * intensity))):
            for field in ("diretor", "sinopse", "palavras_chave", "nota", "votos"):
                movie.pop(field, None)
    elif kind == "catalog_shift":
        movies = [movie for movie in movies if rng.random() >= intensity]
    elif kind != "nominal":
        raise RobustnessError(f"Cenário desconhecido: {kind}")
    return RobustnessCase(scenario, seed, agent_id, tuple(movies), visible, tuple(events))


def run_robustness(catalog: Sequence[Mapping[str, Any]], agent_id: str,
                   profile: Mapping[str, Any], feedback: Sequence[Mapping[str, Any]],
                   runner: Callable[[RobustnessCase], Mapping[str, Any]],
                   config_path: Path = DEFAULT_CONFIG) -> tuple[Mapping[str, Any], ...]:
    scenarios, seeds, digest = load_robustness_config(config_path)
    rows = []
    for seed in seeds:
        for scenario in scenarios:
            case = make_case(scenario, seed, agent_id, catalog, profile, feedback)
            common = {"case_id": case.case_id, "scenario": scenario.name, "intensity": scenario.intensity,
                      "version": scenario.version, "seed": seed, "agent_id": agent_id,
                      "config_sha256": digest, "evidence": "exploratory_synthetic"}
            try:
                result = runner(case)
                if not isinstance(result, Mapping):
                    raise RobustnessError("Runner deve retornar objeto de métricas.")
                rows.append({**common, "status": "completed", "result": dict(result)})
            except Exception as exc:
                rows.append({**common, "status": "failed", "error_type": type(exc).__name__,
                             "error_message": str(exc), "result": None})
    return tuple(rows)


def write_robustness_report(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Persiste inclusive falhas, sem converter um estudo exploratório em conclusão."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
