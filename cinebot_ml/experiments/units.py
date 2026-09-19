"""Unidades sintéticas pareadas e catálogo de holdout para a matriz v1."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cinebot_ml.config import DEFAULT_DATA_PATH, PROJECT_ROOT
from cinebot_ml.dataset import load_catalog
from cinebot_ml.experiment_config import sha256_file
from cinebot_ml.experimental_splits import reconstruct_assignments
from cinebot_ml.experiments.cold_start import profile_payload
from cinebot_ml.experiments.matrix import ExperimentCell, ExperimentMatrix, load_experiment_matrix
from cinebot_ml.personalization import FeedbackEvent, ProfileUpdater, StateSnapshot, UserState
from cinebot_ml.ranking.candidates import build_candidate_set
from cinebot_ml.ranking.contracts import RecommendationRequest
from cinebot_ml.simulation.agents import build_agent, load_agent_config


HOLDOUT_CONFIG = PROJECT_ROOT / "configs/experiment_v1_holdout.yaml"
SPLIT = PROJECT_ROOT / "results/manifests/splits/movie_id_split.json"
HOLDOUT_CATALOG = PROJECT_ROOT / "results/derived/test_catalog.json"
UNITS = PROJECT_ROOT / "results/units"
EPOCH = datetime(2027, 1, 1, tzinfo=timezone.utc)
EVENTS = {"C0": 0, "C1": 1, "C2": 3, "C3": 5, "C4": 10, "C5": 10}
PRIVATE_MARKERS = ("api_key", "apikey", "credential", "email", "name", "password",
                   "phone", "secret", "token")


def _time(step: int) -> str:
    return (EPOCH + timedelta(minutes=step)).isoformat().replace("+00:00", "Z")


def build_holdout_catalog(source: Path = DEFAULT_DATA_PATH, split_path: Path = SPLIT,
                          output: Path = HOLDOUT_CATALOG) -> tuple[list[dict], str]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = reconstruct_assignments(split)
    selected: dict[int | str, dict] = {}
    for movie in load_catalog(source):
        movie_id = movie["id"]
        if assignments.get(str(movie_id)) == "test":
            selected.setdefault(movie_id, movie)
    expected = {movie_id for movie_id, partition in assignments.items() if partition == "test"}
    if {str(movie_id) for movie_id in selected} != expected:
        raise ValueError("Catálogo não cobre todos os movie_id do holdout.")
    groups: dict[str, list[dict]] = defaultdict(list)
    for movie in sorted(selected.values(), key=lambda item: str(item["id"])):
        groups[str(movie["perfil"])].append(movie)
    encoded = json.dumps({"perfis": dict(sorted(groups.items()))}, ensure_ascii=False,
                         sort_keys=True) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != encoded:
            raise FileExistsError("Catálogo de holdout existente diverge da fonte.")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return load_catalog(output), sha256_file(output)


def _unit(cell: ExperimentCell, catalog: list[dict], catalog_sha256: str) -> dict:
    agent = build_agent(cell.persona, cell.seed, catalog)
    initial = StateSnapshot(UserState(agent.agent_id, "synthetic", _time(0),
                                      initial_profile=profile_payload(agent, cell.profile)))
    state = initial
    exposure = list(catalog)
    random.Random(f"neutral-exposure-v1:{agent.agent_id}").shuffle(exposure)
    if len(exposure) <= EVENTS[cell.condition]:
        raise ValueError("Holdout insuficiente para o histórico causal.")
    updater = ProfileUpdater()
    for index, movie in enumerate(exposure[:EVENTS[cell.condition]], 1):
        judgment = agent.evaluate(movie, index - 1)
        event = FeedbackEvent(
            event_id=f"neutral-{agent.agent_id}-{index:04d}", movie_id=movie["id"],
            feedback=judgment.feedback, timestamp=_time(index), sequence=index,
            source="synthetic_user", metadata={"graded_relevance": judgment.graded_relevance},
        )
        # O contrato de estado bloqueia chaves com marcadores de segredo; termos
        # públicos do TMDB que os contenham não podem entrar nas afinidades.
        exposure_movie = dict(movie)
        exposure_movie["palavras_chave"] = [
            term for term in movie.get("palavras_chave", []) or []
            if not any(marker in str(term).lower() for marker in PRIVATE_MARKERS)
        ]
        exposure_movie["sinopse"] = " ".join(
            word for word in str(movie.get("sinopse") or "").split()
            if not any(marker in word.lower() for marker in PRIVATE_MARKERS)
        )
        state = updater.update(state, event, exposure_movie, logical_timestamp=_time(index))
    request = RecommendationRequest(
        "movie-recommender-acm-sac-2027", "protocol-v1.0", "B0", "1.0",
        cell.condition, cell.profile, cell.seed, _time(EVENTS[cell.condition] + 1),
        (), cell.k, unit_id=agent.agent_id,
        session_id=f"neutral-{agent.agent_id}",
        profile_data=profile_payload(agent, cell.profile),
        history=tuple(event.to_dict() for event in state.state.history),
    )
    candidates = build_candidate_set(catalog, request,
                                     catalog_path="results/derived/test_catalog.json",
                                     catalog_sha256=catalog_sha256)
    judgments = [{"movie_id": movie["id"],
                  "value": agent.evaluate(movie, EVENTS[cell.condition]).graded_relevance}
                 for movie in candidates.movies]
    return {
        "relevance_source": "synthetic_user", "relevance_version": "neutral-exposure-v1",
        "units": [{"request": request.to_dict(), "relevance": judgments, "complete": True,
                   "initial_snapshot": initial.to_dict(), "state_snapshot": state.to_dict(),
                   "agent_id": agent.agent_id, "persona": cell.persona,
                   "agent_version": agent.config.version,
                   "simulation_id": hashlib.sha256(cell.comparison_id.encode()).hexdigest()[:24],
                   "interaction": EVENTS[cell.condition],
                   "state_version": state.state.version}],
    }


def write_units(matrix: ExperimentMatrix, catalog: list[dict], catalog_sha256: str,
                output: Path = UNITS) -> int:
    unique = {cell.comparison_id: cell for cell in matrix.cells}
    output.mkdir(parents=True, exist_ok=True)
    for comparison_id, cell in sorted(unique.items()):
        path = output / f"{comparison_id}.json"
        encoded = json.dumps(_unit(cell, catalog, catalog_sha256),
                             ensure_ascii=False, sort_keys=True) + "\n"
        if path.exists():
            if path.read_text(encoding="utf-8") != encoded:
                raise FileExistsError(f"Unidade divergente já existente: {path}")
            continue
        path.write_text(encoded, encoding="utf-8")
    return len(unique)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    from cinebot_ml.experiments.freeze import pilot_matrix
    catalog, digest = build_holdout_catalog()
    matrix = pilot_matrix() if args.pilot else load_experiment_matrix(
        experiment_config_path=HOLDOUT_CONFIG)
    print(json.dumps({"catalog_sha256": digest,
                      "units": write_units(matrix, catalog, digest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
