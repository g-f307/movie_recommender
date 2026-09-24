"""Executa A0--A6 em validação, sem consultar o holdout oficial."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.analysis.inference import describe, paired_inference
from cinebot_ml.analysis.subgroups import holm_adjust
from cinebot_ml.config import DEFAULT_DATA_PATH, PROJECT_ROOT
from cinebot_ml.dataset import load_catalog
from cinebot_ml.experimental_splits import reconstruct_assignments
from cinebot_ml.experiments.ablation import AblationVariant, load_ablation_variants
from cinebot_ml.experiments.cold_start import profile_payload
from cinebot_ml.personalization import FeedbackEvent, ProfileUpdater, StateSnapshot, UserState
from cinebot_ml.ranking import (
    IncrementalRecommender, RecommendationRequest, build_candidate_set,
    load_incremental_config, ndcg_at_k,
)
from cinebot_ml.simulation.agents import build_agent, load_agent_config
from cinebot_ml.simulation.temporal import load_simulation_config

CONFIG = PROJECT_ROOT / "configs/experiments/ablation_execution_v1.yaml"
SCHEMA = PROJECT_ROOT / "configs/experiments/ablation_execution_v1.schema.json"
SPLIT = PROJECT_ROOT / "results/manifests/splits/movie_id_split.json"
EPOCH = datetime(2027, 2, 1, tzinfo=timezone.utc)


class AblationExecutionError(ValueError):
    """Configuração, pareamento ou evidência inválida."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logical_time(step: int) -> str:
    return (EPOCH + timedelta(minutes=step)).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class ExecutionConfig:
    version: str
    partition: str
    profile: str
    conditions: tuple[str, ...]
    personas: tuple[str, ...]
    seeds: tuple[int, ...]
    k: int
    bootstrap_resamples: int
    bootstrap_seed: int
    output_root: Path
    source_sha256: str
    snapshot: Mapping[str, Any]


def load_execution_config(path: Path = CONFIG) -> ExecutionConfig:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise AblationExecutionError(f"Configuração inválida: {exc}") from exc
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        raise AblationExecutionError(f"Configuração inválida: {error.message}")
    source_sha256 = file_digest(path)
    try:
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AblationExecutionError("Lock da execução ausente ou inválido.") from exc
    if lock.get("sha256") != source_sha256:
        raise AblationExecutionError("Configuração alterada sem novo lock.")
    unknown = set(payload["personas"]) - set(load_agent_config().personas)
    if unknown:
        raise AblationExecutionError(f"Persona desconhecida: {sorted(unknown)[0]}")
    return ExecutionConfig(
        str(payload["version"]), str(payload["partition"]), str(payload["profile"]),
        tuple(payload["conditions"]), tuple(payload["personas"]), tuple(payload["seeds"]),
        int(payload["k"]), int(payload["bootstrap_resamples"]), int(payload["bootstrap_seed"]),
        PROJECT_ROOT / payload["output_root"], source_sha256, dict(payload),
    )


def validation_catalog(source: Path = DEFAULT_DATA_PATH,
                       split_path: Path = SPLIT) -> tuple[list[dict], dict[str, Any]]:
    split = json.loads(split_path.read_text(encoding="utf-8"))
    assignments = reconstruct_assignments(split)
    catalog = [movie for movie in load_catalog(source)
               if assignments.get(str(movie["id"])) == "validation"]
    selected = {str(movie["id"]) for movie in catalog}
    test_ids = {movie_id for movie_id, partition in assignments.items() if partition == "test"}
    if not catalog or selected.intersection(test_ids):
        raise AblationExecutionError("Isolamento da validação falhou.")
    return catalog, {
        "partition": "validation", "split_manifest_id": split["manifest_id"],
        "split_sha256": file_digest(split_path),
        "source_catalog_sha256": file_digest(source),
        "movie_ids_sha256": digest(sorted(selected)), "movie_count": len(catalog),
        "official_holdout_reused": False,
    }


def initial_state(agent: Any, profile: str) -> StateSnapshot:
    state = UserState(agent.agent_id, "synthetic", logical_time(0),
                      initial_profile=profile_payload(agent, profile))
    return StateSnapshot(state)


def build_history(agent: Any, catalog: Sequence[Mapping[str, Any]], count: int,
                  initial: StateSnapshot) -> tuple[StateSnapshot, dict[Any, Mapping[str, Any]]]:
    exposure = list(catalog)
    random.Random(f"ablation-validation-v1:{agent.agent_id}:{count}").shuffle(exposure)
    if len(exposure) <= count:
        raise AblationExecutionError("Catálogo insuficiente para histórico e candidatos.")
    state, movies = initial, {}
    updater = ProfileUpdater()
    for index, movie in enumerate(exposure[:count], 1):
        judgment = agent.evaluate(movie, index - 1)
        event = FeedbackEvent(
            f"ablation-{agent.agent_id}-{count}-{index:04d}", movie["id"],
            judgment.feedback, logical_time(index), index, "synthetic_user",
            metadata={"graded_relevance": judgment.graded_relevance},
        )
        sanitized = safe_movie(movie)
        state = updater.update(state, event, sanitized, logical_timestamp=logical_time(index))
        movies[movie["id"]] = sanitized
    return state, movies


def variant_state(initial: StateSnapshot, full: StateSnapshot,
                  movies: Mapping[Any, Mapping[str, Any]],
                  variant: AblationVariant) -> StateSnapshot:
    profile = dict(initial.state.to_dict()["initial_profile"])
    removals = set(variant.removed)
    for component, keys in {
        "genres": ("ranked_genres",),
        "director": ("preferred_directors",),
        "decade_popularity": ("decade_preference", "popularity_preference"),
        "text": ("preferred_keywords",),
    }.items():
        if component in removals:
            for key in keys:
                profile.pop(key, None)
    state = StateSnapshot(UserState(initial.state.subject_id, "synthetic",
                                    logical_time(0), initial_profile=profile))
    if "history" in removals:
        return state
    updater = ProfileUpdater()
    for event in full.state.history:
        if "negative_feedback" in removals and event.feedback == "dislike":
            continue
        rebuilt = FeedbackEvent(
            event.event_id, event.movie_id, event.feedback, event.timestamp,
            state.state.version + 1, event.source,
            metadata=event.to_dict()["metadata"],
        )
        state = updater.update(state, rebuilt, movies[event.movie_id],
                               logical_timestamp=event.timestamp)
    payload = state.state.to_dict()
    learned = dict(payload["learned_preferences"])
    affinities = dict(learned.get("affinities", {}))
    if "genres" in removals:
        affinities["genres"] = {}
    if "director" in removals:
        affinities["directors"] = {}
    if "text" in removals:
        affinities["keywords"], affinities["synopsis_terms"] = {}, {}
    learned["affinities"] = affinities
    rebuilt_state = UserState(
        payload["subject_id"], payload["identity_kind"], payload["logical_timestamp"],
        version=payload["version"], initial_profile=profile, learned_preferences=learned,
        presented_movie_ids=tuple(payload["presented_movie_ids"]),
        consumed_movie_ids=tuple(payload["consumed_movie_ids"]),
        history=tuple(FeedbackEvent.from_dict(item) for item in payload["history"]),
    )
    return StateSnapshot(rebuilt_state, state.previous_state_id, state.transition_event_id)


def request_for(agent_id: str, config: ExecutionConfig, condition: str,
                seed: int, state: StateSnapshot) -> RecommendationRequest:
    return RecommendationRequest(
        "ablation-validation-v1", "protocol-v1.0", "B5",
        load_incremental_config().version, condition, config.profile, seed,
        logical_time(100), (), config.k, unit_id=agent_id,
        session_id=f"ablation-{agent_id}-{condition}",
        profile_data=state.state.to_dict()["initial_profile"],
        history=tuple(event.to_dict() for event in state.state.history),
    )


def validate_pairing(rows: Sequence[Mapping[str, Any]]) -> None:
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["comparison_id"]), []).append(row)
    expected = {f"A{index}" for index in range(7)}
    for comparison_id, values in groups.items():
        if len(values) != 7 or {str(row["variant"]) for row in values} != expected:
            raise AblationExecutionError(f"Pareamento incompleto em {comparison_id}.")
        identities = {(row["agent_id"], row["seed"], row["condition"],
                       row["candidate_set_id"], row["relevance_id"])
                      for row in values}
        if len(identities) != 1:
            raise AblationExecutionError(f"Pareamento incompatível em {comparison_id}.")


def execute_study(config: ExecutionConfig | None = None
                  ) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    config = config or load_execution_config()
    catalog, catalog_identity = validation_catalog()
    variants, variants_sha256 = load_ablation_variants()
    simulation = load_simulation_config()
    rows, invalid = [], []
    for persona in config.personas:
        for seed in config.seeds:
            agent = build_agent(persona, seed, catalog)
            for condition in config.conditions:
                comparison_id = digest([agent.agent_id, condition, config.profile, seed])[:24]
                try:
                    initial = initial_state(agent, config.profile)
                    event_count = simulation.events_for(condition)
                    full, movies = build_history(agent, catalog, event_count, initial)
                    base = request_for(agent.agent_id, config, condition, seed, full)
                    candidates = build_candidate_set(
                        catalog, base, catalog_path="validation://movie_id_split",
                        catalog_sha256=catalog_identity["movie_ids_sha256"],
                    )
                    relevance = {movie["id"]: agent.evaluate(movie, event_count).graded_relevance
                                 for movie in candidates.movies}
                    relevance_id = digest(sorted((str(key), value)
                                                 for key, value in relevance.items()))[:24]
                    unit_rows = []
                    for variant in variants:
                        state = variant_state(initial, full, movies, variant)
                        request = candidates.attach_to_request(
                            request_for(agent.agent_id, config, condition, seed, state))
                        ranking = IncrementalRecommender(candidates, state).recommend(request)
                        ranking_ids = [item.movie_id for item in ranking.ranked_items]
                        metric = ndcg_at_k(ranking_ids, relevance, config.k)
                        if metric is None:
                            raise AblationExecutionError("NDCG@5 indefinido.")
                        unit_rows.append({
                            "comparison_id": comparison_id, "variant": variant.name,
                            "variant_id": variant.variant_id, "agent_id": agent.agent_id,
                            "persona": persona, "seed": seed, "condition": condition,
                            "profile": config.profile, "k": config.k,
                            "candidate_set_id": candidates.candidate_set_id,
                            "candidate_count": len(candidates.movies),
                            "relevance_id": relevance_id, "ndcg_at_5": float(metric),
                            "ranking_id": digest(ranking_ids)[:24],
                            "removed_components": list(variant.removed), "status": "evaluated",
                        })
                    rows.extend(unit_rows)
                except Exception as exc:
                    invalid.append({
                        "comparison_id": comparison_id, "agent_id": agent.agent_id,
                        "condition": condition, "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    })
    if not rows:
        raise AblationExecutionError("Nenhuma célula válida foi produzida.")
    validate_pairing(rows)
    identity = {
        "study_version": config.version,
        "pipeline_revision": "1.0.1",
        "execution_config_sha256": config.source_sha256,
        "variants_config_sha256": variants_sha256, "catalog": catalog_identity,
        "unit_of_analysis": "synthetic_agent_condition",
        "primary_metric": "ndcg_at_5", "rows_sha256": digest(rows),
        "row_count": len(rows), "invalid_count": len(invalid),
        "official_execution_reused": False,
    }
    identity["study_id"] = digest(identity)[:24]
    return identity, rows, invalid


def analyze(rows: Sequence[Mapping[str, Any]], config: ExecutionConfig
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_pairing(rows)
    indexed = {(str(row["comparison_id"]), str(row["variant"])): row for row in rows}
    keys = sorted({key for key, variant in indexed if variant == "A0"})
    comparisons, details = [], []
    for variant in [f"A{index}" for index in range(1, 7)]:
        baseline = [float(indexed[(key, "A0")]["ndcg_at_5"]) for key in keys]
        ablated = [float(indexed[(key, variant)]["ndcg_at_5"]) for key in keys]
        differences = [right - left for left, right in zip(baseline, ablated)]
        inference = paired_inference(
            baseline, ablated, resamples=config.bootstrap_resamples,
            seed=config.bootstrap_seed,
        )
        comparisons.append({
            "comparison": f"{variant}_minus_A0", "variant": variant,
            "removed_components": indexed[(keys[0], variant)]["removed_components"],
            "a0_mean": describe(baseline)["mean"],
            "ablation_mean": describe(ablated)["mean"],
            "difference_median": describe(differences)["median"], **inference,
        })
        for key, left, right in zip(keys, baseline, ablated):
            source = indexed[(key, variant)]
            details.append({
                "comparison_id": key, "variant": variant,
                "agent_id": source["agent_id"], "persona": source["persona"],
                "condition": source["condition"], "seed": source["seed"],
                "a0_ndcg_at_5": left, "ablation_ndcg_at_5": right,
                "difference_ablation_minus_a0": right - left,
            })
    adjusted = holm_adjust([float(row["p_value_two_sided"]) for row in comparisons])
    for row, p_value in zip(comparisons, adjusted):
        row["p_value_holm"] = p_value
        low, high = row["confidence_interval_95"]
        row["interpretation"] = (
            "component_harms" if low > 0 else
            "component_helps" if high < 0 else "inconclusive"
        )
    return comparisons, details


def write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"Artefato não será sobrescrito: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue()


def effect_figure(comparisons: Sequence[Mapping[str, Any]]) -> str:
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="390">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<text x="20" y="28" font-family="sans-serif" font-size="18">Efeito da remoção em NDCG@5 (IC95%)</text>',
             '<line x1="420" y1="45" x2="420" y2="360" stroke="#555" stroke-dasharray="4 4"/>']
    for index, row in enumerate(comparisons):
        y = 75 + index * 45
        low, high = map(float, row["confidence_interval_95"])
        mean = float(row["mean_difference"])
        label = f'{row["variant"]}: {", ".join(row["removed_components"])}'
        lines.extend([
            f'<text x="20" y="{y + 5}" font-family="sans-serif" font-size="14">{label}</text>',
            f'<line x1="{420 + low * 700:.2f}" y1="{y}" x2="{420 + high * 700:.2f}" y2="{y}" stroke="#2563eb" stroke-width="3"/>',
            f'<circle cx="{420 + mean * 700:.2f}" cy="{y}" r="5" fill="#1d4ed8"/>',
        ])
    return "\n".join(lines + ["</svg>"]) + "\n"


def write_study(identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                invalid: Sequence[Mapping[str, Any]], comparisons: Sequence[Mapping[str, Any]],
                details: Sequence[Mapping[str, Any]], config: ExecutionConfig) -> Path:
    target = config.output_root / str(identity["study_id"])
    artifacts = ["cells.json", "comparisons.json", "comparisons.csv",
                 "paired_differences.csv", "effect_ci.svg", "invalid_cells.json"]
    write_new(target / "ablation.manifest.json", json.dumps(
        {**identity, "artifacts": artifacts}, ensure_ascii=False,
        indent=2, sort_keys=True) + "\n")
    write_new(target / "cells.json", json.dumps(
        list(rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_new(target / "comparisons.json", json.dumps(
        list(comparisons), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_new(target / "comparisons.csv", csv_text(comparisons))
    write_new(target / "paired_differences.csv", csv_text(details))
    write_new(target / "invalid_cells.json", json.dumps(
        list(invalid), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_new(target / "effect_ci.svg", effect_figure(comparisons))
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args(argv)
    config = load_execution_config(args.config)
    identity, rows, invalid = execute_study(config)
    comparisons, details = analyze(rows, config)
    output = write_study(identity, rows, invalid, comparisons, details, config)
    print(json.dumps({
        "study_id": identity["study_id"], "rows": len(rows),
        "invalid": len(invalid), "output": output.as_posix(),
    }))
    return 0


PRIVATE_MARKERS = ("api_key", "apikey", "credential", "email", "name", "password",
                   "phone", "secret", "token")
def safe_movie(movie: Mapping[str, Any]) -> dict[str, Any]:
    """Remove termos públicos que colidem com marcadores do contrato privado."""
    value = dict(movie)
    value["palavras_chave"] = [term for term in movie.get("palavras_chave", []) or []
                                if not any(marker in str(term).lower()
                                           for marker in PRIVATE_MARKERS)]
    value["sinopse"] = " ".join(word for word in str(movie.get("sinopse") or "").split()
                                  if not any(marker in word.lower() for marker in PRIVATE_MARKERS))
    return value


if __name__ == "__main__":
    raise SystemExit(main())
