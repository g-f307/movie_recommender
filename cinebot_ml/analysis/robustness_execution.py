"""Executa a matriz exploratória de robustez B4 × B5 em validação."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

from cinebot_ml.analysis.ablation_execution import (
    file_digest,
    logical_time,
    safe_movie,
    validation_catalog,
    write_new,
)
from cinebot_ml.analysis.inference import describe, paired_inference
from cinebot_ml.analysis.subgroups import holm_adjust
from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.experiments.cold_start import profile_payload
from cinebot_ml.experiments.robustness import RobustnessCase, load_robustness_config, make_case
from cinebot_ml.personalization import FeedbackEvent, ProfileUpdater, StateSnapshot, UserState
from cinebot_ml.ranking import (
    IncrementalRecommender,
    RecommendationRequest,
    StaticPersonalizedRecommender,
    build_candidate_set,
    load_incremental_config,
    load_static_personalized_config,
    ndcg_at_k,
    ranking_overlap_at_k,
)
from cinebot_ml.simulation.agents import build_agent, load_agent_config
from cinebot_ml.simulation.temporal import load_simulation_config

OUTPUT_ROOT = PROJECT_ROOT / "results/derived/specialized_v1/robustness"
PIPELINE_REVISION = "1.0.2"


class RobustnessExecutionError(ValueError):
    """Execução, pareamento ou artefato de robustez inválido."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _initial(agent: Any, profile: Mapping[str, Any]) -> StateSnapshot:
    return StateSnapshot(UserState(agent.agent_id, "synthetic", logical_time(0),
                                   initial_profile=profile))


def _source_history(agent: Any, catalog: Sequence[Mapping[str, Any]], count: int
                    ) -> tuple[list[dict[str, Any]], dict[Any, dict[str, Any]]]:
    exposure = list(catalog)
    random.Random(f"robustness-validation-v1:{agent.agent_id}").shuffle(exposure)
    events, movies = [], {}
    for index, movie in enumerate(exposure[:count], 1):
        judgment = agent.evaluate(movie, index - 1)
        event = FeedbackEvent(
            f"robustness-{agent.agent_id}-{index:04d}", movie["id"],
            judgment.feedback, logical_time(index), index, "synthetic_user",
            metadata={"graded_relevance": judgment.graded_relevance},
        )
        events.append(event.to_dict())
        movies[movie["id"]] = safe_movie(movie)
    return events, movies


def _state(agent_id: str, profile: Mapping[str, Any],
           feedback: Sequence[Mapping[str, Any]],
           movies: Mapping[Any, Mapping[str, Any]]) -> StateSnapshot:
    state = StateSnapshot(UserState(agent_id, "synthetic", logical_time(0),
                                    initial_profile=profile))
    updater = ProfileUpdater()
    for index, raw in enumerate(feedback, 1):
        payload = dict(raw)
        payload["sequence"] = index
        event = FeedbackEvent.from_dict(payload)
        state = updater.update(state, event, movies[event.movie_id],
                               logical_timestamp=event.timestamp)
    return state


def _request(agent_id: str, method: str, version: str, seed: int,
             profile: Mapping[str, Any], history: Sequence[Mapping[str, Any]],
             candidate_ids: tuple[Any, ...] = ()) -> RecommendationRequest:
    return RecommendationRequest(
        "robustness-validation-v1", "protocol-v1.0", method, version,
        "C5", "P5", seed, logical_time(100), candidate_ids, 5,
        unit_id=agent_id, session_id=f"robustness-{agent_id}-{seed}",
        profile_data=profile, history=tuple(history),
    )


def _case_candidates(case: RobustnessCase, catalog_sha256: str,
                     consumed: set[Any]):
    eligible = [movie for movie in case.catalog if movie["id"] not in consumed]
    request = _request(case.agent_id, "B5", load_incremental_config().version,
                       case.seed, case.profile, ())
    return build_candidate_set(
        eligible, request, catalog_path=f"validation://robustness/{case.scenario.name}",
        catalog_sha256=catalog_sha256,
    )


def validate_pairs(rows: Sequence[Mapping[str, Any]], scenario_names: Sequence[str]) -> None:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["unit_id"]), str(row["scenario"])), []).append(row)
    available = {scenario for _, scenario in grouped}
    missing = set(scenario_names) - available
    if missing:
        raise RobustnessExecutionError(
            f"Execução não possui cenários: {sorted(missing)}")
    for key, values in grouped.items():
        if len(values) != 2 or {row["method"] for row in values} != {"B4", "B5"}:
            raise RobustnessExecutionError(f"Par B4/B5 incompleto em {key}.")
        identities = {(row["candidate_set_id"], row["relevance_id"],
                       row["candidate_count"]) for row in values}
        if len(identities) != 1:
            raise RobustnessExecutionError(f"Candidatos incompatíveis em {key}.")


def execute() -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    catalog, catalog_identity = validation_catalog()
    scenarios, seeds, config_sha256 = load_robustness_config()
    personas = tuple(sorted(load_agent_config().personas))
    event_count = load_simulation_config().events_for("C5")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for persona in personas:
        for seed in seeds:
            agent = build_agent(persona, seed, catalog)
            unit_id = digest([agent.agent_id, seed])[:24]
            base_profile = profile_payload(agent, "P5")
            source_feedback, movies = _source_history(agent, catalog, event_count)
            consumed = set(movies)
            for scenario in scenarios:
                try:
                    case = make_case(scenario, seed, agent.agent_id, catalog,
                                     base_profile, source_feedback)
                    state = _state(agent.agent_id, case.profile, case.feedback, movies)
                    candidates = _case_candidates(
                        case, digest([catalog_identity["movie_ids_sha256"],
                                      scenario.name, case.case_id]), consumed)
                    if not candidates.movies:
                        raise RobustnessExecutionError("Cenário sem candidatos elegíveis.")
                    relevance = {
                        movie["id"]: agent.evaluate(movie, event_count).graded_relevance
                        for movie in candidates.movies
                    }
                    relevance_id = digest(sorted((str(key), value)
                                                 for key, value in relevance.items()))[:24]
                    b4_request = candidates.attach_to_request(_request(
                        agent.agent_id, "B4", load_static_personalized_config().version,
                        seed, case.profile, case.feedback))
                    b5_request = candidates.attach_to_request(_request(
                        agent.agent_id, "B5", load_incremental_config().version,
                        seed, case.profile, case.feedback))
                    rankings = {
                        "B4": StaticPersonalizedRecommender(
                            candidates, _initial(agent, case.profile).state).recommend(b4_request),
                        "B5": IncrementalRecommender(candidates, state).recommend(b5_request),
                    }
                    for method, result in rankings.items():
                        ranking_ids = [item.movie_id for item in result.ranked_items]
                        metric = ndcg_at_k(ranking_ids, relevance, 5)
                        if metric is None:
                            raise RobustnessExecutionError("NDCG@5 indefinido.")
                        rows.append({
                            "unit_id": unit_id, "case_id": case.case_id,
                            "scenario": scenario.name, "scenario_kind": scenario.kind,
                            "intensity": scenario.intensity, "method": method,
                            "agent_id": agent.agent_id, "persona": persona, "seed": seed,
                            "candidate_set_id": candidates.candidate_set_id,
                            "candidate_count": len(candidates.movies),
                            "insufficient_candidates": len(candidates.movies) < 5,
                            "relevance_id": relevance_id, "ndcg_at_5": float(metric),
                            "ranking": ranking_ids, "ranking_id": digest(ranking_ids)[:24],
                            "status": "evaluated",
                        })
                except Exception as exc:
                    failures.append({
                        "unit_id": unit_id, "agent_id": agent.agent_id,
                        "persona": persona, "seed": seed, "scenario": scenario.name,
                        "error_type": type(exc).__name__, "error_message": str(exc),
                    })
    if not rows:
        raise RobustnessExecutionError("Nenhum par de robustez foi produzido.")

    validate_pairs(rows, [scenario.name for scenario in scenarios])
    identity = {
        "study_version": "1.0", "pipeline_revision": PIPELINE_REVISION,
        "robustness_config_sha256": config_sha256, "catalog": catalog_identity,
        "scenario_count": len(scenarios), "unit_count": len(personas) * len(seeds),
        "row_count": len(rows), "failure_count": len(failures),
        "methods": ["B4", "B5"], "metric": "ndcg_at_5",
        "evidence": "exploratory_synthetic", "official_holdout_reused": False,
        "rows_sha256": digest(rows),
    }
    identity["study_id"] = digest(identity)[:24]
    return identity, rows, failures


def analyze(rows: Sequence[Mapping[str, Any]]
            ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario_names = sorted({str(row["scenario"]) for row in rows})
    validate_pairs(rows, scenario_names)
    indexed = {(str(row["unit_id"]), str(row["scenario"]), str(row["method"])): row
               for row in rows}
    summaries, details = [], []
    for scenario in scenario_names:
        units = sorted({str(row["unit_id"]) for row in rows
                        if row["scenario"] == scenario})
        b4 = [float(indexed[(unit, scenario, "B4")]["ndcg_at_5"]) for unit in units]
        b5 = [float(indexed[(unit, scenario, "B5")]["ndcg_at_5"]) for unit in units]
        inference = paired_inference(b4, b5, resamples=10_000, seed=42)
        differences = [right - left for left, right in zip(b4, b5)]
        summaries.append({
            "scenario": scenario, "b4_mean": describe(b4)["mean"],
            "b5_mean": describe(b5)["mean"],
            "difference_median": describe(differences)["median"], **inference,
        })
        for unit, left, right in zip(units, b4, b5):
            b4_row = indexed[(unit, scenario, "B4")]
            b5_row = indexed[(unit, scenario, "B5")]
            nominal_b4 = indexed[(unit, "nominal", "B4")]
            nominal_b5 = indexed[(unit, "nominal", "B5")]
            details.append({
                "unit_id": unit, "scenario": scenario,
                "agent_id": b4_row["agent_id"], "persona": b4_row["persona"],
                "seed": b4_row["seed"], "candidate_set_id": b4_row["candidate_set_id"],
                "b4_ndcg_at_5": left, "b5_ndcg_at_5": right,
                "difference_b5_minus_b4": right - left,
                "ranking_overlap_at_5": ranking_overlap_at_k(
                    b4_row["ranking"], b5_row["ranking"], 5),
                "b4_overlap_vs_nominal_at_5": ranking_overlap_at_k(
                    nominal_b4["ranking"], b4_row["ranking"], 5),
                "b5_overlap_vs_nominal_at_5": ranking_overlap_at_k(
                    nominal_b5["ranking"], b5_row["ranking"], 5),
            })
    adjusted = holm_adjust([float(row["p_value_two_sided"]) for row in summaries])
    nominal = {row["unit_id"]: row["difference_b5_minus_b4"]
               for row in details if row["scenario"] == "nominal"}
    for row, corrected in zip(summaries, adjusted):
        row["p_value_holm"] = corrected
        scenario_rows = [item for item in details
                         if item["scenario"] == row["scenario"]]
        common_units = [item["unit_id"] for item in scenario_rows
                        if item["unit_id"] in nominal]
        scenario_by_unit = {item["unit_id"]: item["difference_b5_minus_b4"]
                            for item in scenario_rows}
        comparison = paired_inference(
            [nominal[unit] for unit in common_units],
            [scenario_by_unit[unit] for unit in common_units],
            resamples=10_000, seed=42)
        row["versus_nominal"] = comparison
        low, high = row["confidence_interval_95"]
        row["classification"] = (
            "b5_better" if low > 0 else "b5_worse" if high < 0 else "inconclusive")
    nominal_adjusted = holm_adjust([
        float(row["versus_nominal"]["p_value_two_sided"]) for row in summaries])
    for row, corrected in zip(summaries, nominal_adjusted):
        row["versus_nominal"]["p_value_holm"] = corrected

    return summaries, details


def csv_text(rows: Sequence[Mapping[str, Any]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def heatmap(summaries: Sequence[Mapping[str, Any]]) -> str:
    cell_width, cell_height = 250, 34
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="470">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="25" font-family="sans-serif" font-size="18">NDCG@5 por método e cenário</text>',
    ]
    for index, row in enumerate(summaries):
        y = 45 + index * cell_height
        lines.append(f'<text x="20" y="{y + 22}" font-family="sans-serif" font-size="12">{row["scenario"]}</text>')
        for column, method in enumerate(("B4", "B5")):
            value = float(row[f"{method.lower()}_mean"])
            red = round(255 * (1 - value)); green = round(210 * value + 30)
            x = 260 + column * cell_width
            lines.extend([
                f'<rect x="{x}" y="{y}" width="{cell_width - 5}" height="{cell_height - 3}" fill="rgb({red},{green},110)"/>',
                f'<text x="{x + 90}" y="{y + 22}" font-family="sans-serif" font-size="13">{method}: {value:.4f}</text>',
            ])
    return "\n".join(lines + ["</svg>"]) + "\n"


def distribution_figure(details: Sequence[Mapping[str, Any]]) -> str:
    grouped: dict[str, list[float]] = {}
    for row in details:
        grouped.setdefault(str(row["scenario"]), []).append(
            float(row["difference_b5_minus_b4"]))
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="470">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="25" font-family="sans-serif" font-size="18">Distribuição B5 − B4 em NDCG@5</text>',
        '<line x1="500" y1="40" x2="500" y2="455" stroke="#555" stroke-dasharray="4 4"/>',
    ]
    for index, scenario in enumerate(sorted(grouped)):
        values = sorted(grouped[scenario])
        q1, median, q3 = (describe(values)["q1"], describe(values)["median"],
                          describe(values)["q3"])
        y = 55 + index * 33
        lines.extend([
            f'<text x="20" y="{y + 5}" font-family="sans-serif" font-size="12">{scenario}</text>',
            f'<line x1="{500 + q1 * 650:.2f}" y1="{y}" x2="{500 + q3 * 650:.2f}" y2="{y}" stroke="#2563eb" stroke-width="5"/>',
            f'<circle cx="{500 + median * 650:.2f}" cy="{y}" r="5" fill="#1d4ed8"/>',
        ])
    return "\n".join(lines + ["</svg>"]) + "\n"


def write_study(identity: Mapping[str, Any], rows: Sequence[Mapping[str, Any]],
                failures: Sequence[Mapping[str, Any]], summaries: Sequence[Mapping[str, Any]],
                details: Sequence[Mapping[str, Any]]) -> Path:
    target = OUTPUT_ROOT / str(identity["study_id"])
    artifacts = ["cells.json", "scenario_report.json", "scenario_report.csv",
                 "paired_details.csv", "failures.json", "method_scenario_heatmap.svg",
                 "difference_distributions.svg"]
    write_new(target / "robustness.manifest.json", json.dumps(
        {**identity, "artifacts": artifacts}, ensure_ascii=False,
        indent=2, sort_keys=True) + "\n")
    write_new(target / "cells.json", json.dumps(
        list(rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_new(target / "scenario_report.json", json.dumps(
        list(summaries), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_new(target / "scenario_report.csv", csv_text(summaries))
    write_new(target / "paired_details.csv", csv_text(details))
    write_new(target / "failures.json", json.dumps(
        list(failures), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    write_new(target / "method_scenario_heatmap.svg", heatmap(summaries))
    write_new(target / "difference_distributions.svg", distribution_figure(details))
    return target


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    identity, rows, failures = execute()
    summaries, details = analyze(rows)
    output = write_study(identity, rows, failures, summaries, details)
    print(json.dumps({
        "study_id": identity["study_id"], "rows": len(rows),
        "failures": len(failures), "output": output.as_posix(),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
