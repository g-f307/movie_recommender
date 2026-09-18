"""Avaliação unificada e persistência dos resultados individuais B0--B5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from cinebot_ml.ranking.candidates import CandidateSet
from cinebot_ml.ranking.contracts import MovieId, RankingResult, RecommendationRequest, Recommender
from cinebot_ml.ranking.metrics import calculate_top_k_metrics, catalog_coverage_at_k
from cinebot_ml.ranking.discovery_metrics import (
    PopularityReference,
    calculate_discovery_metrics,
    rank_position_variation_at_k,
    ranking_overlap_at_k,
    ranking_repetition_at_k,
)


RELEVANCE_SOURCES = {
    "real_feedback",
    "human_judgment",
    "public_dataset",
    "synthetic_user",
    "heuristic_proxy",
}
METRIC_NAMES = (
    "precision_at_k",
    "recall_at_k",
    "f1_at_k",
    "ndcg_at_k",
    "map_at_k",
    "mrr_at_k",
    "hit_rate_at_k",
    "diversity_at_k",
    "novelty_at_k",
    "popularity_exposure_at_k",
    "popularity_bias_at_k",
    "ranking_repetition_at_k",
    "ranking_overlap_at_k",
    "rank_position_variation_at_k",
)


class RankingEvaluationError(ValueError):
    """Entradas incompatíveis com uma comparação justa de rankings."""


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RelevanceJudgments:
    source: str
    values: Mapping[MovieId, float]
    complete: bool
    version: str

    def __post_init__(self) -> None:
        if self.source not in RELEVANCE_SOURCES:
            raise RankingEvaluationError(f"Fonte de relevância desconhecida: {self.source}")
        if not self.version.strip():
            raise RankingEvaluationError("Versão da relevância deve ser informada.")
        for movie_id, value in self.values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise RankingEvaluationError(f"Julgamento inválido para movie_id={movie_id}.")

    @property
    def relevance_id(self) -> str:
        return _canonical_hash({
            "source": self.source,
            "version": self.version,
            "complete": self.complete,
            "values": sorted((str(key), float(value)) for key, value in self.values.items()),
        })[:20]

    def status_for(self, candidate_ids: Sequence[MovieId]) -> str:
        candidates = set(candidate_ids)
        judged = set(self.values)
        outside = judged - candidates
        if outside:
            raise RankingEvaluationError(f"Julgamento fora dos candidatos: {next(iter(outside))}")
        if not judged:
            return "no_judgments"
        if not self.complete or judged != candidates:
            return "incomplete_judgments"
        if not any(value > 0 for value in self.values.values()):
            return "no_relevant_items"
        return "evaluated"


@dataclass(frozen=True)
class BenchmarkUnit:
    request: RecommendationRequest
    candidates: CandidateSet
    relevance: RelevanceJudgments
    recommenders: Mapping[str, Recommender]
    agent_id: str | None = None
    persona: str | None = None
    agent_version: str | None = None
    interaction: int = 0
    state_version: int = 0
    simulation_id: str | None = None
    popularity_reference: PopularityReference | None = None

    def __post_init__(self) -> None:
        if not self.request.unit_id:
            raise RankingEvaluationError("A unidade de benchmark deve possuir unit_id.")
        methods = set(self.recommenders)
        if not methods or not methods.issubset({f"B{index}" for index in range(6)}):
            raise RankingEvaluationError("O benchmark aceita um conjunto não vazio entre B0 e B5.")
        if isinstance(self.interaction, bool) or not isinstance(self.interaction, int) or self.interaction < 0:
            raise RankingEvaluationError("interaction deve ser um inteiro não negativo.")
        if isinstance(self.state_version, bool) or not isinstance(self.state_version, int) or self.state_version < 0:
            raise RankingEvaluationError("state_version deve ser um inteiro não negativo.")
        context = (self.agent_id, self.persona, self.agent_version, self.simulation_id)
        if any(value is not None for value in context) and any(
            not isinstance(value, str) or not value.strip() for value in context
        ):
            raise RankingEvaluationError("Contexto sintético deve informar agente, persona, versão e simulação.")
        if self.agent_id is not None and self.relevance.source != "synthetic_user":
            raise RankingEvaluationError("Agentes sintéticos exigem relevance_source=synthetic_user.")
        for method, recommender in self.recommenders.items():
            candidate_set = getattr(recommender, "candidates", None)
            if (
                candidate_set is None
                or candidate_set.movie_ids != self.candidates.movie_ids
                or candidate_set.candidate_set_id != self.candidates.candidate_set_id
            ):
                raise RankingEvaluationError(
                    f"Candidatos diferentes entre métodos; comparação bloqueada em {method}."
                )
        self.relevance.status_for(self.candidates.movie_ids)


@dataclass(frozen=True)
class IndividualEvaluation:
    benchmark_id: str
    unit_id: str
    method: str
    method_version: str
    condition: str
    profile: str
    seed: int
    k: int
    candidate_set_id: str
    candidate_count: int
    relevance_source: str
    relevance_version: str
    relevance_id: str
    evaluation_status: str
    ranking_status: str
    agent_id: str | None = None
    persona: str | None = None
    agent_version: str | None = None
    interaction: int = 0
    state_version: int = 0
    simulation_id: str | None = None
    popularity_distribution_id: str | None = None
    ranked_movie_ids: tuple[MovieId, ...] = ()
    scores: tuple[float, ...] = ()
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["ranked_movie_ids"] = list(self.ranked_movie_ids)
        value["scores"] = list(self.scores)
        return value


@dataclass(frozen=True)
class BenchmarkReport:
    benchmark_id: str
    manifest: Mapping[str, Any]
    individual: tuple[IndividualEvaluation, ...]
    aggregates: tuple[Mapping[str, Any], ...]


def _benchmark_identity(
    units: Sequence[BenchmarkUnit],
    k_values: Sequence[int],
    config_sha256: str,
) -> dict[str, Any]:
    return {
        "benchmark_version": "1.0",
        "config_sha256": config_sha256,
        "k_values": list(k_values),
        "units": [
            {
                "unit_id": unit.request.unit_id,
                "request": unit.request.to_dict(),
                "candidate_manifest": unit.candidates.to_manifest(),
                "relevance_id": unit.relevance.relevance_id,
                "relevance_source": unit.relevance.source,
                "relevance_version": unit.relevance.version,
                "relevance_complete": unit.relevance.complete,
                "agent_id": unit.agent_id,
                "persona": unit.persona,
                "agent_version": unit.agent_version,
                "interaction": unit.interaction,
                "state_version": unit.state_version,
                "simulation_id": unit.simulation_id,
                "popularity_distribution_id": (
                    unit.popularity_reference.distribution_id if unit.popularity_reference else None
                ),
                "methods": {
                    method: {
                        "version": recommender.method_version,
                        "manifest": recommender.method_manifest(),
                    }
                    for method, recommender in sorted(unit.recommenders.items())
                },
            }
            for unit in units
        ],
    }


def _evaluate_result(
    benchmark_id: str,
    unit: BenchmarkUnit,
    result: RankingResult,
    k: int,
) -> IndividualEvaluation:
    ranking = tuple(item.movie_id for item in result.ranked_items)
    scores = tuple(float(item.score) for item in result.ranked_items)
    evaluation_status = unit.relevance.status_for(unit.candidates.movie_ids)
    catalog = {movie["id"]: movie for movie in unit.candidates.movies}
    discovery = calculate_discovery_metrics(ranking, catalog, k, unit.popularity_reference)
    distribution_id = discovery.pop("popularity_distribution_id")
    metrics: dict[str, float | None] = (
        {key: value for key, value in discovery.items()}
        if any(value is not None for value in discovery.values())
        else {}
    )
    if evaluation_status in {"evaluated", "no_relevant_items"}:
        metrics.update(calculate_top_k_metrics(ranking, unit.relevance.values, k))
        metrics.update({key: value for key, value in discovery.items()})
    if metrics:
        metrics.update({
            "ranking_repetition_at_k": None,
            "ranking_overlap_at_k": None,
            "rank_position_variation_at_k": None,
        })
    return IndividualEvaluation(
        benchmark_id=benchmark_id,
        unit_id=unit.request.unit_id or "",
        method=result.method,
        method_version=result.method_version,
        condition=result.condition,
        profile=result.profile,
        seed=result.seed,
        k=k,
        candidate_set_id=unit.candidates.candidate_set_id,
        candidate_count=len(unit.candidates.movies),
        relevance_source=unit.relevance.source,
        relevance_version=unit.relevance.version,
        relevance_id=unit.relevance.relevance_id,
        evaluation_status=evaluation_status,
        ranking_status=result.status,
        agent_id=unit.agent_id,
        persona=unit.persona,
        agent_version=unit.agent_version,
        interaction=unit.interaction,
        state_version=unit.state_version,
        simulation_id=unit.simulation_id,
        popularity_distribution_id=distribution_id,
        ranked_movie_ids=ranking,
        scores=scores,
        metrics=metrics,
    )


def _failure_record(
    benchmark_id: str,
    unit: BenchmarkUnit,
    method: str,
    version: str,
    k: int,
    error: Exception,
) -> IndividualEvaluation:
    return IndividualEvaluation(
        benchmark_id=benchmark_id,
        unit_id=unit.request.unit_id or "",
        method=method,
        method_version=version,
        condition=unit.request.condition,
        profile=unit.request.profile,
        seed=unit.request.seed,
        k=k,
        candidate_set_id=unit.candidates.candidate_set_id,
        candidate_count=len(unit.candidates.movies),
        relevance_source=unit.relevance.source,
        relevance_version=unit.relevance.version,
        relevance_id=unit.relevance.relevance_id,
        evaluation_status="ranking_failed",
        ranking_status="failed",
        agent_id=unit.agent_id,
        persona=unit.persona,
        agent_version=unit.agent_version,
        interaction=unit.interaction,
        state_version=unit.state_version,
        simulation_id=unit.simulation_id,
        error_type=type(error).__name__,
        error_message=str(error),
    )


def aggregate_individual_results(
    records: Sequence[IndividualEvaluation],
    units: Sequence[BenchmarkUnit],
) -> tuple[Mapping[str, Any], ...]:
    groups: dict[tuple[str, str, str, int, int, str, str, str, str | None], list[IndividualEvaluation]] = {}
    for record in records:
        key = (
            record.method,
            record.condition,
            record.profile,
            record.k,
            record.interaction,
            record.candidate_set_id,
            record.relevance_source,
            record.relevance_version,
            record.popularity_distribution_id,
        )
        groups.setdefault(key, []).append(record)
    candidates_by_id = {unit.candidates.candidate_set_id: unit.candidates.movie_ids for unit in units}
    output = []
    for key, values in sorted(groups.items(), key=lambda item: repr(item[0])):
        (
            method, condition, profile, k, interaction, candidate_set_id,
            relevance_source, relevance_version, popularity_distribution_id,
        ) = key
        successful = [value for value in values if value.ranking_status == "completed"]
        evaluated = [
            value for value in values
            if value.evaluation_status in {"evaluated", "no_relevant_items"}
        ]
        row: dict[str, Any] = {
            "method": method,
            "condition": condition,
            "profile": profile,
            "k": k,
            "interaction": interaction,
            "candidate_set_id": candidate_set_id,
            "relevance_source": relevance_source,
            "relevance_version": relevance_version,
            "popularity_distribution_id": popularity_distribution_id,
            "unit_count": len(values),
            "evaluated_unit_count": len(evaluated),
            "failed_unit_count": len(values) - len(successful),
            "no_judgment_unit_count": sum(
                value.evaluation_status in {"no_judgments", "incomplete_judgments"}
                for value in values
            ),
            "catalog_coverage_at_k": catalog_coverage_at_k(
                [value.ranked_movie_ids for value in successful],
                candidates_by_id[candidate_set_id],
                k,
            ),
        }
        for metric in METRIC_NAMES:
            observed = [value.metrics.get(metric) for value in successful]
            numeric = [float(value) for value in observed if value is not None]
            row[metric] = mean(numeric) if numeric else None
        output.append(row)
    return tuple(output)


def _add_stability(records: Sequence[IndividualEvaluation]) -> tuple[IndividualEvaluation, ...]:
    previous: dict[tuple[Any, ...], IndividualEvaluation] = {}
    output = []
    for record in records:
        key = (
            record.unit_id, record.method, record.condition, record.profile, record.seed,
            record.k, record.relevance_source, record.relevance_version,
            record.popularity_distribution_id,
        )
        prior = previous.get(key)
        updated = record
        if (
            record.ranking_status == "completed"
            and prior is not None
            and record.interaction > prior.interaction
        ):
            metrics = dict(record.metrics)
            metrics.update({
                "ranking_repetition_at_k": ranking_repetition_at_k(
                    prior.ranked_movie_ids, record.ranked_movie_ids, record.k
                ),
                "ranking_overlap_at_k": ranking_overlap_at_k(
                    prior.ranked_movie_ids, record.ranked_movie_ids, record.k
                ),
                "rank_position_variation_at_k": rank_position_variation_at_k(
                    prior.ranked_movie_ids, record.ranked_movie_ids, record.k
                ),
            })
            updated = replace(record, metrics=metrics)
        if record.ranking_status == "completed" and (
            prior is None or record.interaction > prior.interaction
        ):
            previous[key] = updated
        output.append(updated)
    return tuple(output)


def run_benchmark(
    units: Sequence[BenchmarkUnit],
    *,
    k_values: Sequence[int],
    official_k_values: Sequence[int],
    config_sha256: str,
    git_commit: str = "unavailable",
) -> BenchmarkReport:
    if not units:
        raise RankingEvaluationError("O benchmark exige ao menos uma unidade.")
    if not k_values or any(k not in official_k_values for k in k_values):
        raise RankingEvaluationError("K deve pertencer aos valores oficiais da configuração.")
    identity = {**_benchmark_identity(units, k_values, config_sha256), "git_commit": git_commit}
    benchmark_id = _canonical_hash(identity)[:20]
    records = []
    for unit in units:
        for method, recommender in sorted(unit.recommenders.items()):
            for k in k_values:
                request = replace(
                    unit.request,
                    method=method,
                    method_version=recommender.method_version,
                    candidate_movie_ids=unit.candidates.movie_ids,
                    k=k,
                    history=unit.request.history if method == "B5" else (),
                )
                try:
                    result = recommender.recommend(request)
                    records.append(_evaluate_result(benchmark_id, unit, result, k))
                except Exception as exc:  # falha parcial é evidência e não encerra as demais unidades
                    records.append(
                        _failure_record(
                            benchmark_id, unit, method, recommender.method_version, k, exc
                        )
                    )
    records = list(_add_stability(records))
    aggregates = aggregate_individual_results(records, units)
    manifest = {**identity, "benchmark_id": benchmark_id}
    return BenchmarkReport(benchmark_id, manifest, tuple(records), aggregates)


def write_benchmark_report(report: BenchmarkReport, output_root: Path) -> Mapping[str, Path]:
    """Persiste manifest, JSONL bruto e tabelas CSV sem sobrescrita silenciosa."""

    output_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output_root / f"{report.benchmark_id}.manifest.json",
        "raw": output_root / f"{report.benchmark_id}.individual.jsonl",
        "individual_table": output_root / f"{report.benchmark_id}.individual.csv",
        "aggregate_table": output_root / f"{report.benchmark_id}.aggregate.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Resultados do benchmark já existem e não serão sobrescritos.")
    paths["manifest"].write_text(
        json.dumps(report.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["raw"].write_text(
        "".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n" for item in report.individual),
        encoding="utf-8",
    )
    individual_rows = []
    for item in report.individual:
        row = item.to_dict()
        row.update(row.pop("metrics"))
        row["ranked_movie_ids"] = json.dumps(row["ranked_movie_ids"], ensure_ascii=False)
        row["scores"] = json.dumps(row["scores"])
        individual_rows.append(row)
    _write_csv(paths["individual_table"], individual_rows)
    _write_csv(paths["aggregate_table"], list(report.aggregates))
    return paths


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_benchmark_report_to_config(
    report: BenchmarkReport,
    config: Mapping[str, Any],
    project_root: Path,
) -> Mapping[str, Path]:
    """Distribui saídas pelos caminhos raw/tables/manifests da configuração."""

    from cinebot_ml.experiment_config import resolve_project_path

    paths = {
        "manifest": resolve_project_path(config["paths"]["manifests"], project_root)
        / f"{report.benchmark_id}.benchmark.json",
        "raw": resolve_project_path(config["paths"]["raw"], project_root)
        / f"{report.benchmark_id}.individual.jsonl",
        "individual_table": resolve_project_path(config["paths"]["tables"], project_root)
        / f"{report.benchmark_id}.individual.csv",
        "aggregate_table": resolve_project_path(config["paths"]["tables"], project_root)
        / f"{report.benchmark_id}.aggregate.csv",
    }
    if any(path.exists() for path in paths.values()):
        raise FileExistsError("Resultados do benchmark já existem e não serão sobrescritos.")
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(
        json.dumps(report.manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["raw"].write_text(
        "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in report.individual
        ),
        encoding="utf-8",
    )
    individual_rows = []
    for item in report.individual:
        row = item.to_dict()
        row.update(row.pop("metrics"))
        row["ranked_movie_ids"] = json.dumps(row["ranked_movie_ids"], ensure_ascii=False)
        row["scores"] = json.dumps(row["scores"])
        individual_rows.append(row)
    _write_csv(paths["individual_table"], individual_rows)
    _write_csv(paths["aggregate_table"], list(report.aggregates))
    return paths


def load_benchmark_units(
    path: Path,
    *,
    config_path: Path,
    b2_artifact_path: Path | None = None,
    b3_model_path: Path | None = None,
    b3_metadata_path: Path | None = None,
    methods: Sequence[str] | None = None,
) -> list[BenchmarkUnit]:
    """Reconstrói unidades e métodos B0--B5 a partir de JSON declarativo."""

    from cinebot_ml.experiment_config import load_config
    from cinebot_ml.personalization import StateSnapshot
    from cinebot_ml.ranking.candidates import build_candidate_set_from_config
    from cinebot_ml.ranking.content import ContentRecommender
    from cinebot_ml.ranking.incremental import IncrementalRecommender
    from cinebot_ml.ranking.popularity import PopularityRecommender
    from cinebot_ml.ranking.static_personalized import StaticPersonalizedRecommender
    from cinebot_ml.ranking.supervised import SupervisedArtifact, SupervisedRecommender
    from cinebot_ml.ranking.tfidf import TfidfArtifact, TfidfRecommender

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RankingEvaluationError(f"Não foi possível carregar unidades: {exc}") from exc
    if not isinstance(payload, Mapping) or not isinstance(payload.get("units"), list):
        raise RankingEvaluationError("Arquivo de unidades deve conter uma lista 'units'.")
    source = str(payload.get("relevance_source") or "")
    version = str(payload.get("relevance_version") or "")
    popularity_reference = None
    raw_popularity = payload.get("popularity_reference")
    if raw_popularity is not None:
        if not isinstance(raw_popularity, Mapping) or not isinstance(raw_popularity.get("counts"), list):
            raise RankingEvaluationError("popularity_reference deve conter uma lista counts.")
        counts = {}
        for item in raw_popularity["counts"]:
            if not isinstance(item, Mapping) or "movie_id" not in item or "count" not in item:
                raise RankingEvaluationError("Item inválido em popularity_reference.counts.")
            if item["movie_id"] in counts:
                raise RankingEvaluationError("movie_id duplicado em popularity_reference.counts.")
            counts[item["movie_id"]] = item["count"]
        popularity_reference = PopularityReference(
            counts, str(raw_popularity.get("partition") or ""), str(raw_popularity.get("version") or "")
        )
    enabled = tuple(methods or load_config(config_path)["methods"]["enabled"])
    invalid = sorted(set(enabled) - {f"B{index}" for index in range(6)})
    if not enabled or invalid:
        raise RankingEvaluationError(f"Método não suportado pelo benchmark: {invalid[0] if invalid else 'nenhum'}")
    if "B2" in enabled and b2_artifact_path is None:
        raise RankingEvaluationError("--b2-artifact é obrigatório quando B2 está habilitado.")
    if "B3" in enabled and (b3_model_path is None or b3_metadata_path is None):
        raise RankingEvaluationError("--b3-model e --b3-metadata são obrigatórios quando B3 está habilitado.")
    b2_artifact = TfidfArtifact.load(b2_artifact_path) if "B2" in enabled else None
    b3_artifact = (
        SupervisedArtifact.load(b3_model_path, b3_metadata_path)
        if "B3" in enabled
        else None
    )
    units = []
    for index, raw in enumerate(payload["units"]):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("request"), Mapping):
            raise RankingEvaluationError(f"Unidade {index} é inválida.")
        base_request = RecommendationRequest.from_dict(raw["request"])
        base_request = replace(base_request, candidate_movie_ids=())
        candidates = build_candidate_set_from_config(base_request, config_path)
        raw_relevance = raw.get("relevance", [])
        if not isinstance(raw_relevance, list):
            raise RankingEvaluationError(f"Relevância da unidade {index} deve ser uma lista.")
        values = {}
        for judgment in raw_relevance:
            if (
                not isinstance(judgment, Mapping)
                or "movie_id" not in judgment
                or "value" not in judgment
            ):
                raise RankingEvaluationError(f"Julgamento inválido na unidade {index}.")
            if judgment["movie_id"] in values:
                raise RankingEvaluationError(f"Julgamento duplicado na unidade {index}.")
            values[judgment["movie_id"]] = judgment["value"]
        relevance = RelevanceJudgments(
            source, values, bool(raw.get("complete", False)), version
        )
        state_snapshot = None
        initial_snapshot = None
        if set(enabled).intersection({"B4", "B5"}):
            try:
                initial_snapshot = StateSnapshot.from_dict(raw["initial_snapshot"])
                state_snapshot = StateSnapshot.from_dict(raw["state_snapshot"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RankingEvaluationError(
                    f"Unidade {index} deve informar initial_snapshot e state_snapshot para B4/B5."
                ) from exc
            if initial_snapshot.state.version != 0:
                raise RankingEvaluationError(f"Snapshot inicial da unidade {index} deve possuir versão zero.")
            if initial_snapshot.state.state_id != state_snapshot.state.state_id and state_snapshot.state.version == 0:
                raise RankingEvaluationError(f"Estado inicial divergente na unidade {index}.")
            if initial_snapshot.state.subject_id != state_snapshot.state.subject_id:
                raise RankingEvaluationError(f"Sujeito divergente na unidade {index}.")
            expected_history = tuple(event.to_dict() for event in state_snapshot.state.history)
            if base_request.history != expected_history:
                raise RankingEvaluationError(f"Histórico diverge do snapshot na unidade {index}.")
            context_fields = ("agent_id", "persona", "agent_version", "simulation_id")
            if source == "synthetic_user" and any(not raw.get(field) for field in context_fields):
                raise RankingEvaluationError(
                    f"Unidade sintética {index} deve informar agente, persona, versão e simulação."
                )
        constructors = {
            "B0": lambda: PopularityRecommender(candidates),
            "B1": lambda: ContentRecommender(candidates),
            "B2": lambda: TfidfRecommender(candidates, b2_artifact),
            "B3": lambda: SupervisedRecommender(candidates, b3_artifact),
            "B4": lambda: StaticPersonalizedRecommender(candidates, initial_snapshot.state),
            "B5": lambda: IncrementalRecommender(candidates, state_snapshot),
        }
        recommenders = {method: constructors[method]() for method in enabled}
        state_version = int(raw.get("state_version", state_snapshot.state.version if state_snapshot else 0))
        if state_snapshot is not None and state_version != state_snapshot.state.version:
            raise RankingEvaluationError(f"state_version diverge do snapshot na unidade {index}.")
        units.append(
            BenchmarkUnit(
                base_request,
                candidates,
                relevance,
                recommenders,
                agent_id=raw.get("agent_id"),
                persona=raw.get("persona"),
                agent_version=raw.get("agent_version"),
                interaction=int(raw.get("interaction", 0)),
                state_version=state_version,
                simulation_id=raw.get("simulation_id"),
                popularity_reference=popularity_reference,
            )
        )
    return units


def build_parser() -> argparse.ArgumentParser:
    from cinebot_ml.experiment_config import DEFAULT_CONFIG_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("benchmark",))
    parser.add_argument("--units", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--b2-artifact", type=Path)
    parser.add_argument("--b3-model", type=Path)
    parser.add_argument("--b3-metadata", type=Path)
    parser.add_argument("--method", action="append", choices=[f"B{i}" for i in range(6)])
    parser.add_argument("--k", type=int, action="append")
    return parser


def main() -> None:
    from cinebot_ml.config import PROJECT_ROOT
    from cinebot_ml.experiment_config import environment_diagnostic, load_config, sha256_file

    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
        k_values = args.k or config["k_values"]
        units = load_benchmark_units(
            args.units,
            config_path=args.config,
            b2_artifact_path=args.b2_artifact,
            b3_model_path=args.b3_model,
            b3_metadata_path=args.b3_metadata,
            methods=args.method or config["methods"]["enabled"],
        )
        report = run_benchmark(
            units,
            k_values=k_values,
            official_k_values=config["k_values"],
            config_sha256=sha256_file(args.config),
            git_commit=environment_diagnostic(PROJECT_ROOT)["git_commit"],
        )
        paths = write_benchmark_report_to_config(report, config, PROJECT_ROOT)
        print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))
    except (RankingEvaluationError, FileExistsError, ValueError) as exc:
        raise SystemExit(f"Erro no benchmark: {exc}") from exc


if __name__ == "__main__":
    main()
