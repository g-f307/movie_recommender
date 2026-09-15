import tempfile
import unittest
from pathlib import Path

from cinebot_ml.ranking import CandidateSet, EligibilityPolicy, RecommendationRequest
from cinebot_ml.ranking.contracts import RankingResult, rank_scored_candidates
from cinebot_ml.ranking.evaluation import (
    BenchmarkUnit,
    RankingEvaluationError,
    RelevanceJudgments,
    run_benchmark,
    write_benchmark_report,
    write_benchmark_report_to_config,
)


def candidates(ids=(1, 2, 3)):
    movies = tuple({"id": item, "titulo": f"Filme {item}"} for item in ids)
    return CandidateSet(
        candidate_set_id="set-" + "-".join(map(str, ids)),
        version="1.0",
        catalog_path="data/filmes.json",
        catalog_sha256="e" * 64,
        policy=EligibilityPolicy(),
        filters={},
        movies=movies,
        total_catalog_items=len(movies),
    )


def request():
    return RecommendationRequest(
        experiment_id="benchmark-test",
        protocol_version="protocol-v1.0",
        method="B0",
        method_version="1.0",
        condition="C0",
        profile="P0",
        seed=42,
        logical_timestamp="2026-09-15T07:00:00Z",
        candidate_movie_ids=(),
        k=5,
        unit_id="synthetic-unit-001",
    )


class FixedRecommender:
    method_version = "1.0"

    def __init__(self, method, candidate_set, order=None, fail=False):
        self.method = method
        self.candidates = candidate_set
        self.order = order or candidate_set.movie_ids
        self.fail = fail

    def recommend(self, req):
        if self.fail:
            raise RuntimeError("falha controlada")
        scores = {movie_id: 1.0 - index * 0.1 for index, movie_id in enumerate(self.order)}
        result = RankingResult(
            experiment_id=req.experiment_id,
            method=req.method,
            method_version=req.method_version,
            condition=req.condition,
            profile=req.profile,
            seed=req.seed,
            logical_timestamp=req.logical_timestamp,
            status="completed",
            latency_ms=0.0,
            ranked_items=rank_scored_candidates(
                [(movie_id, scores[movie_id], {}) for movie_id in req.candidate_movie_ids], req.k
            ),
            metadata={"candidate_set_id": self.candidates.candidate_set_id},
        )
        result.validate_against(req)
        return result

    def method_manifest(self):
        return {"method": self.method, "version": self.method_version}


def unit(*, relevance=None, complete=True, failing_method=None, candidate_set=None):
    candidate_set = candidate_set or candidates()
    recommenders = {
        method: FixedRecommender(method, candidate_set, fail=method == failing_method)
        for method in ("B0", "B1", "B2", "B3")
    }
    return BenchmarkUnit(
        request=request(),
        candidates=candidate_set,
        relevance=RelevanceJudgments(
            source="synthetic_user",
            values=relevance if relevance is not None else {1: 1, 2: 0, 3: 0},
            complete=complete,
            version="synthetic-v1",
        ),
        recommenders=recommenders,
    )


class RankingEvaluationTests(unittest.TestCase):
    def test_executa_b0_a_b3_sob_mesmos_candidatos_e_k(self):
        report = run_benchmark(
            [unit()], k_values=[5, 10], official_k_values=[5, 10], config_sha256="f" * 64
        )
        self.assertEqual(len(report.individual), 8)
        self.assertEqual({item.method for item in report.individual}, {"B0", "B1", "B2", "B3"})
        self.assertEqual({item.k for item in report.individual}, {5, 10})
        self.assertTrue(all(item.candidate_set_id == "set-1-2-3" for item in report.individual))

    def test_unidade_sem_julgamento_e_identificada_sem_metricas(self):
        report = run_benchmark(
            [unit(relevance={})], k_values=[5], official_k_values=[5, 10], config_sha256="f" * 64
        )
        self.assertTrue(all(item.evaluation_status == "no_judgments" for item in report.individual))
        self.assertTrue(all(item.metrics == {} for item in report.individual))

    def test_julgamento_incompleto_nao_vira_zero(self):
        report = run_benchmark(
            [unit(relevance={1: 1}, complete=False)],
            k_values=[5], official_k_values=[5], config_sha256="f" * 64,
        )
        self.assertTrue(
            all(item.evaluation_status == "incomplete_judgments" for item in report.individual)
        )
        self.assertTrue(all(not item.metrics for item in report.individual))

    def test_falha_parcial_e_preservada_e_demais_metodos_continuam(self):
        report = run_benchmark(
            [unit(failing_method="B2")], k_values=[5], official_k_values=[5], config_sha256="f" * 64
        )
        failed = [item for item in report.individual if item.ranking_status == "failed"]
        completed = [item for item in report.individual if item.ranking_status == "completed"]
        self.assertEqual(len(failed), 1)
        self.assertEqual(failed[0].method, "B2")
        self.assertEqual(failed[0].error_type, "RuntimeError")
        self.assertEqual(len(completed), 3)

    def test_candidatos_diferentes_bloqueiam_comparacao(self):
        common = candidates()
        recommenders = {
            method: FixedRecommender(method, common if method != "B3" else candidates((1, 2)))
            for method in ("B0", "B1", "B2", "B3")
        }
        with self.assertRaisesRegex(RankingEvaluationError, "Candidatos diferentes"):
            BenchmarkUnit(
                request=request(), candidates=common,
                relevance=RelevanceJudgments("synthetic_user", {1: 1, 2: 0, 3: 0}, True, "v1"),
                recommenders=recommenders,
            )

    def test_k_fora_da_configuracao_e_bloqueado(self):
        with self.assertRaisesRegex(RankingEvaluationError, "valores oficiais"):
            run_benchmark(
                [unit()], k_values=[7], official_k_values=[5, 10], config_sha256="f" * 64
            )

    def test_execucao_equivalente_produz_mesmo_id(self):
        first = run_benchmark(
            [unit()], k_values=[5], official_k_values=[5], config_sha256="f" * 64
        )
        second = run_benchmark(
            [unit()], k_values=[5], official_k_values=[5], config_sha256="f" * 64
        )
        self.assertEqual(first.benchmark_id, second.benchmark_id)

    def test_agregacao_nao_substitui_resultados_individuais(self):
        report = run_benchmark(
            [unit()], k_values=[5], official_k_values=[5], config_sha256="f" * 64
        )
        self.assertEqual(len(report.individual), 4)
        self.assertEqual(len(report.aggregates), 4)
        self.assertTrue(all(row["evaluated_unit_count"] == 1 for row in report.aggregates))
        self.assertTrue(all(row["catalog_coverage_at_k"] == 1.0 for row in report.aggregates))

    def test_resultados_brutos_manifest_e_tabelas_sao_persistidos(self):
        report = run_benchmark(
            [unit()], k_values=[5], official_k_values=[5], config_sha256="f" * 64
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = write_benchmark_report(report, Path(directory))
            self.assertEqual(set(paths), {"manifest", "raw", "individual_table", "aggregate_table"})
            self.assertTrue(all(path.is_file() for path in paths.values()))
            raw_lines = paths["raw"].read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(raw_lines), len(report.individual))
            with self.assertRaisesRegex(FileExistsError, "não serão sobrescritos"):
                write_benchmark_report(report, Path(directory))

    def test_saidas_seguem_diretorios_da_configuracao(self):
        report = run_benchmark(
            [unit()], k_values=[5], official_k_values=[5], config_sha256="f" * 64
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {"paths": {"manifests": "manifests", "raw": "raw", "tables": "tables"}}
            paths = write_benchmark_report_to_config(report, config, root)
            self.assertEqual(paths["manifest"].parent, root / "manifests")
            self.assertEqual(paths["raw"].parent, root / "raw")
            self.assertEqual(paths["individual_table"].parent, root / "tables")


if __name__ == "__main__":
    unittest.main()
