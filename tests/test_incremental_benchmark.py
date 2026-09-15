import unittest
from dataclasses import replace

from cinebot_ml.incremental_benchmark import (
    IncrementalBenchmarkError,
    build_incremental_benchmark_units,
    run_incremental_benchmark,
)
from cinebot_ml.ranking.contracts import RankingResult, rank_scored_candidates
from cinebot_ml.simulation import TemporalSimulator
from tests.test_temporal_simulation import setup


class FixedRecommender:
    method_version = "1.0"

    def __init__(self, method, candidates, fail=False):
        self.method = method
        self.candidates = candidates
        self.fail = fail

    def recommend(self, request):
        if self.fail:
            raise RuntimeError("falha controlada B5")
        result = RankingResult(
            request.experiment_id,
            request.method,
            request.method_version,
            request.condition,
            request.profile,
            request.seed,
            request.logical_timestamp,
            "completed",
            0.0,
            rank_scored_candidates(
                [(movie_id, 1.0 - index * 0.01, {}) for index, movie_id in enumerate(request.candidate_movie_ids)],
                request.k,
            ),
            {"candidate_set_id": self.candidates.candidate_set_id},
        )
        result.validate_against(request)
        return result

    def method_manifest(self):
        return {"method": self.method, "version": self.method_version}


def trajectory(condition="C2"):
    movies, agent, scenario = setup(method="B5", condition=condition)
    result = TemporalSimulator(movies).run(scenario, agent)
    return movies, agent, result


def factories(*, fail_b5=False):
    return {
        method: (lambda candidates, method=method: FixedRecommender(method, candidates))
        for method in ("B2", "B3")
    } | {
        "B5": lambda candidates: FixedRecommender("B5", candidates, fail=fail_b5)
    }


class IncrementalBenchmarkTests(unittest.TestCase):
    def test_benchmark_somente_b0_a_b3_permanece_compativel(self):
        movies, agent, result = trajectory("C1")
        report = run_incremental_benchmark(
            result, movies, agent,
            methods=("B0", "B1", "B2", "B3"),
            factories=factories(), k_values=(5,), official_k_values=(5, 10),
            config_sha256="f" * 64,
        )
        self.assertEqual({record.method for record in report.individual}, {"B0", "B1", "B2", "B3"})

    def test_benchmark_b0_a_b5_pelo_mesmo_entrypoint(self):
        movies, agent, result = trajectory("C1")
        report = run_incremental_benchmark(
            result, movies, agent,
            methods=tuple(f"B{i}" for i in range(6)), factories=factories(),
            k_values=(5,), official_k_values=(5, 10), config_sha256="f" * 64,
        )
        self.assertEqual({record.method for record in report.individual}, {f"B{i}" for i in range(6)})

    def test_b4_e_estatico_em_multiplas_interacoes(self):
        movies, agent, result = trajectory("C2")
        units = build_incremental_benchmark_units(result, movies, agent, methods=("B4", "B5"))
        manifests = [unit.recommenders["B4"].method_manifest() for unit in units]
        self.assertEqual(
            {manifest["initial_state_id"] for manifest in manifests},
            {result.steps[0].input_snapshot.state.state_id},
        )
        self.assertEqual({manifest["state_source"] for manifest in manifests}, {"immutable_initial_profile"})

    def test_b5_recebe_versao_incremental_em_cada_interacao(self):
        movies, agent, result = trajectory("C2")
        units = build_incremental_benchmark_units(result, movies, agent, methods=("B5",))
        self.assertEqual([unit.state_version for unit in units], [0, 1, 2, 3])
        self.assertEqual(
            [unit.recommenders["B5"].method_manifest()["state_version"] for unit in units],
            [0, 1, 2, 3],
        )

    def test_mesmo_agente_e_seed_sao_preservados_entre_metodos(self):
        movies, agent, result = trajectory("C1")
        report = run_incremental_benchmark(
            result, movies, agent, methods=("B4", "B5"), k_values=(5,),
            official_k_values=(5,), config_sha256="f" * 64,
        )
        self.assertEqual({record.agent_id for record in report.individual}, {agent.agent_id})
        self.assertEqual({record.seed for record in report.individual}, {agent.seed})
        self.assertEqual({record.simulation_id for record in report.individual}, {result.simulation_id})

    def test_candidatos_divergentes_bloqueiam_comparacao(self):
        movies, agent, result = trajectory("C1")
        broken_step = replace(result.steps[0], candidate_set_id="divergente")
        broken = replace(result, steps=(broken_step,))
        with self.assertRaisesRegex(IncrementalBenchmarkError, "Candidatos divergentes"):
            build_incremental_benchmark_units(broken, movies, agent)

    def test_snapshot_futuro_e_bloqueado(self):
        movies, agent, result = trajectory("C2")
        second = result.steps[1]
        broken_step = replace(second, logical_timestamp="2027-01-01T00:00:01Z")
        broken = replace(result, steps=(result.steps[0], broken_step, *result.steps[2:]))
        with self.assertRaisesRegex(Exception, "anterior"):
            build_incremental_benchmark_units(broken, movies, agent)

    def test_fonte_de_relevancia_incorreta_e_bloqueada(self):
        movies, agent, result = trajectory("C1")
        manifest = dict(result.manifest)
        manifest["agent"] = {**manifest["agent"], "source": "heuristic_proxy"}
        with self.assertRaisesRegex(IncrementalBenchmarkError, "synthetic_user"):
            build_incremental_benchmark_units(replace(result, manifest=manifest), movies, agent)

    def test_falha_parcial_de_b5_nao_interrompe_b4(self):
        movies, agent, result = trajectory("C1")
        report = run_incremental_benchmark(
            result, movies, agent, methods=("B4", "B5"), factories=factories(fail_b5=True),
            k_values=(5,), official_k_values=(5,), config_sha256="f" * 64,
        )
        b4 = [record for record in report.individual if record.method == "B4"]
        b5 = [record for record in report.individual if record.method == "B5"]
        self.assertTrue(all(record.ranking_status == "completed" for record in b4))
        self.assertTrue(all(record.ranking_status == "failed" for record in b5))

    def test_resultados_individuais_preservam_interacoes(self):
        movies, agent, result = trajectory("C2")
        report = run_incremental_benchmark(
            result, movies, agent, methods=("B4", "B5"), k_values=(5,),
            official_k_values=(5,), config_sha256="f" * 64,
        )
        self.assertEqual(len(report.individual), 8)
        self.assertEqual({record.interaction for record in report.individual}, {0, 1, 2, 3})
        self.assertEqual({record.relevance_source for record in report.individual}, {"synthetic_user"})

    def test_agregacao_separa_metodo_condicao_e_interacao(self):
        movies, agent, result = trajectory("C2")
        report = run_incremental_benchmark(
            result, movies, agent, methods=("B4", "B5"), k_values=(5,),
            official_k_values=(5,), config_sha256="f" * 64,
        )
        keys = {(row["method"], row["condition"], row["interaction"]) for row in report.aggregates}
        self.assertEqual(keys, {(method, "C2", interaction) for method in ("B4", "B5") for interaction in range(4)})

    def test_execucao_equivalente_produz_mesmo_id(self):
        movies, agent, result = trajectory("C1")
        arguments = dict(
            methods=("B4", "B5"), k_values=(5,), official_k_values=(5,),
            config_sha256="f" * 64, git_commit="abc123",
        )
        first = run_incremental_benchmark(result, movies, agent, **arguments)
        second = run_incremental_benchmark(result, movies, agent, **arguments)
        self.assertEqual(first.benchmark_id, second.benchmark_id)

    def test_configuracao_sem_b4_ou_b5_e_aceita(self):
        movies, agent, result = trajectory("C0")
        units = build_incremental_benchmark_units(
            result, movies, agent, methods=("B0", "B1")
        )
        self.assertEqual(set(units[0].recommenders), {"B0", "B1"})


if __name__ == "__main__":
    unittest.main()
