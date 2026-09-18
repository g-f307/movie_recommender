import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cinebot_ml.experiments.cold_start import (
    ColdStartValidationError,
    build_cold_start_profiles,
    load_cold_start_config,
    profile_payload,
    run_cold_start_study,
    validate_profile_payload,
    write_cold_start_report,
)
from cinebot_ml.ranking.contracts import RankingResult, rank_scored_candidates
from cinebot_ml.simulation import SyntheticAgent, TemporalSimulator, build_agent
from cinebot_ml.simulation.agents import LatentPreference
from tests.test_temporal_simulation import catalog


class FixedRecommender:
    method_version = "1.0"

    def __init__(self, method, candidates):
        self.method = method
        self.candidates = candidates

    def recommend(self, request):
        result = RankingResult(
            request.experiment_id, request.method, request.method_version,
            request.condition, request.profile, request.seed,
            request.logical_timestamp, "completed", 0.0,
            rank_scored_candidates([
                (movie_id, 1.0 - index * 0.01, {})
                for index, movie_id in enumerate(request.candidate_movie_ids)
            ], request.k),
            {"candidate_set_id": self.candidates.candidate_set_id},
        )
        result.validate_against(request)
        return result

    def method_manifest(self):
        return {"method": self.method, "version": self.method_version}


def factories():
    return {
        method: (lambda candidates, method=method: FixedRecommender(method, candidates))
        for method in ("B2", "B3")
    }


def allowed_history(movies, agent, condition="C2"):
    from cinebot_ml.personalization import StateSnapshot, UserState
    from cinebot_ml.simulation import SimulationScenario

    initial = StateSnapshot(UserState(
        agent.agent_id, "synthetic", "2027-01-01T00:00:00Z",
        initial_profile=profile_payload(agent, "P5"),
    ))
    scenario = SimulationScenario(
        "B5", condition, "P5", agent.seed, 5, "test.json", "a" * 64,
        initial, "2027-01-01T00:00:00Z",
    )
    return TemporalSimulator(movies).run(scenario, agent).final_snapshot


class ColdStartStudyTests(unittest.TestCase):
    def test_conteudo_autorizado_p0_a_p5(self):
        agent = build_agent("consistent", 42, catalog())
        expected = {
            "P0": set(),
            "P1": {"ranked_genres"},
            "P2": {"ranked_genres"},
            "P3": {"ranked_genres", "decade_preference"},
            "P4": {"ranked_genres", "decade_preference", "popularity_preference"},
            "P5": {"ranked_genres", "decade_preference", "popularity_preference", "preferred_directors", "preferred_keywords"},
        }
        for profile, fields in expected.items():
            with self.subTest(profile=profile):
                self.assertEqual(set(profile_payload(agent, profile)), fields)
        self.assertEqual(profile_payload(agent, "P1")["ranked_genres"], [agent.preference.genres[0][0]])
        self.assertLessEqual(len(profile_payload(agent, "P2")["ranked_genres"]), 3)

    def test_mesmo_agente_e_preferencia_latente_entre_perfis(self):
        movies = catalog()
        agent = build_agent("consistent", 42, movies)
        profiles = build_cold_start_profiles(agent, allowed_history(movies, agent))
        self.assertEqual({profile.agent_id for profile in profiles}, {agent.agent_id})
        self.assertEqual({profile.preference_id for profile in profiles}, {agent.preference_id})
        self.assertEqual({profile.profile for profile in profiles}, {f"P{i}" for i in range(6)})
        self.assertTrue(all(profile.state.state.version == 0 for profile in profiles[:5]))
        self.assertEqual(profiles[-1].state.state.version, 3)

    def test_tentativa_de_reconstrucao_e_bloqueada(self):
        config = load_cold_start_config()
        with self.assertRaisesRegex(ColdStartValidationError, "não autorizada"):
            validate_profile_payload("P1", {"ranked_genres": ["drama"], "decade_preference": "moderno"}, config)

    def test_genero_ausente_falha_e_menos_de_tres_e_aceito(self):
        movies = catalog()
        base = build_agent("consistent", 42, movies)
        no_genres = SyntheticAgent(
            base.agent_id, base.persona, base.seed,
            replace(base.preference, genres=()), base.parameters, base.config,
        )
        with self.assertRaisesRegex(ColdStartValidationError, "gêneros"):
            build_cold_start_profiles(no_genres, allowed_history(movies, base))
        one_genre_movies = [
            dict(movie, perfil="drama", genero="drama", generos_secundarios=[])
            for movie in movies
        ]
        one_genre = build_agent("consistent", 42, one_genre_movies)
        self.assertEqual(len(profile_payload(one_genre, "P2")["ranked_genres"]), 1)

    def test_comparacao_pareada_preserva_resultados_por_perfil(self):
        movies = catalog()
        agent = build_agent("consistent", 42, movies)
        report = run_cold_start_study(
            movies, agent, condition="C1", k=5,
            methods=("B0", "B1", "B2", "B3", "B4", "B5"),
            official_k_values=(5, 10), experiment_config_sha256="f" * 64,
            factories=factories(), catalog_sha256="a" * 64,
        )
        self.assertEqual(len(report.benchmark.individual), 36)
        self.assertEqual({row.profile for row in report.benchmark.individual}, {f"P{i}" for i in range(6)})
        self.assertEqual({row.agent_id for row in report.benchmark.individual}, {agent.agent_id})
        self.assertEqual(len(report.summary), 36)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_cold_start_report(report, Path(directory))
            self.assertTrue(all(path.is_file() for path in paths.values()))
            with self.assertRaises(FileExistsError):
                write_cold_start_report(report, Path(directory))

    def test_candidatos_insuficientes_sao_registrados(self):
        movies = catalog(2)
        agent = build_agent("consistent", 42, movies)
        report = run_cold_start_study(
            movies, agent, condition="C1", k=5, methods=("B0", "B1"),
            official_k_values=(5,), experiment_config_sha256="f" * 64,
            catalog_sha256="a" * 64,
        )
        self.assertIn("insufficient_candidates", {row["status"] for row in report.availability})

    def test_reexecucao_equivalente_e_deterministica(self):
        movies = catalog()
        agent = build_agent("consistent", 42, movies)
        kwargs = dict(
            condition="C1", k=5, methods=("B0", "B1"), official_k_values=(5,),
            experiment_config_sha256="f" * 64, catalog_sha256="a" * 64,
        )
        first = run_cold_start_study(movies, agent, **kwargs)
        second = run_cold_start_study(movies, agent, **kwargs)
        self.assertEqual(first.study_id, second.study_id)
        self.assertEqual(first.benchmark.benchmark_id, second.benchmark.benchmark_id)


if __name__ == "__main__":
    unittest.main()
