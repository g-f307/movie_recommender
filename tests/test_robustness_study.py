import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
import json

from cinebot_ml.experiments.robustness import load_robustness_config, make_case, run_robustness, write_robustness_report


MOVIES = tuple({"id": i, "generos": ["rare" if i == 0 else "common"], "votos": i * 10,
                "diretor": "d", "sinopse": "s"} for i in range(10))
PROFILE = {"ranked_genres": ["common"]}
EVENTS = ({"feedback": "like"}, {"feedback": "dislike"})


class RobustnessStudyTests(unittest.TestCase):
    def test_cenarios_seeds_e_perturbacoes(self):
        scenarios, seeds, _ = load_robustness_config()
        self.assertEqual(len(seeds), 5)
        by_name = {scenario.name: scenario for scenario in scenarios}
        self.assertEqual(len(by_name), 12)
        case = lambda name, seed=42: make_case(by_name[name], seed, "agent", MOVIES, PROFILE, EVENTS)
        self.assertEqual(case("nominal").catalog, MOVIES)
        self.assertEqual(case("extreme_negative").feedback[0]["feedback"], "dislike")
        self.assertEqual(case("contradictory_feedback").feedback[0]["feedback"], "dislike")
        self.assertEqual(case("specific_preference").profile["ranked_genres"], ["common"])
        self.assertEqual(set(case("broad_preference").profile["ranked_genres"]), {"common", "rare"})
        self.assertEqual(case("rare_genre").profile["ranked_genres"], ["rare"])
        self.assertEqual(len(case("few_candidates").catalog), 5)
        self.assertEqual(len(case("popularity_concentration").catalog), 2)
        self.assertEqual(len(case("long_tail").catalog), 2)
        self.assertTrue(any("diretor" not in movie for movie in case("missing_metadata").catalog))
        self.assertNotEqual(case("catalog_shift").catalog, case("nominal").catalog)
        self.assertEqual(case("random_feedback").case_id, case("random_feedback").case_id)
        self.assertNotEqual(case("random_feedback", 42).case_id, case("random_feedback", 137).case_id)

    def test_falha_parcial_e_marcacao_exploratoria(self):
        def runner(case):
            if case.scenario.name == "few_candidates":
                raise RuntimeError("insuficiente")
            return {"ndcg_at_k": 0.5}
        rows = run_robustness(MOVIES, "agent", PROFILE, EVENTS, runner)
        self.assertEqual(len(rows), 60)
        self.assertEqual(sum(row["status"] == "failed" for row in rows), 5)
        self.assertEqual({row["evidence"] for row in rows}, {"exploratory_synthetic"})
        self.assertEqual(rows, run_robustness(MOVIES, "agent", PROFILE, EVENTS, runner))
        with TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "robustness.jsonl"
            write_robustness_report(rows, target)
            persisted = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(persisted), 60)
            self.assertEqual(sum(row["status"] == "failed" for row in persisted), 5)

    def test_generos_do_catalogo_real(self):
        scenarios, _, _ = load_robustness_config()
        by_name = {scenario.name: scenario for scenario in scenarios}
        catalog = ({"id": 1, "perfil": "Ação", "generos_secundarios": []},
                   {"id": 2, "perfil": "Comédia", "generos_secundarios": []},
                   {"id": 3, "perfil": "Comédia", "generos_secundarios": []})
        broad = make_case(by_name["broad_preference"], 42, "a", catalog, PROFILE, EVENTS)
        rare = make_case(by_name["rare_genre"], 42, "a", catalog, PROFILE, EVENTS)
        self.assertEqual(len(broad.profile["ranked_genres"]), 2)
        self.assertEqual(rare.profile["ranked_genres"], ["acao"])


if __name__ == "__main__":
    unittest.main()
