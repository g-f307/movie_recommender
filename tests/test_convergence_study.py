import tempfile
import unittest
from pathlib import Path

from cinebot_ml.experiments.convergence import (
    _difference, load_convergence_config, run_convergence_study, write_convergence_report,
)
from cinebot_ml.experiments.cold_start import profile_payload
from cinebot_ml.personalization import StateSnapshot, UserState
from cinebot_ml.simulation import SimulationScenario, TemporalSimulator, build_agent
from tests.test_temporal_simulation import catalog


def trajectories(size=25):
    movies = catalog(size)
    agent = build_agent("consistent", 42, movies)
    initial = StateSnapshot(UserState(
        agent.agent_id, "synthetic", "2027-01-01T00:00:00Z",
        initial_profile=profile_payload(agent, "P5"),
    ))
    results = {}
    for condition in (f"C{i}" for i in range(6)):
        scenario = SimulationScenario(
            "B5", condition, "P5", 42, 5, "test.json", "a" * 64,
            initial, "2027-01-01T00:00:00Z",
        )
        results[condition] = TemporalSimulator(movies).run(scenario, agent)
    return movies, agent, results


class ConvergenceStudyTests(unittest.TestCase):
    def test_trajetoria_completa_preserva_condicoes_e_agente(self):
        movies, agent, simulations = trajectories()
        report = run_convergence_study(
            simulations, movies, agent, k=5, official_k_values=(5,),
            experiment_config_sha256="f" * 64,
        )
        self.assertEqual({row["condition"] for row in report.longitudinal}, {f"C{i}" for i in range(6)})
        self.assertEqual({row["agent_id"] for row in report.longitudinal}, {agent.agent_id})
        self.assertEqual({record.method for record in report.benchmark.individual}, {"B4", "B5"})
        self.assertEqual(len(report.curves), len(report.longitudinal))

    def test_c5_coincidente_com_c4_e_catalogo_esgotado(self):
        movies, agent, simulations = trajectories(8)
        report = run_convergence_study(
            simulations, movies, agent, k=5, official_k_values=(5,),
            experiment_config_sha256="f" * 64,
        )
        c5 = [row for row in report.longitudinal if row["condition"] == "C5"]
        self.assertTrue(all(row["sequence_status"] == "exhausted" for row in c5))
        self.assertTrue(all(row["coincident_with_previous"] for row in c5))
        self.assertTrue(all(row["actual_events"] == 8 for row in c5))

    def test_sequencia_interrompida_permanece_sem_preenchimento(self):
        movies, agent, simulations = trajectories()
        broken = simulations["C3"]
        simulations["C3"] = type(broken)(
            broken.scenario_id, broken.simulation_id, "failed", broken.target_events,
            broken.steps[:1], broken.steps[0].output_snapshot, None, broken.manifest,
            "RuntimeError: interrupção",
        )
        report = run_convergence_study(
            simulations, movies, agent, k=5, official_k_values=(5,),
            experiment_config_sha256="f" * 64,
        )
        c3 = [row for row in report.longitudinal if row["condition"] == "C3"]
        self.assertTrue(all(row["sequence_status"] == "failed" for row in c3))
        self.assertTrue(all(row["value"] is None for row in c3))

    def test_b4_estatico_e_b5_reage_ao_feedback(self):
        movies, agent, simulations = trajectories()
        report = run_convergence_study(
            simulations, movies, agent, k=5, official_k_values=(5,),
            experiment_config_sha256="f" * 64,
        )
        manifests = [unit["methods"]["B4"]["manifest"] for unit in report.benchmark.manifest["units"]]
        self.assertEqual(len({item["initial_state_id"] for item in manifests}), 1)
        rankings = {
            record.condition: record.ranked_movie_ids
            for record in report.benchmark.individual if record.method == "B5"
        }
        self.assertNotEqual(rankings["C0"], rankings["C1"])

    def test_ganhos_positivo_negativo_nulo_e_valor_marginal(self):
        self.assertAlmostEqual(_difference(0.8, 0.5), 0.3)
        self.assertAlmostEqual(_difference(0.2, 0.5), -0.3)
        self.assertEqual(_difference(0.5, 0.5), 0.0)
        self.assertIsNone(_difference(None, 0.5))

    def test_reexecucao_e_persistencia_sao_deterministicas(self):
        movies, agent, simulations = trajectories()
        kwargs = dict(k=5, official_k_values=(5,), experiment_config_sha256="f" * 64)
        first = run_convergence_study(simulations, movies, agent, **kwargs)
        second = run_convergence_study(simulations, movies, agent, **kwargs)
        self.assertEqual(first.study_id, second.study_id)
        with tempfile.TemporaryDirectory() as directory:
            paths = write_convergence_report(first, Path(directory))
            self.assertTrue(all(path.is_file() for path in paths.values()))
            with self.assertRaises(FileExistsError):
                write_convergence_report(first, Path(directory))


if __name__ == "__main__":
    unittest.main()
