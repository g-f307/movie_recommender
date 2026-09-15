import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cinebot_ml.personalization import FeedbackEvent, StateSnapshot, UserState
from cinebot_ml.simulation import (
    SimulationScenario,
    TemporalSimulationError,
    TemporalSimulator,
    build_agent,
    load_simulation_config,
)
from cinebot_ml.simulation.temporal import main


def catalog(size=20):
    genres = ("drama", "terror", "comedia", "acao", "romance")
    return [
        {
            "id": index,
            "titulo": f"Filme {index}",
            "perfil": genres[index % len(genres)],
            "genero": genres[index % len(genres)],
            "generos_secundarios": [genres[(index + 1) % len(genres)]],
            "ano": 1990 + index,
            "votos": index * 100,
            "nota": 6.0 + (index % 4),
            "diretor": f"Diretor {index % 3}",
            "palavras_chave": [f"tema {index % 4}"],
            "sinopse": "Uma história de teste reproduzível.",
        }
        for index in range(1, size + 1)
    ]


def setup(method="B5", condition="C1", seed=42, size=20):
    movies = catalog(size)
    agent = build_agent("consistent", seed, movies)
    initial = StateSnapshot(
        UserState(
            agent.agent_id,
            "synthetic",
            "2027-01-01T00:00:00Z",
            initial_profile={"ranked_genres": ["drama", "terror", "comedia"]},
        )
    )
    scenario = SimulationScenario(
        method,
        condition,
        "P2",
        seed,
        3,
        "data/test.json",
        "a" * 64,
        initial,
        "2027-01-01T00:00:00Z",
    )
    return movies, agent, scenario


class TemporalSimulationTests(unittest.TestCase):
    def test_condicoes_c0_a_c5_usam_quantidades_oficiais(self):
        expected = {"C0": 0, "C1": 1, "C2": 3, "C3": 5, "C4": 10, "C5": 15}
        for condition, event_count in expected.items():
            with self.subTest(condition=condition):
                movies, agent, scenario = setup(condition=condition)
                result = TemporalSimulator(movies).run(scenario, agent)
                self.assertEqual(result.status, "completed")
                self.assertEqual(len(result.steps), event_count)
                self.assertEqual(result.final_snapshot.state.version, event_count)
                self.assertIsNotNone(result.final_ranking)

    def test_passo_preserva_ranking_feedback_e_snapshots(self):
        movies, agent, scenario = setup(condition="C1")
        result = TemporalSimulator(movies).run(scenario, agent)
        step = result.steps[0]
        self.assertEqual(step.input_snapshot.state.version, 0)
        self.assertEqual(step.output_snapshot.state.version, 1)
        self.assertEqual(step.feedback_event.movie_id, step.presented_movie_ids[0])
        self.assertEqual(step.judgment.source, "synthetic_user")
        self.assertEqual(step.ranking.logical_timestamp, step.logical_timestamp)
        self.assertNotIn(step.feedback_event.event_id, step.input_snapshot.to_json())

    def test_feedback_afeta_somente_ranking_seguinte(self):
        movies, agent, scenario = setup(condition="C1")
        result = TemporalSimulator(movies).run(scenario, agent)
        step = result.steps[0]
        self.assertEqual(step.ranking.condition, "C1")
        self.assertEqual(step.input_snapshot.state.history, ())
        self.assertEqual(result.final_snapshot.state.history[-1], step.feedback_event)
        self.assertGreater(result.final_ranking.logical_timestamp, step.feedback_event.timestamp)

    def test_item_consumido_e_removido_dos_candidatos(self):
        movies, agent, scenario = setup(condition="C2")
        result = TemporalSimulator(movies).run(scenario, agent)
        selected = [step.feedback_event.movie_id for step in result.steps]
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(set(selected).isdisjoint({item.movie_id for item in result.final_ranking.ranked_items}))
        self.assertNotEqual(result.steps[0].candidate_set_id, result.steps[1].candidate_set_id)

    def test_catalogo_esgotado_preserva_sequencia_parcial(self):
        movies, agent, scenario = setup(condition="C3", size=2)
        result = TemporalSimulator(movies).run(scenario, agent)
        self.assertEqual(result.status, "exhausted")
        self.assertEqual(len(result.steps), 2)
        self.assertEqual(result.final_snapshot.state.version, 2)
        self.assertEqual(result.final_ranking.ranked_items, ())

    def test_falha_intermediaria_e_retomada_segura(self):
        movies, agent, scenario = setup(condition="C2")

        class FailingSimulator(TemporalSimulator):
            def _rank(self, scenario, snapshot, logical_time):
                if snapshot.state.version == 1:
                    raise RuntimeError("falha controlada")
                return super()._rank(scenario, snapshot, logical_time)

        checkpoints = []
        partial = FailingSimulator(movies).run(scenario, agent, checkpoint=checkpoints.append)
        self.assertEqual(partial.status, "failed")
        self.assertEqual(len(partial.steps), 1)
        self.assertIn("falha controlada", partial.failure)
        self.assertEqual(checkpoints[0].status, "partial")
        self.assertEqual(checkpoints[-1].status, "failed")
        resumed = TemporalSimulator(movies).run(scenario, agent, resume_from=partial)
        self.assertEqual(resumed.status, "completed")
        self.assertEqual(len(resumed.steps), 3)

    def test_reexecucao_e_identificadores_sao_deterministicos(self):
        movies, agent, scenario = setup(condition="C2")
        first = TemporalSimulator(movies).run(scenario, agent)
        second = TemporalSimulator(movies).run(scenario, agent)
        self.assertEqual(first.scenario_id, second.scenario_id)
        self.assertEqual(first.simulation_id, second.simulation_id)
        self.assertEqual(
            [step.output_snapshot.snapshot_id for step in first.steps],
            [step.output_snapshot.snapshot_id for step in second.steps],
        )

    def test_execucao_em_lote_aceita_multiplos_agentes_e_seeds(self):
        movies, first_agent, first_scenario = setup(condition="C1", seed=42)
        _, second_agent, second_scenario = setup(condition="C1", seed=137)
        results = TemporalSimulator(movies).run_many(
            ((first_scenario, first_agent), (second_scenario, second_agent))
        )
        self.assertEqual([result.status for result in results], ["completed", "completed"])
        self.assertEqual({result.manifest["agent"]["seed"] for result in results}, {42, 137})

    def test_b4_e_b5_partem_do_mesmo_estado_e_ranking(self):
        movies, agent, b4 = setup(method="B4", condition="C1")
        _, _, b5 = setup(method="B5", condition="C1")
        first_b4 = TemporalSimulator(movies).run(b4, agent).steps[0]
        first_b5 = TemporalSimulator(movies).run(b5, agent).steps[0]
        self.assertEqual(first_b4.input_snapshot.snapshot_id, first_b5.input_snapshot.snapshot_id)
        self.assertEqual(
            [(item.movie_id, item.score) for item in first_b4.ranking.ranked_items],
            [(item.movie_id, item.score) for item in first_b5.ranking.ranked_items],
        )

    def test_evento_futuro_e_retomada_incompativel_sao_bloqueados(self):
        movies, agent, scenario = setup(condition="C1")
        simulator = TemporalSimulator(movies)
        event = FeedbackEvent(
            "future", 1, "like", "2028-01-01T00:00:00Z", 1, "synthetic_user"
        )
        future = StateSnapshot(
            UserState(
                agent.agent_id,
                "synthetic",
                "2028-01-01T00:00:00Z",
                version=1,
                initial_profile=scenario.initial_state.state.initial_profile,
                presented_movie_ids=(1,),
                consumed_movie_ids=(1,),
                history=(event,),
            ),
            scenario.initial_state.state.state_id,
            event.event_id,
        )
        with self.assertRaisesRegex(Exception, "anterior|futuro"):
            simulator._request(scenario, future, "2027-01-01T00:00:01Z")
        other = setup(condition="C2")[2]
        partial = simulator.run(scenario, agent)
        with self.assertRaisesRegex(TemporalSimulationError, "outro cenário"):
            simulator.run(other, agent, resume_from=partial)

    def test_manifesto_reconstroi_politicas_e_fontes(self):
        movies, agent, scenario = setup(condition="C1")
        result = TemporalSimulator(movies).run(scenario, agent)
        self.assertEqual(result.manifest["temporal_policy"], "rank_then_feedback_then_update")
        self.assertEqual(result.manifest["agent"]["source"], "synthetic_user")
        self.assertEqual(result.manifest["target_events"], 1)
        self.assertEqual(result.manifest["scenario"]["initial_snapshot_id"], scenario.initial_state.snapshot_id)

    def test_comando_persiste_resultado_regeneravel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "catalog.json"
            output = root / "result.json"
            catalog_path.write_text(
                json.dumps({"perfis": {"drama": catalog(3)}}), encoding="utf-8"
            )
            exit_code = main(
                [
                    "--condition", "C0", "--profile", "P0", "--catalog",
                    str(catalog_path), "--output", str(output),
                ]
            )
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(
                hashlib.sha256(catalog_path.read_bytes()).hexdigest(),
                payload["manifest"]["scenario"]["catalog_sha256"],
            )

    def test_configuracao_oficial_esta_congelada(self):
        config = load_simulation_config()
        self.assertEqual(config.events_for("C4"), 10)
        self.assertEqual(config.events_for("C5"), config.full_history_events)


if __name__ == "__main__":
    unittest.main()
