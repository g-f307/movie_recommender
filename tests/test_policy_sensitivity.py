import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cinebot_ml.experiments.policy_sensitivity import (
    DEFAULT_POLICY_SENSITIVITY_CONFIG_PATH,
    PolicySensitivityError,
    build_sensitivity_runtime,
    execute_validation_scenarios,
    freeze_validation_selection,
    load_frozen_validation_selection,
    load_policy_sensitivity_config,
    select_validation_scenario,
    write_validation_observations,
)
from cinebot_ml.personalization import StateSnapshot, UserState
from cinebot_ml.simulation import SimulationScenario, TemporalSimulator, build_agent


def catalog():
    return [
        {
            "id": index,
            "titulo": f"Filme {index}",
            "perfil": "drama",
            "genero": "drama",
            "generos_secundarios": ["romance"],
            "ano": 2000 + index,
            "nota": 7.0,
            "votos": index * 100,
            "diretor": "Diretor",
            "palavras_chave": ["tema"],
            "sinopse": "História de teste reproduzível.",
        }
        for index in range(1, 12)
    ]


class PolicySensitivityTests(unittest.TestCase):
    @staticmethod
    def validation_rows(config):
        return [
            {
                "scenario_id": scenario.scenario_id,
                "partition": "validation",
                "comparison_id": "agent-1",
                "agent_id": "agent-1",
                "seed": 42,
                "candidate_set_id": "candidates-1",
                "ndcg_at_5": 0.60 if scenario.scenario_id == "S1" else 0.40,
                "status": "evaluated",
            }
            for scenario in config.scenarios
        ]

    def test_rejeita_cenarios_com_candidatos_diferentes_no_mesmo_par(self):
        config = load_policy_sensitivity_config()
        observations = self.validation_rows(config)
        observations[2]["candidate_set_id"] = "candidates-2"

        with self.assertRaisesRegex(PolicySensitivityError, "candidatos incompatíveis"):
            select_validation_scenario(config, observations)

    def test_rejeita_matriz_que_nao_varia_os_tres_parametros(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy_sensitivity_v2.yaml"
            payload = yaml.safe_load(DEFAULT_POLICY_SENSITIVITY_CONFIG_PATH.read_text())
            for scenario in payload["scenarios"].values():
                scenario["feedback_adjustment_weight"] = 0.50
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            path.with_suffix(".lock.json").write_text(
                json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(PolicySensitivityError, "variar os três parâmetros"):
                load_policy_sensitivity_config(path)

    def test_materializa_parametros_de_cada_cenario_sem_alterar_a_politica_v1(self):
        config = load_policy_sensitivity_config()
        scenario = next(item for item in config.scenarios if item.scenario_id == "S6")

        runtime = build_sensitivity_runtime(config, scenario)

        self.assertEqual(runtime.profile_updater.config.learning_rate, scenario.learning_rate)
        self.assertEqual(runtime.profile_updater.config.decay, scenario.decay)
        self.assertEqual(
            runtime.incremental_config.adjustment_weight,
            scenario.feedback_adjustment_weight,
        )
        self.assertEqual(runtime.incremental_config.source_sha256, config.source_sha256)

    def test_simulador_registra_as_configuracoes_materializadas_do_cenario(self):
        config = load_policy_sensitivity_config()
        scenario_config = next(item for item in config.scenarios if item.scenario_id == "S6")
        runtime = build_sensitivity_runtime(config, scenario_config)
        movies = catalog()
        agent = build_agent("consistent", 42, movies)
        initial = StateSnapshot(
            UserState(
                agent.agent_id,
                "synthetic",
                "2027-01-01T00:00:00Z",
                initial_profile={"ranked_genres": ["drama"]},
            )
        )
        scenario = SimulationScenario(
            "B5", "C1", "P1", 42, 3, "data/test.json", "a" * 64,
            initial, "2027-01-01T00:00:00Z",
        )

        result = TemporalSimulator(
            movies,
            profile_updater=runtime.profile_updater,
            incremental_config=runtime.incremental_config,
        ).run(scenario, agent)

        self.assertEqual(result.steps[0].ranking.method_version, runtime.incremental_config.version)
        self.assertEqual(
            result.manifest["profile_update"]["config_sha256"],
            config.source_sha256,
        )

    def test_executa_todos_os_cenarios_somente_na_validacao(self):
        config = load_policy_sensitivity_config()

        rows = execute_validation_scenarios(
            config,
            lambda scenario: {
                "comparison_id": "agent-1",
                "agent_id": "agent-1",
                "seed": 42,
                "candidate_set_id": "candidates-1",
                "ndcg_at_5": scenario.learning_rate,
            },
        )

        self.assertEqual(len(rows), len(config.scenarios))
        self.assertEqual({row["scenario_id"] for row in rows}, {
            scenario.scenario_id for scenario in config.scenarios
        })
        self.assertTrue(all(row["partition"] == "validation" for row in rows))
        self.assertTrue(all(row["status"] == "evaluated" for row in rows))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "observations.jsonl"
            write_validation_observations(rows, path)
            persisted = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(persisted, list(rows))

    def test_seleciona_cenario_na_validacao_e_congela_a_decisao(self):
        config = load_policy_sensitivity_config()
        observations = self.validation_rows(config)

        selection = select_validation_scenario(config, observations)
        self.assertEqual(selection["selected_scenario_id"], "S1")
        self.assertFalse(selection["holdout_used_for_selection"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            freeze_validation_selection(selection, path)
            self.assertTrue(path.exists())
            self.assertTrue(path.with_suffix(".lock.json").exists())
            recovered = load_frozen_validation_selection(config, path)
            self.assertEqual(recovered["selected_scenario_id"], "S1")


if __name__ == "__main__":
    unittest.main()
