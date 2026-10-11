import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cinebot_ml.experiments.closed_loop import (
    DEFAULT_CLOSED_LOOP_CONFIG_PATH,
    ClosedLoopExposureError,
    audit_closed_loop_trajectory,
    load_closed_loop_exposure_config,
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


class ClosedLoopExposureTests(unittest.TestCase):
    def test_configuracao_fechada_exige_lock_compativel(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "closed_loop_v2.yaml"
            path.write_bytes(DEFAULT_CLOSED_LOOP_CONFIG_PATH.read_bytes())
            path.with_suffix(".lock.json").write_text(
                json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}),
                encoding="utf-8",
            )
            self.assertEqual(load_closed_loop_exposure_config(path).events_for("C5"), 15)

            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ClosedLoopExposureError, "alterada sem nova versão"):
                load_closed_loop_exposure_config(path)

    def test_auditoria_rejeita_feedback_que_nao_veio_do_ranking_anterior(self):
        movies = catalog()
        config = load_closed_loop_exposure_config()
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
        result = TemporalSimulator(movies, config.as_simulation_config()).run(scenario, agent)
        self.assertTrue(audit_closed_loop_trajectory(result, config)["complete"])

        step = result.steps[0]
        alternate_movie = step.presented_movie_ids[1]
        tampered = replace(
            result,
            steps=(
                replace(
                    step,
                    feedback_event=replace(step.feedback_event, movie_id=alternate_movie),
                ),
            ),
        )
        audit = audit_closed_loop_trajectory(tampered, config)

        self.assertFalse(audit["complete"])
        self.assertEqual(audit["failures"][0]["reason"], "feedback_not_from_prior_ranking")


if __name__ == "__main__":
    unittest.main()
