import tempfile
import unittest
from pathlib import Path

from cinebot_ml.analysis.loader import MetricRecord
from cinebot_ml.analysis.synthesis import (
    SynthesisError, cold_start_summary, convergence_summary, write_outputs,
)


def record(agent, method, condition, profile, value):
    identity = f"{agent}-{condition}-{profile}"
    return MetricRecord(identity + method, identity, method, condition, profile,
                        "persona", 42, 1, 5, agent, "evaluated", value)


def sample_records():
    rows = []
    for agent_index in range(3):
        agent = f"agent-{agent_index}"
        for method_index in range(6):
            method = f"B{method_index}"
            for condition_index in range(6):
                for profile_index in range(6):
                    value = 0.1 + method_index / 100 + condition_index / 1000 + profile_index / 10000
                    rows.append(record(agent, method, f"C{condition_index}", f"P{profile_index}", value))
    return rows


class SynthesisTests(unittest.TestCase):
    def test_cold_start_cobre_perfis_metodos_e_preserva_agente(self):
        rows = cold_start_summary(sample_records())
        self.assertEqual(len(rows), 36)
        self.assertEqual({row["profile"] for row in rows}, {f"P{i}" for i in range(6)})
        self.assertTrue(all(row["n_units"] == 3 for row in rows))

    def test_convergencia_cobre_trajetoria_e_condiciona_post_testes(self):
        rows, inference = convergence_summary(sample_records())
        self.assertEqual(len(rows), 12)
        self.assertEqual(inference["n_longitudinal_units"], 3)
        self.assertEqual(inference["conditions"], [f"C{i}" for i in range(6)])
        self.assertEqual(inference["post_tests_performed"], inference["p_value"] < 0.05)

    def test_rejeita_trajetoria_incompleta(self):
        rows = sample_records()
        rows = [row for row in rows if not (row.agent_id == "agent-0" and row.method == "B5" and row.condition == "C5")]
        with self.assertRaises(SynthesisError):
            convergence_summary(rows)

    def test_saidas_separadas_nao_sao_sobrescritas(self):
        cold = cold_start_summary(sample_records())
        convergence, _ = convergence_summary(sample_records())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "synthesis"
            paths = write_outputs({"matrix_id": "matrix"}, cold, convergence, output_dir=output)
            self.assertEqual(len(paths), 3)
            self.assertTrue(all(path.is_file() for path in paths))
            with self.assertRaises(FileExistsError):
                write_outputs({"matrix_id": "matrix"}, cold, convergence, output_dir=output)


if __name__ == "__main__":
    unittest.main()
