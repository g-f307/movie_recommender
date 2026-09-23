import tempfile
import unittest
from pathlib import Path

from cinebot_ml.analysis.loader import MetricRecord
from cinebot_ml.analysis.reporting import general_method_table, inference_table


def record(method, agent, value):
    return MetricRecord(f"{method}-{agent}", agent, method, "C1", "P1", "persona",
                        42, 1, 5, agent, "evaluated", value)


class ReportingTableTests(unittest.TestCase):
    def test_tabela_geral_cobre_b0_b5_e_agrega_por_agente(self):
        records = []
        for method_index in range(6):
            for agent_index in range(3):
                records.extend([record(f"B{method_index}", f"a{agent_index}", .2),
                                record(f"B{method_index}", f"a{agent_index}", .4)])
        rows = general_method_table(records)
        self.assertEqual([row["method"] for row in rows], [f"B{i}" for i in range(6)])
        self.assertTrue(all(row["n_independent_agents"] == 3 for row in rows))
        self.assertTrue(all(abs(row["mean"] - .3) < 1e-12 for row in rows))

    def test_tabela_de_inferencia_preserva_ausencias(self):
        confirmatory = {
            "accounting": {"agent_units": 35},
            "paired_difference_b5_minus_b4": {
                "mean_difference": -.01, "confidence_interval_95": [-.03, -.002],
                "p_value_two_sided": .04, "rank_biserial_correlation": -.4,
            },
        }
        synthesis = {"exploratory": {"convergence": {"inference": {
            "n_longitudinal_units": 35, "statistic": 9.3, "p_value": .096,
        }}}}
        rows = inference_table(confirmatory, synthesis)
        self.assertEqual(len(rows), 2)
        self.assertIsNone(rows[1]["ci95_low"])
        self.assertIsNone(rows[1]["effect_size"])


if __name__ == "__main__":
    unittest.main()
