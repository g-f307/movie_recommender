import unittest

from cinebot_ml.analysis.robustness_execution import (
    RobustnessExecutionError,
    validate_pairs,
)


class RobustnessExecutionTests(unittest.TestCase):
    @staticmethod
    def row(scenario, method, candidates="same"):
        return {
            "unit_id": "unit", "scenario": scenario, "method": method,
            "candidate_set_id": candidates, "relevance_id": "same",
            "candidate_count": 10,
        }

    def test_pares_completos_sao_aceitos(self):
        rows = [
            self.row(scenario, method)
            for scenario in ("nominal", "noise")
            for method in ("B4", "B5")
        ]
        validate_pairs(rows, ["nominal", "noise"])

    def test_metodo_ausente_e_rejeitado(self):
        with self.assertRaisesRegex(RobustnessExecutionError, "incompleto"):
            validate_pairs([self.row("nominal", "B4")], ["nominal"])

    def test_candidatos_diferentes_sao_rejeitados(self):
        rows = [self.row("nominal", "B4"),
                self.row("nominal", "B5", candidates="other")]
        with self.assertRaisesRegex(RobustnessExecutionError, "incompatíveis"):
            validate_pairs(rows, ["nominal"])

    def test_cenario_ausente_e_rejeitado(self):
        rows = [self.row("nominal", method) for method in ("B4", "B5")]
        with self.assertRaisesRegex(RobustnessExecutionError, "não possui cenários"):
            validate_pairs(rows, ["nominal", "noise"])


if __name__ == "__main__":
    unittest.main()
