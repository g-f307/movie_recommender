import unittest

from cinebot_ml.experiments.freeze import pilot_matrix, preflight


class FreezePreflightTests(unittest.TestCase):
    def test_pilot_dimensions_and_identity(self):
        matrix = pilot_matrix()
        self.assertEqual(matrix.total_cells, 432)
        self.assertEqual({cell.method for cell in matrix.cells}, {f"B{i}" for i in range(6)})
        self.assertEqual({cell.condition for cell in matrix.cells}, {"C0", "C2", "C4"})
        self.assertEqual({cell.profile for cell in matrix.cells}, {"P0", "P4", "P5"})
        self.assertEqual({cell.persona for cell in matrix.cells}, {"consistent", "noisy"})
        self.assertEqual({cell.seed for cell in matrix.cells}, {42, 137})
        self.assertEqual(matrix.matrix_id, pilot_matrix().matrix_id)

    def test_preflight_does_not_claim_freeze(self):
        report = preflight()
        self.assertEqual(report["purpose"], "preflight_only")
        self.assertEqual(report["status"], "blocked")
        self.assertIn("advisor_protocol_approval", report["blockers"])
        self.assertIn("b6_decision", report["blockers"])
        self.assertIn("official_cell_runner", report["blockers"])
        self.assertEqual(len(report["commit"]), 40)


if __name__ == "__main__":
    unittest.main()
