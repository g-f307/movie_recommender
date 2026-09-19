import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cinebot_ml.experiments.freeze import audit_execution, pilot_matrix, preflight
from cinebot_ml.experiments.matrix import ExperimentCell, ExperimentMatrix, execute_matrix


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
        self.assertEqual(report["status"], "blocked" if report["blockers"] else "ready_for_pilot")
        self.assertEqual(report["review_status"], "advisor_unavailable_not_approved")
        self.assertEqual(report["method_scope"]["b6"], "deferred_no_distinct_method")
        self.assertGreaterEqual(report["missing_paired_unit_files"], 0)
        self.assertEqual(len(report["commit"]), 40)

    def test_interrupcao_retomada_sem_sobrescrita(self):
        cells = (ExperimentCell("B0", "C0", "P0", "consistent", 42, 5),
                 ExperimentCell("B1", "C0", "P0", "consistent", 42, 5))
        matrix = ExperimentMatrix("1", "protocol-v1.0", "a", "b", "c", cells, ())
        calls = []

        def interrupted(cell):
            calls.append(cell.method)
            if cell.method == "B1":
                raise KeyboardInterrupt()
            return {"individual": {"method": cell.method, "ranking_status": "completed",
                    "evaluation_status": "evaluated", "metrics": {"ndcg_at_k": 0.5}}}

        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                execute_matrix(matrix, root, interrupted)
            first = root / matrix.matrix_id / "cells" / f"{cells[0].cell_id}.json"
            before = first.read_bytes()
            report = execute_matrix(matrix, root, lambda cell: {
                "individual": {"method": cell.method, "ranking_status": "completed",
                "evaluation_status": "evaluated", "metrics": {"ndcg_at_k": 0.5}}})
            self.assertEqual((report.completed, report.skipped), (1, 1))
            self.assertEqual(first.read_bytes(), before)
            self.assertTrue(audit_execution(matrix, root)["complete"])
            first.write_text("{broken", encoding="utf-8")
            self.assertFalse(audit_execution(matrix, root)["complete"])


if __name__ == "__main__":
    unittest.main()
