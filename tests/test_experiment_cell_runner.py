import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cinebot_ml.experiments.cell_runner import run_cell
from cinebot_ml.experiments.matrix import MatrixValidationError, ExperimentCell


class MatrixCellRunnerTests(unittest.TestCase):
    def test_unidade_ausente_nao_gera_resultado_ficticio(self):
        cell = ExperimentCell("B0", "C0", "P0", "consistent", 42, 5)
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"CINEBOT_CELL_UNITS_DIR": directory}):
                with self.assertRaisesRegex(MatrixValidationError, "Unidade experimental ausente"):
                    run_cell(cell)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_unidade_divergente_e_rejeitada_antes_do_benchmark(self):
        cell = ExperimentCell("B0", "C0", "P0", "consistent", 42, 5)
        unit = SimpleNamespace(
            request=SimpleNamespace(condition="C2", profile="P0", seed=42, k=5),
            persona="consistent",
        )
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / f"{cell.comparison_id}.json").write_text("{}", encoding="utf-8")
            with patch.dict("os.environ", {"CINEBOT_CELL_UNITS_DIR": directory}), \
                 patch("cinebot_ml.experiments.cell_runner.load_benchmark_units", return_value=[unit]), \
                 patch("cinebot_ml.experiments.cell_runner.run_benchmark") as benchmark:
                with self.assertRaisesRegex(MatrixValidationError, "dimensões da célula divergem"):
                    run_cell(cell)
                benchmark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
