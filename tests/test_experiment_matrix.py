import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from cinebot_ml.experiments.matrix import (
    ExperimentMatrix,
    MatrixValidationError,
    execute_matrix,
    load_experiment_matrix,
)


class ExperimentMatrixTests(unittest.TestCase):
    def minimal(self, **overrides):
        values = {
            "methods": ["B0"], "conditions": ["C0"], "profiles": ["P0"],
            "personas": ["consistent"], "seeds": [42], "k_values": [5],
        }
        values.update(overrides)
        return load_experiment_matrix(**values)

    def matrix_config(self, exclusions):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "matrix.yaml"
        payload = {
            "version": "test-v1", "frozen": True,
            "deterministic_methods": ["B0", "B1", "B2", "B3", "B4", "B5"],
            "exclusions": exclusions,
        }
        path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        path.with_suffix(".lock.json").write_text(json.dumps({
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()
        }), encoding="utf-8")
        return directory, path

    def test_matriz_minima_e_completa_sao_enumeradas_sem_execucao(self):
        self.assertEqual(self.minimal().total_cells, 1)
        complete = load_experiment_matrix()
        self.assertEqual(complete.total_cells, 6 * 6 * 6 * 7 * 5 * 2)
        self.assertEqual(len({cell.cell_id for cell in complete.cells}), complete.total_cells)

    def test_combinacao_invalida_e_removida_com_motivo(self):
        directory, path = self.matrix_config([
            {"when": {"method": ["B0"], "profile": ["P0"]}, "reason": "controle inválido"}
        ])
        with directory:
            matrix = load_experiment_matrix(
                matrix_config_path=path, methods=["B0", "B1"], conditions=["C0"],
                profiles=["P0"], personas=["consistent"], seeds=[42], k_values=[5],
            )
        self.assertEqual(matrix.total_cells, 1)
        self.assertEqual(matrix.cells[0].method, "B1")
        self.assertEqual(matrix.excluded[0]["reason"], "controle inválido")

    def test_metodo_nao_habilitado_e_bloqueado(self):
        with self.assertRaisesRegex(MatrixValidationError, "Método não habilitado"):
            self.minimal(methods=["B6"])

    def test_interrupcao_e_retomada_nao_duplicam_celula_concluida(self):
        matrix = self.minimal(methods=["B0", "B1"])
        calls = []
        def interrupted(cell):
            calls.append(cell.cell_id)
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return {"ok": True}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.assertRaises(KeyboardInterrupt):
                execute_matrix(matrix, output, interrupted)
            resumed_calls = []
            report = execute_matrix(
                matrix, output,
                lambda cell: resumed_calls.append(cell.cell_id) or {"resumed": True},
            )
            self.assertEqual(report.skipped, 1)
            self.assertEqual(report.completed, 1)
            self.assertEqual(resumed_calls, [matrix.cells[1].cell_id])

    def test_resultado_existente_e_preservado(self):
        matrix = self.minimal()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            execute_matrix(matrix, output, lambda cell: {"value": 1})
            report = execute_matrix(matrix, output, lambda cell: {"value": 2})
            record = json.loads(next((output / matrix.matrix_id / "cells").iterdir()).read_text())
            self.assertEqual(report.skipped, 1)
            self.assertEqual(record["result"]["value"], 1)

    def test_falha_em_uma_celula_nao_interrompe_as_demais(self):
        matrix = self.minimal(methods=["B0", "B1"])
        def runner(cell):
            if cell.method == "B0":
                raise RuntimeError("falha controlada")
            return {"ok": True}
        with tempfile.TemporaryDirectory() as directory:
            report = execute_matrix(matrix, Path(directory), runner)
            self.assertEqual((report.failed, report.completed), (1, 1))
            records = [json.loads(path.read_text()) for path in (Path(directory) / matrix.matrix_id / "cells").iterdir()]
            self.assertEqual({record["status"] for record in records}, {"failed", "completed"})

    def test_ordem_diferente_mantem_ids(self):
        matrix = self.minimal(methods=["B0", "B1"])
        reversed_matrix = replace(matrix, cells=tuple(reversed(matrix.cells)))
        self.assertEqual(matrix.matrix_id, reversed_matrix.matrix_id)
        self.assertEqual(
            {cell.cell_id for cell in matrix.cells}, {cell.cell_id for cell in reversed_matrix.cells}
        )

    def test_multiplas_seeds_e_comparacao_pareada(self):
        matrix = self.minimal(methods=["B0", "B1"], seeds=[42, 137])
        self.assertEqual(matrix.total_cells, 4)
        groups = {}
        for cell in matrix.cells:
            groups.setdefault(cell.comparison_id, set()).add(cell.method)
        self.assertEqual(len(groups), 2)
        self.assertTrue(all(methods == {"B0", "B1"} for methods in groups.values()))

    def test_metodos_deterministicos_nao_recebem_repeticoes_artificiais(self):
        matrix = self.minimal(methods=["B0", "B5"])
        self.assertEqual(matrix.total_cells, 2)
        self.assertEqual({cell.run for cell in matrix.cells}, {1})


if __name__ == "__main__":
    unittest.main()
