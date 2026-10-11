import tempfile
import unittest
import json
from pathlib import Path

from cinebot_ml.experiments.closed_loop import load_closed_loop_exposure_config
from cinebot_ml.experiments.execution import load_execution_spec
from cinebot_ml.experiments.matrix import ExperimentCell, MatrixValidationError
from cinebot_ml.experiments.units import _closed_loop_unit
from cinebot_ml.experiments.v2_cell_runner import run_v2_cell
from cinebot_ml.simulation import load_profile_observation_config


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
        for index in range(1, 20)
    ]


class V2CellRunnerTests(unittest.TestCase):
    def test_rejects_cell_without_v2_partition_before_reading_any_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(MatrixValidationError, "partição v2"):
                run_v2_cell(
                    ExperimentCell("B4", "C0", "P0", "consistent", 42, 5),
                    execution_spec=load_execution_spec(),
                    units_root=Path(directory),
                )

    def test_missing_v2_unit_does_not_fall_back_to_v1_paths(self):
        spec = load_execution_spec()
        cell = ExperimentCell(
            "B4", "C0", "P0", "consistent", 42, 5,
            execution_id=spec.execution_id,
            execution_config_sha256=spec.sha256,
            partition="validation",
            policy_scenario_id="S0",
            policy_scenario_hash="a" * 24,
        )
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(MatrixValidationError, "Unidade v2 ausente"):
                run_v2_cell(cell, execution_spec=spec, units_root=Path(directory))

    def test_evaluates_a_closed_loop_b4_cell_with_v2_provenance(self):
        spec = load_execution_spec()
        cell = ExperimentCell(
            "B4", "C1", "P1", "consistent", 42, 5,
            execution_id=spec.execution_id,
            execution_config_sha256=spec.sha256,
            partition="validation",
            policy_scenario_id="S0",
            policy_scenario_hash="a" * 24,
        )
        payload = _closed_loop_unit(
            cell, catalog(), "a" * 64, catalog_path="fixtures/v2_catalog.json",
            exposure_config=load_closed_loop_exposure_config(),
            profile_observation_config=load_profile_observation_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"{cell.cell_id}.json").write_text(json.dumps(payload), encoding="utf-8")
            source = root / "catalog.json"
            source.write_text(json.dumps({"perfis": {"drama": catalog()}}), encoding="utf-8")
            result = run_v2_cell(
                cell, execution_spec=spec, units_root=root, catalog_path=source
            )

        self.assertEqual(result["individual"]["method"], "B4")
        self.assertEqual(result["individual"]["run"], 1)
        self.assertEqual(result["individual"]["partition"], "validation")
        self.assertEqual(result["individual"]["policy_scenario_id"], "S0")
        self.assertEqual(result["partition"], "validation")
        self.assertEqual(result["exposure_policy"], "closed-loop-v2")
        self.assertTrue(result["trajectory_audit"]["complete"])


if __name__ == "__main__":
    unittest.main()
