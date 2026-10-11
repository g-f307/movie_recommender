import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cinebot_ml.experiments.matrix import ExperimentCell, ExperimentMatrix
from cinebot_ml.experiments.policy_sensitivity import (
    freeze_validation_selection,
    load_policy_sensitivity_config,
    select_validation_scenario,
)
from cinebot_ml.experiments.v2_config import (
    DEFAULT_V2_HOLDOUT_CONFIG_PATH,
    DEFAULT_V2_VALIDATION_CONFIG_PATH,
    V2ExperimentConfigError,
    build_v2_experiment_matrix,
    load_v2_experiment_plan,
)


class V2ExperimentConfigTests(unittest.TestCase):
    @staticmethod
    def _validation_selection():
        sensitivity = load_policy_sensitivity_config()
        observations = [
            {
                "scenario_id": scenario.scenario_id,
                "partition": "validation",
                "comparison_id": "agent-1",
                "agent_id": "agent-1",
                "seed": 42,
                "candidate_set_id": "candidates-1",
                "ndcg_at_5": 0.90 if scenario.scenario_id == "S1" else 0.40,
                "status": "evaluated",
            }
            for scenario in sensitivity.scenarios
        ]
        return select_validation_scenario(sensitivity, observations)

    def test_validation_matrix_records_partition_and_every_declared_scenario(self):
        plan = load_v2_experiment_plan(DEFAULT_V2_VALIDATION_CONFIG_PATH)

        matrix = build_v2_experiment_matrix(
            plan,
            conditions=["C0"],
            profiles=["P0"],
            personas=["consistent"],
            seeds=[42],
            k_values=[5],
        )

        self.assertEqual(matrix.total_cells, 14)
        self.assertEqual({cell.partition for cell in matrix.cells}, {"validation"})
        self.assertEqual(
            {cell.policy_scenario_id for cell in matrix.cells},
            {f"S{index}" for index in range(7)},
        )
        self.assertTrue(all(cell.policy_scenario_hash for cell in matrix.cells))

    def test_holdout_requires_compatible_frozen_validation_selection(self):
        plan = load_v2_experiment_plan(DEFAULT_V2_HOLDOUT_CONFIG_PATH)

        with self.assertRaisesRegex(V2ExperimentConfigError, "seleção congelada"):
            build_v2_experiment_matrix(plan)

        with tempfile.TemporaryDirectory() as directory:
            selection_path = Path(directory) / "selection.json"
            freeze_validation_selection(self._validation_selection(), selection_path)
            matrix = build_v2_experiment_matrix(
                plan,
                selection_path=selection_path,
                conditions=["C0"],
                profiles=["P0"],
                personas=["consistent"],
                seeds=[42],
                k_values=[5],
            )

        self.assertEqual(matrix.total_cells, 2)
        self.assertEqual({cell.partition for cell in matrix.cells}, {"holdout"})
        self.assertEqual({cell.policy_scenario_id for cell in matrix.cells}, {"S1"})

    def test_validation_and_holdout_have_distinct_auditable_identities(self):
        validation = load_v2_experiment_plan(DEFAULT_V2_VALIDATION_CONFIG_PATH)
        holdout = load_v2_experiment_plan(DEFAULT_V2_HOLDOUT_CONFIG_PATH)
        filters = {
            "conditions": ["C0"],
            "profiles": ["P0"],
            "personas": ["consistent"],
            "seeds": [42],
            "k_values": [5],
        }

        with tempfile.TemporaryDirectory() as directory:
            selection_path = Path(directory) / "selection.json"
            freeze_validation_selection(self._validation_selection(), selection_path)
            validation_matrix = build_v2_experiment_matrix(validation, **filters)
            holdout_matrix = build_v2_experiment_matrix(
                holdout, selection_path=selection_path, **filters
            )

        self.assertNotEqual(validation_matrix.matrix_id, holdout_matrix.matrix_id)
        self.assertNotEqual(
            validation_matrix.cells[0].comparison_id,
            holdout_matrix.cells[0].comparison_id,
        )

    def test_v1_cell_identity_remains_unchanged_when_v2_fields_are_absent(self):
        cell = ExperimentCell("B4", "C1", "P5", "consistent", 42, 5)

        self.assertEqual(
            cell._identity(),
            {
                "method": "B4",
                "condition": "C1",
                "profile": "P5",
                "persona": "consistent",
                "seed": 42,
                "k": 5,
                "run": 1,
            },
        )

        matrix = ExperimentMatrix(
            "1.0", "protocol-v1.0", "a" * 64, "b" * 64, "c" * 64, (cell,), ()
        )
        self.assertNotIn("partition", matrix.to_manifest())
        self.assertNotIn("policy_sensitivity_config_sha256", matrix.to_manifest())

    def test_divergent_lock_is_rejected_before_matrix_enumeration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "experiment_v2_validation.yaml"
            payload = yaml.safe_load(DEFAULT_V2_VALIDATION_CONFIG_PATH.read_text(encoding="utf-8"))
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            path.with_suffix(".lock.json").write_text(
                json.dumps({"sha256": hashlib.sha256(b"different").hexdigest()}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(V2ExperimentConfigError, "lock"):
                load_v2_experiment_plan(path)


if __name__ == "__main__":
    unittest.main()
