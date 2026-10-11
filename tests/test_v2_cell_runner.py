import tempfile
import unittest
import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from cinebot_ml.experiments.closed_loop import (
    DEFAULT_V2_1_CLOSED_LOOP_CONFIG_PATH,
    DEFAULT_V2_1_CLOSED_LOOP_SCHEMA_PATH,
    load_closed_loop_exposure_config,
)
from cinebot_ml.experiments.execution import (
    DEFAULT_V2_1_EXECUTION_SCHEMA_PATH,
    DEFAULT_V2_1_EXECUTION_SPEC_PATH,
    load_execution_spec,
)
from cinebot_ml.experiments.matrix import ExperimentCell, ExperimentMatrix, MatrixValidationError
from cinebot_ml.experiments.freeze import audit_execution
from cinebot_ml.experiments.policy_sensitivity import (
    DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH,
    DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH,
    load_policy_sensitivity_config,
)
from cinebot_ml.experiments.units import _closed_loop_unit, _shared_exposure_pair, write_units
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
    def test_v2_1_reuses_a_single_trajectory_for_b4_and_b5(self):
        spec = load_execution_spec(
            DEFAULT_V2_1_EXECUTION_SPEC_PATH,
            schema_path=DEFAULT_V2_1_EXECUTION_SCHEMA_PATH,
        )
        sensitivity = load_policy_sensitivity_config(
            DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH,
            schema_path=DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH,
        )
        scenario = sensitivity.scenarios[0]
        common = {
            "execution_id": spec.execution_id,
            "execution_config_sha256": spec.sha256,
            "partition": "validation",
            "policy_scenario_id": scenario.scenario_id,
            "policy_scenario_hash": scenario.scenario_hash,
        }
        b4 = ExperimentCell("B4", "C3", "P2", "consistent", 42, 5, **common)
        b5 = ExperimentCell("B5", "C3", "P2", "consistent", 42, 5, **common)
        payload = _shared_exposure_pair(
            (b4, b5),
            catalog(),
            "a" * 64,
            catalog_path="fixtures/v2_1_catalog.json",
            exposure_config=load_closed_loop_exposure_config(
                DEFAULT_V2_1_CLOSED_LOOP_CONFIG_PATH,
                schema_path=DEFAULT_V2_1_CLOSED_LOOP_SCHEMA_PATH,
            ),
            profile_observation_config=load_profile_observation_config(),
        )

        self.assertEqual(payload["pairing"]["exposure_id"], b4.exposure_id)
        self.assertEqual(payload["pairing"]["comparison_ids"], [b4.comparison_id])
        self.assertNotIn("method", payload["trajectory"])
        self.assertTrue(payload["trajectory_audit"]["complete"])
        self.assertEqual(
            payload["method_units"]["B4"]["request"]["candidate_movie_ids"],
            payload["method_units"]["B5"]["request"]["candidate_movie_ids"],
        )
        self.assertEqual(
            payload["method_units"]["B4"]["relevance"],
            payload["method_units"]["B5"]["relevance"],
        )
        self.assertEqual(payload["method_units"]["B4"]["trajectory_sha256"], payload["trajectory_sha256"])
        self.assertEqual(payload["method_units"]["B5"]["trajectory_sha256"], payload["trajectory_sha256"])

    def test_v2_1_reuses_one_trajectory_across_policy_scenarios(self):
        spec = load_execution_spec(
            DEFAULT_V2_1_EXECUTION_SPEC_PATH,
            schema_path=DEFAULT_V2_1_EXECUTION_SCHEMA_PATH,
        )
        sensitivity = load_policy_sensitivity_config(
            DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH,
            schema_path=DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH,
        )
        cells = []
        for scenario in sensitivity.scenarios[:2]:
            common = {
                "execution_id": spec.execution_id,
                "execution_config_sha256": spec.sha256,
                "partition": "validation",
                "policy_scenario_id": scenario.scenario_id,
                "policy_scenario_hash": scenario.scenario_hash,
            }
            cells.extend((
                ExperimentCell("B4", "C3", "P2", "consistent", 42, 5, **common),
                ExperimentCell("B5", "C3", "P2", "consistent", 42, 5, **common),
            ))

        payload = _shared_exposure_pair(
            cells,
            catalog(),
            "a" * 64,
            catalog_path="fixtures/v2_1_catalog.json",
            exposure_config=load_closed_loop_exposure_config(
                DEFAULT_V2_1_CLOSED_LOOP_CONFIG_PATH,
                schema_path=DEFAULT_V2_1_CLOSED_LOOP_SCHEMA_PATH,
            ),
            profile_observation_config=load_profile_observation_config(),
        )

        self.assertEqual(payload["pairing"]["exposure_id"], cells[0].exposure_id)
        self.assertEqual(
            payload["pairing"]["comparison_ids"],
            sorted({cell.comparison_id for cell in cells}),
        )
        self.assertEqual(payload["trajectory"]["exposure_id"], cells[0].exposure_id)

    def test_v2_1_shared_trajectory_respects_every_history_condition(self):
        spec = load_execution_spec(
            DEFAULT_V2_1_EXECUTION_SPEC_PATH,
            schema_path=DEFAULT_V2_1_EXECUTION_SCHEMA_PATH,
        )
        scenario = load_policy_sensitivity_config(
            DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH,
            schema_path=DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH,
        ).scenarios[0]
        exposure = load_closed_loop_exposure_config(
            DEFAULT_V2_1_CLOSED_LOOP_CONFIG_PATH,
            schema_path=DEFAULT_V2_1_CLOSED_LOOP_SCHEMA_PATH,
        )
        common = {
            "execution_id": spec.execution_id,
            "execution_config_sha256": spec.sha256,
            "partition": "validation",
            "policy_scenario_id": scenario.scenario_id,
            "policy_scenario_hash": scenario.scenario_hash,
        }
        observed = {}
        for condition in ("C0", "C1", "C2", "C3", "C4", "C5"):
            b4 = ExperimentCell("B4", condition, "P0", "consistent", 42, 5, **common)
            b5 = ExperimentCell("B5", condition, "P0", "consistent", 42, 5, **common)
            payload = _shared_exposure_pair(
                (b4, b5),
                catalog(),
                "a" * 64,
                catalog_path="fixtures/v2_1_catalog.json",
                exposure_config=exposure,
                profile_observation_config=load_profile_observation_config(),
            )
            observed[condition] = len(payload["trajectory"]["steps"])
            self.assertTrue(payload["trajectory_audit"]["complete"])

        self.assertEqual(observed, {"C0": 0, "C1": 1, "C2": 3, "C3": 5, "C4": 10, "C5": 15})

    def test_v2_1_runner_preserves_shared_evidence_across_policy_scenarios(self):
        spec = load_execution_spec(
            DEFAULT_V2_1_EXECUTION_SPEC_PATH,
            schema_path=DEFAULT_V2_1_EXECUTION_SCHEMA_PATH,
        )
        sensitivity = load_policy_sensitivity_config(
            DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH,
            schema_path=DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH,
        )
        cells = []
        for scenario in sensitivity.scenarios[:2]:
            common = {
                "execution_id": spec.execution_id,
                "execution_config_sha256": spec.sha256,
                "partition": "validation",
                "policy_scenario_id": scenario.scenario_id,
                "policy_scenario_hash": scenario.scenario_hash,
            }
            cells.extend((
                ExperimentCell("B4", "C2", "P2", "consistent", 42, 5, **common),
                ExperimentCell("B5", "C2", "P2", "consistent", 42, 5, **common),
            ))
        payload = _shared_exposure_pair(
            cells,
            catalog(),
            "a" * 64,
            catalog_path="fixtures/v2_1_catalog.json",
            exposure_config=load_closed_loop_exposure_config(
                DEFAULT_V2_1_CLOSED_LOOP_CONFIG_PATH,
                schema_path=DEFAULT_V2_1_CLOSED_LOOP_SCHEMA_PATH,
            ),
            profile_observation_config=load_profile_observation_config(),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / f"{cells[0].exposure_id}.json").write_text(json.dumps(payload), encoding="utf-8")
            source = root / "catalog.json"
            source.write_text(json.dumps({"perfis": {"drama": catalog()}}), encoding="utf-8")
            results = [
                run_v2_cell(cell, execution_spec=spec, units_root=root, catalog_path=source)
                for cell in cells
            ]

        self.assertEqual({result["candidate_set_id"] for result in results}, {results[0]["candidate_set_id"]})
        self.assertEqual({result["relevance_id"] for result in results}, {results[0]["relevance_id"]})
        self.assertEqual({result["trajectory_sha256"] for result in results}, {results[0]["trajectory_sha256"]})
        self.assertEqual({result["pairing"]["exposure_id"] for result in results}, {cells[0].exposure_id})

    def test_v2_1_materializes_one_file_per_exposure_not_per_scenario_or_method(self):
        base_spec = load_execution_spec(
            DEFAULT_V2_1_EXECUTION_SPEC_PATH,
            schema_path=DEFAULT_V2_1_EXECUTION_SCHEMA_PATH,
        )
        scenarios = load_policy_sensitivity_config(
            DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH,
            schema_path=DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH,
        ).scenarios[:2]
        cells = []
        for scenario in scenarios:
            common = {
                "execution_id": base_spec.execution_id,
                "execution_config_sha256": base_spec.sha256,
                "partition": "validation",
                "policy_scenario_id": scenario.scenario_id,
                "policy_scenario_hash": scenario.scenario_hash,
            }
            cells.extend((
                ExperimentCell("B4", "C1", "P2", "consistent", 42, 5, **common),
                ExperimentCell("B5", "C1", "P2", "consistent", 42, 5, **common),
            ))
        matrix = ExperimentMatrix(
            "2.1", base_spec.protocol_version, "a" * 64, "b" * 64, "c" * 64,
            tuple(cells), (), execution_id=base_spec.execution_id,
            execution_config_sha256=base_spec.sha256, partition="validation",
        )
        exposure = load_closed_loop_exposure_config(
            DEFAULT_V2_1_CLOSED_LOOP_CONFIG_PATH,
            schema_path=DEFAULT_V2_1_CLOSED_LOOP_SCHEMA_PATH,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = replace(
                base_spec,
                paths={**base_spec.paths, "units": "units", "catalog": "catalog.json"},
                profile_observation_config="profile.yaml",
                closed_loop_config="closed_loop.yaml",
                project_root=root,
            )
            with (
                patch("cinebot_ml.experiments.units.load_profile_observation_config", return_value=load_profile_observation_config()),
                patch("cinebot_ml.experiments.units.load_closed_loop_exposure_config", return_value=exposure),
            ):
                count = write_units(matrix, catalog(), "a" * 64, root / "units", execution_spec=spec)
            files = sorted((root / "units").glob("*.json"))

        self.assertEqual(count, 1)
        self.assertEqual([path.name for path in files], [f"{cells[0].exposure_id}.json"])

    def test_v2_1_materializes_distinct_holdout_units_for_distinct_frozen_selections(self):
        base_spec = load_execution_spec(
            DEFAULT_V2_1_EXECUTION_SPEC_PATH,
            schema_path=DEFAULT_V2_1_EXECUTION_SCHEMA_PATH,
        )
        scenario = load_policy_sensitivity_config(
            DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH,
            schema_path=DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH,
        ).scenarios[0]
        exposure = load_closed_loop_exposure_config(
            DEFAULT_V2_1_CLOSED_LOOP_CONFIG_PATH,
            schema_path=DEFAULT_V2_1_CLOSED_LOOP_SCHEMA_PATH,
        )

        def holdout(selection_id: str, selection_sha256: str) -> ExperimentMatrix:
            common = {
                "execution_id": base_spec.execution_id,
                "execution_config_sha256": base_spec.sha256,
                "partition": "holdout",
                "policy_scenario_id": scenario.scenario_id,
                "policy_scenario_hash": scenario.scenario_hash,
                "policy_selection_id": selection_id,
                "policy_selection_sha256": selection_sha256,
            }
            cells = (
                ExperimentCell("B4", "C0", "P0", "consistent", 42, 5, **common),
                ExperimentCell("B5", "C0", "P0", "consistent", 42, 5, **common),
            )
            return ExperimentMatrix(
                "2.1", base_spec.protocol_version, "a" * 64, "b" * 64, "c" * 64,
                cells, (), execution_id=base_spec.execution_id,
                execution_config_sha256=base_spec.sha256, partition="holdout",
                policy_selection_id=selection_id,
                policy_selection_sha256=selection_sha256,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = replace(
                base_spec,
                paths={**base_spec.paths, "units": "units", "catalog": "catalog.json"},
                profile_observation_config="profile.yaml",
                closed_loop_config="closed_loop.yaml",
                project_root=root,
            )
            with (
                patch("cinebot_ml.experiments.units.load_profile_observation_config", return_value=load_profile_observation_config()),
                patch("cinebot_ml.experiments.units.load_closed_loop_exposure_config", return_value=exposure),
            ):
                first = holdout("selection-a", "a" * 64)
                second = holdout("selection-b", "b" * 64)
                write_units(first, catalog(), "a" * 64, root / "units", execution_spec=spec)
                write_units(second, catalog(), "a" * 64, root / "units", execution_spec=spec)
            files = sorted((root / "units").glob("*.json"))
            source = root / "catalog.json"
            source.write_text(json.dumps({"perfis": {"drama": catalog()}}), encoding="utf-8")
            result = run_v2_cell(second.cells[0], execution_spec=spec, units_root=root / "units", catalog_path=source)

        self.assertEqual(len(files), 2)
        self.assertEqual(
            [path.name for path in files],
            sorted(f"{matrix.cells[0].materialization_id}.json" for matrix in (first, second)),
        )
        self.assertEqual(result["individual"]["method"], "B4")

    def test_v2_1_audit_rejeita_trajectory_hash_divergente_no_mesmo_par(self):
        base_spec = load_execution_spec(
            DEFAULT_V2_1_EXECUTION_SPEC_PATH,
            schema_path=DEFAULT_V2_1_EXECUTION_SCHEMA_PATH,
        )
        scenario = load_policy_sensitivity_config(
            DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH,
            schema_path=DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH,
        ).scenarios[0]
        common = {
            "execution_id": base_spec.execution_id,
            "execution_config_sha256": base_spec.sha256,
            "partition": "validation",
            "policy_scenario_id": scenario.scenario_id,
            "policy_scenario_hash": scenario.scenario_hash,
        }
        b4 = ExperimentCell("B4", "C0", "P0", "consistent", 42, 5, **common)
        b5 = ExperimentCell("B5", "C0", "P0", "consistent", 42, 5, **common)
        matrix = ExperimentMatrix(
            "2.1", base_spec.protocol_version, "a" * 64, "b" * 64, "c" * 64,
            (b4, b5), (), execution_id=base_spec.execution_id,
            execution_config_sha256=base_spec.sha256, partition="validation",
        )
        individual = {
            "ranking_status": "completed",
            "evaluation_status": "evaluated",
            "metrics": {"ndcg_at_k": 0.5},
            "candidate_set_id": "shared-candidates",
            "relevance_id": "shared-relevance",
            "agent_id": "agent-1",
            "persona": "consistent",
            "state_version": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            from cinebot_ml.experiments.matrix import execute_matrix

            execute_matrix(
                matrix,
                output,
                lambda cell: {
                    "individual": {**individual, "method": cell.method},
                    "trajectory_sha256": "a" * 64 if cell.method == "B4" else "b" * 64,
                },
            )
            audit = audit_execution(matrix, output)

        self.assertFalse(audit["complete"])
        self.assertEqual(audit["failures"][0]["reason"], "trajectory_mismatch")

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

    def test_materializes_deterministic_closed_loop_units(self):
        spec = load_execution_spec()
        cell = ExperimentCell(
            "B4", "C0", "P0", "consistent", 42, 5,
            execution_id=spec.execution_id,
            execution_config_sha256=spec.sha256,
            partition="validation",
            policy_scenario_id="S0",
            policy_scenario_hash="a" * 24,
        )
        kwargs = {
            "catalog_path": "fixtures/v2_catalog.json",
            "exposure_config": load_closed_loop_exposure_config(),
            "profile_observation_config": load_profile_observation_config(),
        }

        first = _closed_loop_unit(cell, catalog(), "a" * 64, **kwargs)
        second = _closed_loop_unit(cell, catalog(), "a" * 64, **kwargs)

        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=True),
            json.dumps(second, ensure_ascii=False, sort_keys=True),
        )


if __name__ == "__main__":
    unittest.main()
