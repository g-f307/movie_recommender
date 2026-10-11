import tempfile
import unittest
import csv
from dataclasses import replace
from pathlib import Path

from cinebot_ml.analysis.v2_analysis import analyze_v2_execution
from cinebot_ml.experiments.execution import load_execution_spec
from cinebot_ml.experiments.v2_config import (
    DEFAULT_V2_HOLDOUT_CONFIG_PATH,
    DEFAULT_V2_VALIDATION_CONFIG_PATH,
    load_v2_experiment_plan,
)
from cinebot_ml.experiments.v2_orchestration import (
    V2OrchestrationError,
    execute_v2_holdout,
    execute_v2_validation,
    finalize_v2_execution,
    freeze_v2_selection,
    prepare_v2_split,
)


def _result(cell):
    metric = 0.8 if cell.method == "B5" and cell.policy_scenario_id == "S1" else 0.4
    return {
        "individual": {
            "method": cell.method,
            "condition": cell.condition,
            "profile": cell.profile,
            "persona": cell.persona,
            "seed": cell.seed,
            "k": cell.k,
            "run": cell.run,
            "ranking_status": "completed",
            "evaluation_status": "evaluated",
            "metrics": {"ndcg_at_k": metric},
            "candidate_set_id": "candidates-shared",
            "relevance_id": "relevance-shared",
            "agent_id": "agent-shared",
            "state_version": 1,
            "partition": cell.partition,
            "policy_scenario_id": cell.policy_scenario_id,
            "policy_scenario_hash": cell.policy_scenario_hash,
            "profile_noise_level": "configured-v2",
            "exposure_policy": "closed-loop-v2",
            "trajectory_status": "completed",
        },
        "benchmark_manifest": {"git_commit": "a" * 40},
    }


class V2OrchestrationTests(unittest.TestCase):
    def test_exposes_the_v2_orchestration_through_the_experiments_package(self):
        from cinebot_ml.experiments import execute_v2_validation

        self.assertTrue(callable(execute_v2_validation))

    def _plans(self, root):
        execution = load_execution_spec()
        execution = replace(execution, project_root=root)
        for key in ("source_catalog", "split_manifest", "catalog"):
            path = execution.resolve_path(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}\n", encoding="utf-8")
        validation = replace(
            load_v2_experiment_plan(DEFAULT_V2_VALIDATION_CONFIG_PATH), execution=execution
        )
        holdout = replace(
            load_v2_experiment_plan(DEFAULT_V2_HOLDOUT_CONFIG_PATH), execution=execution
        )
        return validation, holdout

    def _filters(self):
        return {
            "conditions": ["C0"], "profiles": ["P0"], "personas": ["consistent"],
            "seeds": [42], "k_values": [5],
        }

    def test_rejects_holdout_before_a_frozen_validation_selection_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            _, holdout = self._plans(Path(directory))

            with self.assertRaisesRegex(V2OrchestrationError, "seleção congelada"):
                execute_v2_holdout(holdout, _result, **self._filters())

    def test_prepares_a_deterministic_v2_split_without_reusing_the_legacy_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation, _ = self._plans(root)
            dataset = root / "datasets" / "movie_preferences.csv"
            dataset.parent.mkdir(parents=True)
            with dataset.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["movie_id", "relevante"])
                writer.writeheader()
                for movie_id in range(12):
                    writer.writerow({"movie_id": movie_id, "relevante": movie_id % 2})
            validation.execution.resolve_path("split_manifest").unlink()

            manifest = prepare_v2_split(validation.execution, dataset_path=dataset, attempts=4)

            self.assertEqual(manifest["execution_id"], "execution-v2")
            self.assertTrue(validation.execution.resolve_path("split_manifest").is_file())
            self.assertNotIn("results/manifests/splits", str(validation.execution.resolve_path("split_manifest")))

    def test_runs_validation_freezes_selection_and_seals_a_partitioned_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation, holdout = self._plans(root)

            validation_matrix, _ = execute_v2_validation(validation, _result, **self._filters())
            selection_path = freeze_v2_selection(validation, validation_matrix)
            holdout_matrix, _ = execute_v2_holdout(
                holdout, _result, selection_path=selection_path, **self._filters()
            )
            manifest_path = finalize_v2_execution(
                validation, validation_matrix, holdout, holdout_matrix, selection_path
            )
            report, pairs, _ = analyze_v2_execution(
                manifest_path, selection_path=selection_path, bootstrap_resamples=100
            )

        self.assertEqual(report["selection"]["selected_scenario_id"], "S1")
        self.assertEqual({row["partition"] for row in pairs}, {"validation", "holdout"})


if __name__ == "__main__":
    unittest.main()
