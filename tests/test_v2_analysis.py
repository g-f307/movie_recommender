import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cinebot_ml.analysis.v2_analysis import (
    V2AnalysisError,
    analyze_v2_execution,
    write_v2_outputs,
)
from cinebot_ml.experiments.policy_sensitivity import (
    freeze_validation_selection,
    load_policy_sensitivity_config,
    select_validation_scenario,
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cell(cell_id, comparison_id, method, value, *, agent, condition, scenario, partition):
    policy = load_policy_sensitivity_config()
    scenario_hash = next(item.scenario_hash for item in policy.scenarios if item.scenario_id == scenario)
    return {
        "status": "completed",
        "cell": {
            "cell_id": cell_id,
            "comparison_id": comparison_id,
            "method": method,
            "condition": condition,
            "profile": "P2",
            "persona": "consistent",
            "seed": 42,
            "k": 5,
            "run": 1,
            "execution_id": "execution-v2",
            "execution_config_sha256": "e" * 64,
        },
        "result": {
            "individual": {
                "method": method,
                "condition": condition,
                "profile": "P2",
                "persona": "consistent",
                "seed": 42,
                "k": 5,
                "run": 1,
                "agent_id": agent,
                "candidate_set_id": f"candidates-{agent}-{condition}-{scenario}",
                "evaluation_status": "evaluated",
                "metrics": {"ndcg_at_k": value},
                "partition": partition,
                "policy_scenario_id": scenario,
                "policy_scenario_hash": scenario_hash,
                "profile_noise_level": "configured-v2",
                "exposure_policy": "closed-loop-v2",
                "trajectory_status": "completed",
            }
        },
    }


def write_v2_execution(root: Path, *, include_holdout=True, mixed_execution=False) -> Path:
    matrix_id = "matrix-v2-test"
    execution_root = root / matrix_id
    cells_root = execution_root / "cells"
    cells_root.mkdir(parents=True)
    cells = []
    for partition, scenarios in (("validation", ("S0", "S1")), ("holdout", ("S1",))):
        if partition == "holdout" and not include_holdout:
            continue
        for agent_index, agent in enumerate(("agent-a", "agent-b")):
            for scenario in scenarios:
                comparison_id = f"{partition}-{agent}-{scenario}"
                for method, baseline in (("B4", 0.40), ("B5", 0.45 if scenario == "S1" else 0.35)):
                    cell_id = f"{comparison_id}-{method.lower()}"
                    payload = _cell(
                        cell_id,
                        comparison_id,
                        method,
                        baseline + agent_index * 0.01,
                        agent=agent,
                        condition="C1",
                        scenario=scenario,
                        partition=partition,
                    )
                    if mixed_execution and cell_id.endswith("b5"):
                        payload["cell"]["execution_id"] = "execution-v1.1"
                    path = cells_root / f"{cell_id}.json"
                    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
                    cells.append(payload["cell"])

    matrix = {
        "matrix_id": matrix_id,
        "total_cells": len(cells),
        "cells": cells,
        "execution_id": "execution-v2",
        "execution_config_sha256": "e" * 64,
    }
    matrix_path = execution_root / "matrix.manifest.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "status": "complete",
        "execution": {
            "id": "execution-v2",
            "version": "2.2.0",
            "protocol_version": "protocol-v2.0",
            "config_sha256": "e" * 64,
            "exposure_policy": "closed-loop-v2",
        },
        "matrix": {
            "matrix_id": matrix_id,
            "total_cells": len(cells),
            "manifest_sha256": _digest(matrix_path),
        },
        "cells_sha256": {
            path.stem: _digest(path) for path in sorted(cells_root.glob("*.json"))
        },
    }
    manifest_path = execution_root / "execution.manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    return manifest_path


class V2AnalysisTests(unittest.TestCase):
    def test_le_selecao_congelada_antes_de_analisar_o_holdout(self):
        policy = load_policy_sensitivity_config()
        observations = [
            {
                "scenario_id": scenario.scenario_id,
                "partition": "validation",
                "comparison_id": "selection-agent",
                "agent_id": "selection-agent",
                "seed": 42,
                "candidate_set_id": "selection-candidates",
                "ndcg_at_5": 0.8 if scenario.scenario_id == "S1" else 0.4,
                "status": "evaluated",
            }
            for scenario in policy.scenarios
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection_path = root / "selection.json"
            freeze_validation_selection(
                select_validation_scenario(policy, observations), selection_path
            )

            report, _, _ = analyze_v2_execution(
                write_v2_execution(root / "source"),
                selection_path=selection_path,
                bootstrap_resamples=100,
            )

        self.assertEqual(report["selection"]["selected_scenario_id"], "S1")
        self.assertTrue(report["selection"]["locked_validation_selection"])

    def test_analisa_validacao_e_holdout_sem_agrupar_particoes(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_v2_execution(Path(temporary))

            report, pairs, summaries = analyze_v2_execution(
                manifest,
                selected_scenario_id="S1",
                bootstrap_resamples=100,
            )

        self.assertEqual(report["execution"]["id"], "execution-v2")
        self.assertEqual(report["accounting"]["valid_pairs"], 6)
        self.assertEqual({row["partition"] for row in pairs}, {"validation", "holdout"})
        self.assertEqual(
            {row["scenario_id"] for row in pairs if row["partition"] == "holdout"},
            {"S1"},
        )
        self.assertTrue(all(row["analysis_status"] == "exploratory" for row in summaries))
        self.assertIn("v1.1 and v2 are reported separately", report["guardrails"])

    def test_rejeita_celula_de_execucao_incompativel(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_v2_execution(Path(temporary), mixed_execution=True)

            with self.assertRaisesRegex(V2AnalysisError, "execução"):
                analyze_v2_execution(manifest, selected_scenario_id="S1")

    def test_rejeita_holdout_com_cenario_nao_congelado(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_v2_execution(Path(temporary))

            with self.assertRaisesRegex(V2AnalysisError, "holdout"):
                analyze_v2_execution(manifest, selected_scenario_id="S0")

    def test_escreve_fontes_estruturadas_e_figuras_sem_sobrescrever(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report, pairs, summaries = analyze_v2_execution(
                write_v2_execution(root / "source"),
                selected_scenario_id="S1",
                bootstrap_resamples=100,
            )
            output = root / "outputs"

            outputs = write_v2_outputs(report, pairs, summaries, output)

            self.assertTrue((output / "v2_analysis.json").is_file())
            self.assertTrue((output / "v2_paired_rows.csv").is_file())
            self.assertTrue((output / "v2_subgroups.csv").is_file())
            self.assertTrue((output / "analysis.manifest.json").is_file())
            self.assertEqual(
                {path.suffix for path in outputs if path.parent.name == "figures"},
                {".pdf", ".png"},
            )
            with self.assertRaises(FileExistsError):
                write_v2_outputs(report, pairs, summaries, output)


if __name__ == "__main__":
    unittest.main()
