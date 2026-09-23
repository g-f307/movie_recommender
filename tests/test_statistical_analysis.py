import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cinebot_ml.analysis.inference import (
    bootstrap_mean_ci,
    paired_inference,
    paired_rank_biserial,
)
from cinebot_ml.analysis.loader import MetricRecord, OfficialResultsError, load_official_records
from cinebot_ml.analysis.statistical import _agent_pairs, write_outputs


def record(agent, comparison, method, value, condition="C1", k=5):
    persona, seed = agent
    return MetricRecord(
        cell_id=f"{comparison}-{method}",
        comparison_id=comparison,
        method=method,
        condition=condition,
        profile="P0",
        persona=persona,
        seed=seed,
        run=1,
        k=k,
        agent_id=f"synthetic-{persona}-{seed}-0000",
        status="evaluated",
        metric=value,
    )


class InferenceTests(unittest.TestCase):
    def test_bootstrap_e_reprodutivel_e_rank_biserial_preserva_sinal(self):
        differences = [0.1, 0.1, -0.1, 0.0]
        self.assertEqual(
            bootstrap_mean_ci(differences, resamples=500, seed=42),
            bootstrap_mean_ci(differences, resamples=500, seed=42),
        )
        self.assertAlmostEqual(paired_rank_biserial(differences), 1 / 3)
        result = paired_inference([0.2] * 4, [0.3, 0.3, 0.1, 0.2], resamples=500)
        self.assertEqual(result["valid_pairs"], 4)
        self.assertEqual((result["positive"], result["negative"], result["ties"]), (2, 1, 1))
        self.assertGreaterEqual(result["p_value_two_sided"], 0.0)
        self.assertLessEqual(result["p_value_two_sided"], 1.0)

    def test_empate_total_tem_p_um_e_efeito_zero(self):
        result = paired_inference([0.2, 0.3], [0.2, 0.3], resamples=50)
        self.assertEqual(result["p_value_two_sided"], 1.0)
        self.assertEqual(result["rank_biserial_correlation"], 0.0)


class PairingTests(unittest.TestCase):
    def test_agrega_condicoes_dentro_do_agente_antes_da_comparacao(self):
        records = []
        for index, condition in enumerate(("C1", "C2")):
            comparison = f"pair-{index}"
            records.extend([
                record(("consistent", 42), comparison, "B4", 0.4, condition),
                record(("consistent", 42), comparison, "B5", 0.5, condition),
            ])
        records.extend([
            record(("noisy", 137), "pair-2", "B4", 0.6),
            record(("noisy", 137), "pair-2", "B5", 0.5),
            record(("noisy", 137), "c0", "B4", 0.7, "C0"),
            record(("noisy", 137), "c0", "B5", 0.7, "C0"),
            record(("noisy", 137), "k10", "B4", 0.9, k=10),
            record(("noisy", 137), "k10", "B5", 0.1, k=10),
        ])
        rows, accounting = _agent_pairs(records)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["observations"], 2)
        self.assertAlmostEqual(rows[0]["difference_b5_minus_b4"], 0.1)
        self.assertEqual(accounting["c0_records_excluded"], 2)

    def test_status_invalido_e_par_incompleto_sao_contabilizados(self):
        good = record(("consistent", 42), "pair", "B4", 0.4)
        bad = MetricRecord(**{**record(("consistent", 42), "pair", "B5", 0.5).__dict__,
                              "status": "failed", "metric": None})
        with self.assertRaisesRegex(ValueError, "Nenhum par"):
            _agent_pairs([good, bad])


class LoaderAndOutputTests(unittest.TestCase):
    def _official_tree(self, root):
        matrix_id = "matrix-test"
        official = root / matrix_id
        cells = official / "cells"
        cells.mkdir(parents=True)
        hashes = {}
        for method, value in (("B4", 0.4), ("B5", 0.5)):
            cell_id = f"cell-{method.lower()}"
            payload = {
                "status": "completed",
                "cell": {
                    "cell_id": cell_id, "comparison_id": "pair", "method": method,
                    "condition": "C1", "profile": "P0", "persona": "consistent",
                    "seed": 42, "run": 1, "k": 5,
                },
                "result": {"individual": {
                    "agent_id": "synthetic-consistent-42-0000", "method": method,
                    "condition": "C1", "profile": "P0", "persona": "consistent",
                    "seed": 42, "k": 5, "evaluation_status": "evaluated",
                    "metrics": {"ndcg_at_k": value},
                }},
            }
            path = cells / f"{cell_id}.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            hashes[cell_id] = hashlib.sha256(path.read_bytes()).hexdigest()
        matrix_path = official / "matrix.manifest.json"
        matrix_path.write_text(json.dumps({"matrix_id": matrix_id, "total_cells": 2}), encoding="utf-8")
        manifest = {
            "status": "complete", "evidence": "exploratory_synthetic",
            "matrix_id": matrix_id, "total_cells": 2, "official": {"completed": 2},
            "dimensions": {"method": ["B4", "B5"], "k": [5]},
            "matrix_manifest_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
            "cells_sha256": hashes,
        }
        manifest_path = official / "experiment.manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path, cells / "cell-b4.json"

    def test_loader_valida_hash_e_rejeita_celula_alterada(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest, cell = self._official_tree(Path(temporary))
            _, records = load_official_records(manifest)
            self.assertEqual(len(records), 2)
            cell.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(OfficialResultsError, "Hash divergente"):
                load_official_records(manifest)

    def test_saidas_nao_sobrescrevem_e_preservam_csv_por_agente(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            json_path, csv_path = root / "report.json", root / "pairs.csv"
            rows = [{"agent_id": "a", "b4": 0.4, "b5": 0.5}]
            write_outputs({"ok": True}, rows, json_path, csv_path)
            self.assertIn("agent_id", csv_path.read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                write_outputs({"ok": True}, rows, json_path, csv_path)


if __name__ == "__main__":
    unittest.main()
