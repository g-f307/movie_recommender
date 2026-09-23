import tempfile
import unittest
from pathlib import Path

from cinebot_ml.analysis.loader import MetricRecord
from cinebot_ml.analysis.subgroups import (
    _paired_details,
    _seed_stability,
    holm_adjust,
    subgroup_summaries,
    write_subgroup_outputs,
)


def metric_record(
    agent_id,
    persona,
    seed,
    comparison,
    method,
    value,
    *,
    condition,
    profile,
    k,
):
    return MetricRecord(
        cell_id=f"{comparison}-{method}",
        comparison_id=comparison,
        method=method,
        condition=condition,
        profile=profile,
        persona=persona,
        seed=seed,
        run=1,
        k=k,
        agent_id=agent_id,
        status="evaluated",
        metric=value,
    )


def sample_records():
    records = []
    agents = (
        ("agent-a", "consistent", 42, 0.03),
        ("agent-b", "consistent", 137, 0.01),
        ("agent-c", "noisy", 42, -0.02),
        ("agent-d", "noisy", 137, -0.04),
    )
    for agent_id, persona, seed, effect in agents:
        for k in (5, 10):
            for condition in ("C0", "C1"):
                for profile in ("P0", "P1"):
                    comparison = f"{agent_id}-{k}-{condition}-{profile}"
                    difference = 0.0 if condition == "C0" else effect
                    records.extend([
                        metric_record(
                            agent_id, persona, seed, comparison, "B4", 0.5,
                            condition=condition, profile=profile, k=k,
                        ),
                        metric_record(
                            agent_id, persona, seed, comparison, "B5", 0.5 + difference,
                            condition=condition, profile=profile, k=k,
                        ),
                    ])
    return records


class HolmTests(unittest.TestCase):
    def test_correcao_preserva_ordem_step_down_e_limite(self):
        adjusted = holm_adjust([0.01, 0.04, 0.02])
        self.assertEqual(adjusted, [0.03, 0.04, 0.04])
        self.assertEqual(holm_adjust([1.0, 1.0]), [1.0, 1.0])
        with self.assertRaises(ValueError):
            holm_adjust([-0.1])


class PairAndSummaryTests(unittest.TestCase):
    def test_detalhes_preservam_dimensoes_e_ids_das_celulas(self):
        details, accounting = _paired_details(sample_records())
        self.assertEqual(accounting["valid_comparisons"], 32)
        self.assertEqual(accounting["invalid_metric_or_status"], 0)
        self.assertEqual(accounting["incomplete_comparisons"], 0)
        expected = {
            "comparison_id", "b4_cell_id", "b5_cell_id", "agent_id", "persona",
            "seed", "condition", "profile", "run", "k", "b4_ndcg_at_k",
            "b5_ndcg_at_k", "difference_b5_minus_b4",
        }
        self.assertEqual(set(details[0]), expected)

    def test_todas_as_categorias_e_k_sao_reportados(self):
        details, _ = _paired_details(sample_records())
        summaries = subgroup_summaries(
            details, bootstrap_resamples=100, bootstrap_seed=42
        )
        self.assertEqual(len(summaries), 16)
        self.assertEqual({row["k"] for row in summaries}, {5, 10})
        stability = _seed_stability(summaries)
        self.assertEqual({row["k"] for row in stability}, {5, 10})
        self.assertTrue(all(not row["sign_consistent"] for row in stability))
        c0 = next(
            row for row in summaries
            if row["dimension"] == "condition" and row["category"] == "C0" and row["k"] == 5
        )
        self.assertEqual(c0["direction"], "neutral")
        self.assertEqual(c0["ties"], 4)
        persona = next(row for row in summaries if row["dimension"] == "persona")
        self.assertTrue(persona["small_subgroup_warning"])
        self.assertIn("p_value_holm", persona)

    def test_par_incompleto_e_contabilizado_sem_imputacao(self):
        records = sample_records()
        records.pop()
        details, accounting = _paired_details(records)
        self.assertEqual(len(details), 31)
        self.assertEqual(accounting["incomplete_comparisons"], 1)


class OutputTests(unittest.TestCase):
    def test_escreve_tabelas_relatorio_e_quatro_figuras(self):
        details, _ = _paired_details(sample_records())
        summaries = subgroup_summaries(details, bootstrap_resamples=50)
        report = {"matrix_id": "test", "summaries": summaries}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outputs = write_subgroup_outputs(
                report,
                summaries,
                details,
                json_output=root / "report.json",
                summary_csv=root / "summary.csv",
                pairs_csv=root / "pairs.csv",
                figures_dir=root / "figures",
            )
            self.assertEqual(len(outputs), 7)
            for path in outputs:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
            for path in (root / "figures").glob("*.png"):
                self.assertEqual(path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with self.assertRaises(FileExistsError):
                write_subgroup_outputs(
                    report,
                    summaries,
                    details,
                    json_output=root / "report.json",
                    summary_csv=root / "summary.csv",
                    pairs_csv=root / "pairs.csv",
                    figures_dir=root / "figures",
                )


if __name__ == "__main__":
    unittest.main()
