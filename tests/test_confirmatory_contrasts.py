import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from cinebot_ml.analysis.confirmatory_contrasts import (
    ContrastAnalysisError,
    _decision,
    _h3,
    _h4,
    write_outputs,
)
from cinebot_ml.analysis.loader import MetricRecord


def record(
    agent: str,
    method: str,
    condition: str,
    profile: str,
    value: float,
    *,
    k: int = 5,
    candidate: str = "candidates",
) -> MetricRecord:
    return MetricRecord(
        cell_id=f"{agent}-{method}-{condition}-{profile}-{k}",
        comparison_id=f"{agent}-{condition}-{profile}-{k}",
        method=method,
        condition=condition,
        profile=profile,
        persona="consistent",
        seed=42,
        run=1,
        k=k,
        agent_id=agent,
        status="evaluated",
        metric=value,
        candidate_set_id=candidate,
    )


def complete_records(agents: int = 3, k: int = 5) -> list[MetricRecord]:
    records = []
    for index in range(agents):
        agent = f"agent-{index}"
        for condition in ("C0", "C1", "C2", "C3", "C4", "C5"):
            for profile_index in range(6):
                profile = f"P{profile_index}"
                candidate = f"{agent}-{condition}-{k}"
                records.extend([
                    record(
                        agent, "B0", condition, profile,
                        0.20 + index * 0.01, k=k, candidate=candidate,
                    ),
                    record(
                        agent, "B5", condition, profile,
                        0.30 + profile_index * 0.01 + index * 0.01,
                        k=k, candidate=candidate,
                    ),
                ])
    return records


class PairingTests(unittest.TestCase):
    def test_h3_pareia_perfis_e_agrega_condicoes_por_agente(self):
        result, rows = _h3(
            complete_records(), 5, bootstrap_resamples=100, bootstrap_seed=42
        )
        self.assertEqual(result["accounting"]["complete_observation_pairs"], 15)
        self.assertEqual(result["accounting"]["agent_units"], 3)
        self.assertEqual(rows[0]["observations"], 5)
        self.assertAlmostEqual(rows[0]["difference"], 0.05)
        self.assertEqual(len(result["profile_curve"]), 6)
        self.assertEqual(result["monotonicity"]["nondecreasing_fraction"], 1.0)

    def test_h4_separa_c0_de_pos_feedback(self):
        records = complete_records()
        c0, c0_rows = _h4(
            records, 5, "no_feedback", bootstrap_resamples=100, bootstrap_seed=42
        )
        post, post_rows = _h4(
            records, 5, "post_feedback", bootstrap_resamples=100, bootstrap_seed=42
        )
        self.assertEqual(c0["conditions"], ["C0"])
        self.assertEqual(post["conditions"], ["C1", "C2", "C3", "C4", "C5"])
        self.assertEqual(c0_rows[0]["observations"], 6)
        self.assertEqual(post_rows[0]["observations"], 30)
        self.assertGreater(post["summary"]["difference"]["mean_difference"], 0.0)

    def test_rejeita_conjuntos_candidatos_divergentes(self):
        records = complete_records(agents=1)
        records = [
            replace(item, candidate_set_id="different")
            if (
                item.method == "B5"
                and item.condition == "C1"
                and item.profile == "P5"
            )
            else item for item in records
        ]
        with self.assertRaisesRegex(ContrastAnalysisError, "candidatos divergentes"):
            _h3(records, 5, bootstrap_resamples=50, bootstrap_seed=42)

    def test_par_incompleto_e_contabilizado_sem_imputacao(self):
        records = complete_records(agents=1)
        records = [
            item for item in records
            if not (
                item.method == "B5"
                and item.condition == "C1"
                and item.profile == "P5"
            )
        ]
        result, rows = _h3(
            records, 5, bootstrap_resamples=50, bootstrap_seed=42
        )
        self.assertEqual(result["accounting"]["incomplete_pairs"], 1)
        self.assertEqual(rows[0]["observations"], 4)


class DecisionAndOutputTests(unittest.TestCase):
    def test_h4_permanece_condicional_sem_evidencia_de_descoberta(self):
        result = {
            "summary": {"difference": {
                "mean_difference": 0.1,
                "confidence_interval_95": [0.05, 0.15],
            }},
            "accounting": {"agent_units": 35},
        }
        decision = _decision(result, adjusted_p=0.01, discovery_required=True)
        self.assertEqual(decision["status"], "relevance_supported_discovery_pending")
        self.assertTrue(decision["statistical_criteria_met"])

    def test_saidas_sao_imutaveis_e_manifestadas(self):
        report = {
            "analysis": "test", "matrix_id": "matrix",
            "decisions": {},
            "results": {
                "h3_k5": {"ok": True},
                "h4_post_feedback_k5": {"ok": True},
            },
        }
        rows = {"p5_vs_p0_k5": [{
            "agent_id": "a", "p0_ndcg_at_k": 0.2,
            "p5_ndcg_at_k": 0.3, "difference": 0.1,
        }]}
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "contrasts"
            write_outputs(report, rows, output)
            self.assertTrue((output / "analysis.manifest.json").exists())
            self.assertTrue((output / "p5_vs_p0_k5.csv").exists())
            with self.assertRaises(FileExistsError):
                write_outputs(report, rows, output)


if __name__ == "__main__":
    unittest.main()
