import unittest
from pathlib import Path

from cinebot_ml.analysis.discovery_cost import (
    _load_config,
    discovery_analysis,
    noninferiority,
    pareto_frontier,
)
from cinebot_ml.ranking.discovery_metrics import PopularityReference


CONFIG = {
    "post_feedback_conditions": ["C1", "C2", "C3", "C4", "C5"],
    "diversity_noninferiority": {
        "absolute_margin": -0.05,
        "rationale": "margem congelada",
    },
    "popularity_reference": {
        "partition": "catalog_metadata",
        "version": "catalog-v1",
        "source": "votos",
        "field": "votos",
    },
}
CATALOG = {
    1: {"id": 1, "generos": ["Drama"], "diretor": "A", "ano": 2000, "votos": 100},
    2: {"id": 2, "generos": ["Comedy"], "diretor": "B", "ano": 2010, "votos": 10},
    3: {"id": 3, "generos": ["Horror"], "diretor": "C", "ano": 2020, "votos": 1},
}


def cells():
    rows = []
    for agent_index in range(35):
        agent = f"agent-{agent_index}"
        for method_index in range(6):
            method = f"B{method_index}"
            for k in (5, 10):
                for condition_index, condition in enumerate(("C0", "C1")):
                    ranking = [1, 2, 3] if method != "B5" else [3, 2, 1]
                    rows.append({
                        "cell_id": f"{agent}-{method}-{k}-{condition}",
                        "comparison_id": f"{agent}-{k}-{condition}",
                        "method": method,
                        "condition": condition,
                        "profile": "P0",
                        "persona": "consistent",
                        "seed": 42,
                        "run": 1,
                        "k": k,
                        "agent_id": agent,
                        "candidate_set_id": f"{agent}-{condition}",
                        "ranked_movie_ids": ranking,
                        "metrics": {"ndcg_at_k": 0.5 + method_index * 0.01},
                    })
    return rows


class DiscoveryTests(unittest.TestCase):
    def test_referencia_de_catalogo_e_identificada(self):
        reference = PopularityReference(
            {1: 100, 2: 10, 3: 1}, "catalog_metadata", "catalog-v1"
        )
        self.assertEqual(reference.partition, "catalog_metadata")
        self.assertTrue(reference.distribution_id)

    def test_metricas_possuem_direcao_e_denominador_explicito(self):
        table, stability, metadata = discovery_analysis(cells(), CATALOG, CONFIG)
        self.assertEqual(len(table), 12)
        self.assertTrue(all(row["coverage_denominator"] == 3 for row in table))
        self.assertTrue(all(row["catalog_coverage_at_k"] == 1.0 for row in table))
        self.assertEqual(len(stability), 12)
        self.assertEqual(metadata["metric_directions"]["novelty_at_k"], "higher")

    def test_h5_exige_relevancia_e_nao_inferioridade(self):
        relevance = {
            "mean_difference": -0.01,
            "confidence_interval_95": [-0.02, -0.001],
        }
        result, rows = noninferiority(
            cells(), CATALOG, CONFIG, relevance_result=relevance
        )
        self.assertEqual(len(rows), 35)
        self.assertTrue(result["diversity_noninferior"])
        self.assertFalse(result["relevance_gain_observed"])
        self.assertEqual(result["decision"], "not_supported")


class CostAndProtocolTests(unittest.TestCase):
    def test_configuracao_oficial_esta_congelada(self):
        config = _load_config(
            Path("configs/experiments/discovery_cost_v1.json")
        )
        self.assertTrue(config["frozen"])
        self.assertEqual(
            config["diversity_noninferiority"]["absolute_margin"], -0.05
        )

    def test_pareto_nao_cria_score_composto(self):
        discovery = [
            {
                "method": "B0", "k": 5, "ndcg_at_k_mean": 0.4,
                "diversity_at_k_mean": 0.5, "novelty_at_k_mean": 0.5,
                "catalog_coverage_at_k": 0.5,
            },
            {
                "method": "B1", "k": 5, "ndcg_at_k_mean": 0.5,
                "diversity_at_k_mean": 0.6, "novelty_at_k_mean": 0.6,
                "catalog_coverage_at_k": 0.6,
            },
        ]
        efficiency = [
            {"method": "B0", "latency_ms_median": 2.0, "peak_memory_bytes_mean": 20},
            {"method": "B1", "latency_ms_median": 1.0, "peak_memory_bytes_mean": 10},
        ]
        frontier = pareto_frontier(discovery, efficiency)
        by_method = {row["method"]: row for row in frontier}
        self.assertFalse(by_method["B0"]["pareto_optimal"])
        self.assertEqual(by_method["B0"]["dominated_by"], "B1")
        self.assertNotIn("score", by_method["B1"])


if __name__ == "__main__":
    unittest.main()
