import math
import unittest

from cinebot_ml.ranking.metrics import (
    RankingMetricError,
    calculate_top_k_metrics,
    catalog_coverage_at_k,
)


class RankingMetricsTests(unittest.TestCase):
    def test_ranking_binario_perfeito(self):
        metrics = calculate_top_k_metrics([1, 2, 3, 4], {1: 1, 2: 1, 3: 0, 4: 0}, 2)
        self.assertEqual(metrics["precision_at_k"], 1.0)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertEqual(metrics["f1_at_k"], 1.0)
        self.assertEqual(metrics["ndcg_at_k"], 1.0)
        self.assertEqual(metrics["map_at_k"], 1.0)
        self.assertEqual(metrics["mrr_at_k"], 1.0)
        self.assertEqual(metrics["hit_rate_at_k"], 1.0)

    def test_ranking_invertido_tem_resultado_manual(self):
        metrics = calculate_top_k_metrics([3, 2, 1], {1: 1, 2: 0, 3: 0}, 3)
        self.assertAlmostEqual(metrics["precision_at_k"], 1 / 3)
        self.assertEqual(metrics["recall_at_k"], 1.0)
        self.assertAlmostEqual(metrics["map_at_k"], 1 / 3)
        self.assertAlmostEqual(metrics["mrr_at_k"], 1 / 3)
        self.assertAlmostEqual(metrics["ndcg_at_k"], 1 / math.log2(4))

    def test_nenhum_item_relevante_preserva_metricas_indefinidas(self):
        metrics = calculate_top_k_metrics([1, 2], {1: 0, 2: 0}, 2)
        self.assertEqual(metrics["precision_at_k"], 0.0)
        self.assertIsNone(metrics["recall_at_k"])
        self.assertIsNone(metrics["f1_at_k"])
        self.assertIsNone(metrics["ndcg_at_k"])
        self.assertIsNone(metrics["map_at_k"])
        self.assertEqual(metrics["mrr_at_k"], 0.0)
        self.assertEqual(metrics["hit_rate_at_k"], 0.0)

    def test_todos_os_itens_relevantes(self):
        metrics = calculate_top_k_metrics([1, 2, 3], {1: 1, 2: 1, 3: 1}, 2)
        self.assertEqual(metrics["precision_at_k"], 1.0)
        self.assertAlmostEqual(metrics["recall_at_k"], 2 / 3)
        self.assertEqual(metrics["ndcg_at_k"], 1.0)

    def test_k_maior_que_ranking_usa_itens_disponiveis(self):
        metrics = calculate_top_k_metrics([1, 2], {1: 1, 2: 0}, 10)
        self.assertEqual(metrics["precision_at_k"], 0.5)
        self.assertEqual(metrics["recall_at_k"], 1.0)

    def test_relevancia_graduada_afeta_ndcg(self):
        perfect = calculate_top_k_metrics([1, 2], {1: 3, 2: 1}, 2)
        inverted = calculate_top_k_metrics([2, 1], {1: 3, 2: 1}, 2)
        self.assertEqual(perfect["ndcg_at_k"], 1.0)
        self.assertLess(inverted["ndcg_at_k"], perfect["ndcg_at_k"])

    def test_ranking_com_duplicidade_e_bloqueado(self):
        with self.assertRaisesRegex(RankingMetricError, "duplicado"):
            calculate_top_k_metrics([1, 1], {1: 1}, 2)

    def test_cobertura_do_catalogo_usa_uniao_das_unidades(self):
        coverage = catalog_coverage_at_k([[1, 2], [2, 3]], [1, 2, 3, 4], 2)
        self.assertEqual(coverage, 0.75)

    def test_contrato_prepara_diversidade_e_novidade(self):
        metrics = calculate_top_k_metrics([1], {1: 1}, 1)
        self.assertIn("diversity_at_k", metrics)
        self.assertIn("novelty_at_k", metrics)
        self.assertIsNone(metrics["diversity_at_k"])
        self.assertIsNone(metrics["novelty_at_k"])


if __name__ == "__main__":
    unittest.main()
