import unittest

from cinebot_ml.ranking.discovery_metrics import (
    PopularityReference,
    calculate_discovery_metrics,
    intra_list_diversity_at_k,
    novelty_at_k,
    popularity_bias_at_k,
    popularity_exposure_at_k,
    rank_position_variation_at_k,
    ranking_overlap_at_k,
    ranking_repetition_at_k,
)
from cinebot_ml.ranking.metrics import RankingMetricError


CATALOG = {
    1: {"id": 1, "generos": ["drama"], "ano": 1991},
    2: {"id": 2, "generos": ["drama"], "ano": 1991},
    3: {"id": 3, "generos": ["ficcao"], "ano": 2012},
    4: {"id": 4, "generos": ["comedia"], "ano": 1975},
    5: {"id": 5},
}


class DiscoveryMetricsTests(unittest.TestCase):
    def setUp(self):
        self.popularity = PopularityReference(
            {1: 1000, 2: 500, 3: 10, 4: 1}, "train", "popularity-v1"
        )

    def test_lista_homogenea_e_diversa_possuem_extremos_conhecidos(self):
        self.assertEqual(intra_list_diversity_at_k([1, 2], CATALOG, 2), 0.0)
        self.assertEqual(intra_list_diversity_at_k([1, 3, 4], CATALOG, 3), 1.0)

    def test_cauda_longa_e_mais_nova_e_menos_exposta_que_populares(self):
        popular_novelty = novelty_at_k([1, 2], self.popularity, 2)
        tail_novelty = novelty_at_k([3, 4], self.popularity, 2)
        self.assertLess(popular_novelty, tail_novelty)
        self.assertGreater(
            popularity_exposure_at_k([1, 2], self.popularity, 2),
            popularity_exposure_at_k([3, 4], self.popularity, 2),
        )
        self.assertGreater(popularity_bias_at_k([1, 2], self.popularity, 2), 0)
        self.assertLess(popularity_bias_at_k([3, 4], self.popularity, 2), 0)

    def test_referencia_registra_identidade_e_bloqueia_particao_proibida(self):
        first = self.popularity.distribution_id
        second = PopularityReference(dict(self.popularity.counts), "train", "popularity-v1")
        self.assertEqual(first, second.distribution_id)
        with self.assertRaisesRegex(RankingMetricError, "partição train"):
            PopularityReference({1: 1}, "test", "v1")

    def test_rankings_identicos_e_completamente_diferentes(self):
        self.assertEqual(ranking_overlap_at_k([1, 2], [1, 2], 5), 1.0)
        self.assertEqual(ranking_repetition_at_k([1, 2], [1, 2], 5), 1.0)
        self.assertEqual(rank_position_variation_at_k([1, 2], [1, 2], 5), 0.0)
        self.assertEqual(ranking_overlap_at_k([1, 2], [3, 4], 5), 0.0)
        self.assertEqual(ranking_repetition_at_k([1, 2], [3, 4], 5), 0.0)
        self.assertIsNone(rank_position_variation_at_k([1, 2], [3, 4], 5))

    def test_vazio_curto_e_metadado_ausente_sao_explicitos(self):
        self.assertIsNone(intra_list_diversity_at_k([], CATALOG, 5))
        self.assertIsNone(ranking_overlap_at_k([], [], 5))
        self.assertEqual(ranking_repetition_at_k([1], [1], 5), 1.0)
        self.assertIsNone(intra_list_diversity_at_k([1, 5], CATALOG, 5))
        self.assertIsNone(novelty_at_k([1, 5], self.popularity, 5))

    def test_contrato_retorna_intervalos_e_distribuicao(self):
        metrics = calculate_discovery_metrics([1, 3], CATALOG, 5, self.popularity)
        self.assertTrue(0 <= metrics["diversity_at_k"] <= 1)
        self.assertTrue(0 <= metrics["novelty_at_k"] <= 1)
        self.assertTrue(0 <= metrics["popularity_exposure_at_k"] <= 1)
        self.assertTrue(-1 <= metrics["popularity_bias_at_k"] <= 1)
        self.assertEqual(metrics["popularity_distribution_id"], self.popularity.distribution_id)


if __name__ == "__main__":
    unittest.main()
