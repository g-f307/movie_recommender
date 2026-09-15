import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cinebot_ml.experiment_config import build_execution_manifest, load_config
from cinebot_ml.ranking.candidates import CandidateSet, EligibilityPolicy
from cinebot_ml.ranking.contracts import ContractValidationError, RecommendationRequest, Recommender
from cinebot_ml.ranking.popularity import (
    DEFAULT_B0_CONFIG_PATH,
    PopularityConfigError,
    PopularityRecommender,
    bayesian_popularity_score,
    load_popularity_config,
)


def movie(movie_id, nota=7.0, votos=100):
    return {
        "id": movie_id,
        "titulo": f"Filme {movie_id}",
        "nota": nota,
        "votos": votos,
        "ano": "2020",
        "perfil": "drama",
    }


def candidate_set(movies):
    return CandidateSet(
        candidate_set_id="candidate-set-test",
        version="1.0",
        catalog_path="data/filmes.json",
        catalog_sha256="a" * 64,
        policy=EligibilityPolicy(),
        filters={},
        movies=tuple(movies),
        total_catalog_items=len(movies),
    )


def request(candidates, **changes):
    values = {
        "experiment_id": "experiment-v1.0__b0__c0__p0__s42",
        "protocol_version": "protocol-v1.0",
        "method": "B0",
        "method_version": "1.0",
        "condition": "C0",
        "profile": "P0",
        "seed": 42,
        "logical_timestamp": "2026-09-14T20:00:00Z",
        "candidate_movie_ids": tuple(item["id"] for item in candidates.movies),
        "k": 5,
    }
    values.update(changes)
    return RecommendationRequest(**values)


class PopularityBaselineTests(unittest.TestCase):
    def test_formula_bayesiana_com_exemplo_manual(self):
        config = load_popularity_config()
        score, components = bayesian_popularity_score(
            movie(1, nota=10.0, votos=100),
            catalog_mean=0.7,
            prior_strength=100.0,
            config=config,
        )
        self.assertAlmostEqual(score, 0.85)
        self.assertEqual(components["formula"], "bayesian_weighted_rating")

    def test_filmes_com_notas_iguais_desempatam_por_movie_id(self):
        candidates = candidate_set([movie(3, 8.0, 100), movie(1, 8.0, 100)])
        result = PopularityRecommender(candidates).recommend(request(candidates))
        self.assertEqual([item.movie_id for item in result.ranked_items], [1, 3])

    def test_filmes_com_votos_iguais_ordenam_por_nota(self):
        candidates = candidate_set([movie(1, 6.0, 100), movie(2, 9.0, 100)])
        result = PopularityRecommender(candidates).recommend(request(candidates))
        self.assertEqual([item.movie_id for item in result.ranked_items], [2, 1])

    def test_poucos_votos_sao_suavizados_para_media(self):
        candidates = candidate_set(
            [movie(1, 10.0, 1), movie(2, 8.0, 1000), movie(3, 5.0, 100)]
        )
        result = PopularityRecommender(candidates).recommend(request(candidates))
        scores = {item.movie_id: item.score for item in result.ranked_items}
        self.assertGreater(scores[2], scores[1])

    def test_nota_ausente_usa_media_e_votos_ausentes_usam_zero(self):
        candidates = candidate_set([movie(1, None, 100), movie(2, 8.0, None), movie(3, 6.0, 100)])
        result = PopularityRecommender(candidates).recommend(request(candidates))
        items = {item.movie_id: item for item in result.ranked_items}
        self.assertEqual(items[1].metadata["score_components"]["rating_source"], "catalog_mean")
        self.assertEqual(items[2].metadata["score_components"]["votes"], 0)
        self.assertAlmostEqual(items[2].score, items[1].metadata["score_components"]["catalog_mean"])

    def test_catalogo_com_um_filme_e_k_maior_que_candidatos(self):
        candidates = candidate_set([movie(7, 8.0, 20)])
        result = PopularityRecommender(candidates).recommend(request(candidates, k=10))
        self.assertEqual(len(result.ranked_items), 1)
        self.assertEqual(result.ranked_items[0].movie_id, 7)
        self.assertAlmostEqual(result.ranked_items[0].score, 0.8)

    def test_conjunto_vazio_produz_ranking_vazio_valido(self):
        candidates = candidate_set([])
        result = PopularityRecommender(candidates).recommend(request(candidates))
        self.assertEqual(result.ranked_items, ())

    def test_perfil_e_feedback_nao_alteram_ranking_do_mesmo_conjunto(self):
        candidates = candidate_set([movie(1, 8.0, 100), movie(2, 7.0, 200)])
        recommender = PopularityRecommender(candidates)
        first = recommender.recommend(request(candidates, profile="P0", condition="C0"))
        second = recommender.recommend(
            request(
                candidates,
                profile="P5",
                condition="C5",
                profile_data={"ranked_genres": ["terror"]},
                history=({"timestamp": "2026-09-14T19:00:00Z", "movie_id": 99},),
            )
        )
        self.assertEqual(
            [(item.movie_id, item.score) for item in first.ranked_items],
            [(item.movie_id, item.score) for item in second.ranked_items],
        )

    def test_mesmo_conjunto_produz_ranking_identico(self):
        candidates = candidate_set([movie(1, 8.0, 100), movie(2, 7.0, 200)])
        recommender = PopularityRecommender(candidates)
        first = recommender.recommend(request(candidates))
        second = recommender.recommend(request(candidates))
        self.assertEqual(first.ranked_items, second.ranked_items)

    def test_implementa_protocolo_e_valida_identidade(self):
        candidates = candidate_set([movie(1)])
        recommender = PopularityRecommender(candidates)
        self.assertIsInstance(recommender, Recommender)
        with self.assertRaisesRegex(ContractValidationError, "method=B0"):
            recommender.recommend(request(candidates, method="B1"))
        with self.assertRaisesRegex(ContractValidationError, "CandidateSet"):
            recommender.recommend(request(candidates, candidate_movie_ids=(99,)))

    def test_configuracao_congelada_alterada_e_bloqueada(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b0_popularity_v1.yaml"
            path.write_bytes(DEFAULT_B0_CONFIG_PATH.read_bytes())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".lock.json").write_text(
                json.dumps({"sha256": digest}), encoding="utf-8"
            )
            load_popularity_config(path)
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(PopularityConfigError, "congelada"):
                load_popularity_config(path)

    def test_manifesto_registra_configuracao_e_parametros_derivados(self):
        candidates = candidate_set([movie(1), movie(2, 8.0, 200)])
        recommender = PopularityRecommender(candidates)
        method_manifest = recommender.method_manifest()
        self.assertEqual(method_manifest["method"], "B0")
        self.assertEqual(len(method_manifest["config_sha256"]), 64)
        self.assertIn("catalog_mean", method_manifest["derived_parameters"])
        config = load_config()
        execution = build_execution_manifest(
            config,
            method="B0",
            condition="C0",
            profile="P0",
            seed=42,
            run=1,
            method_manifest=method_manifest,
        )
        self.assertEqual(execution["method_manifest"], method_manifest)


if __name__ == "__main__":
    unittest.main()
