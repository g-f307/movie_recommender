import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cinebot_ml.experiment_config import build_execution_manifest, load_config
from tests.experiment_helpers import materialized_experiment_inputs
from cinebot_ml.ranking import (
    CandidateSet,
    ContentConfigError,
    ContentRecommender,
    EligibilityPolicy,
    RecommendationRequest,
    Recommender,
    load_content_config,
)
from cinebot_ml.ranking.content import DEFAULT_B1_CONFIG_PATH
from cinebot_ml.ranking.contracts import ContractValidationError


def movie(
    movie_id,
    *,
    genres=("drama",),
    year="2020",
    votes=100,
    rating=7.0,
    director="Diretora Teste",
    keywords=("amizade",),
):
    primary, *secondary = genres
    return {
        "id": movie_id,
        "titulo": f"Filme {movie_id}",
        "perfil": primary,
        "genero": primary,
        "generos_secundarios": list(secondary),
        "ano": year,
        "votos": votes,
        "nota": rating,
        "diretor": director,
        "palavras_chave": list(keywords) if keywords is not None else None,
    }


def candidate_set(movies):
    return CandidateSet(
        candidate_set_id="candidate-set-content-test",
        version="1.0",
        catalog_path="data/filmes.json",
        catalog_sha256="b" * 64,
        policy=EligibilityPolicy(),
        filters={},
        movies=tuple(movies),
        total_catalog_items=len(movies),
    )


def request(candidates, **changes):
    values = {
        "experiment_id": "experiment-v1.0__b1__c0__p0__s42",
        "protocol_version": "protocol-v1.0",
        "method": "B1",
        "method_version": "1.0",
        "condition": "C0",
        "profile": "P0",
        "seed": 42,
        "logical_timestamp": "2026-09-15T04:00:00Z",
        "candidate_movie_ids": tuple(item["id"] for item in candidates.movies),
        "k": 10,
    }
    values.update(changes)
    return RecommendationRequest(**values)


class ContentBaselineTests(unittest.TestCase):
    def test_p0_sem_preferencias_produz_fallback_deterministico(self):
        candidates = candidate_set([movie(3), movie(1), movie(2)])
        result = ContentRecommender(candidates).recommend(request(candidates))
        self.assertEqual([item.movie_id for item in result.ranked_items], [1, 2, 3])
        self.assertTrue(all(item.score == 0.0 for item in result.ranked_items))

    def test_genero_exato_recebe_maior_score(self):
        candidates = candidate_set([movie(1, genres=("drama",)), movie(2, genres=("terror",))])
        result = ContentRecommender(candidates).recommend(
            request(candidates, profile="P1", profile_data={"ranked_genres": ["terror"]})
        )
        self.assertEqual(result.ranked_items[0].movie_id, 2)

    def test_generos_ordenados_tem_pesos_decrescentes(self):
        candidates = candidate_set([movie(1, genres=("drama",)), movie(2, genres=("terror",))])
        result = ContentRecommender(candidates).recommend(
            request(
                candidates,
                profile="P2",
                profile_data={"ranked_genres": ["terror", "drama", "comedia"]},
            )
        )
        self.assertEqual([item.movie_id for item in result.ranked_items], [2, 1])
        self.assertGreater(result.ranked_items[0].score, result.ranked_items[1].score)

    def test_p1_ignora_atributos_nao_permitidos(self):
        candidates = candidate_set(
            [movie(1, genres=("drama",), year="1990"), movie(2, genres=("drama",), year="2020")]
        )
        first = ContentRecommender(candidates).recommend(
            request(candidates, profile="P1", profile_data={"ranked_genres": ["drama"]})
        )
        second = ContentRecommender(candidates).recommend(
            request(
                candidates,
                profile="P1",
                profile_data={
                    "ranked_genres": ["drama", "terror", "acao"],
                    "decade_preference": "moderno",
                    "popularity_preference": "popular",
                    "preferred_directors": ["Diretor Proibido"],
                    "preferred_keywords": ["segredo"],
                },
            )
        )
        self.assertEqual(
            [(item.movie_id, item.score) for item in first.ranked_items],
            [(item.movie_id, item.score) for item in second.ranked_items],
        )

    def test_p3_considera_decada_e_p4_considera_popularidade(self):
        candidates = candidate_set(
            [
                movie(1, genres=("drama",), year="1995", votes=1, rating=8.5),
                movie(2, genres=("drama",), year="2020", votes=1000, rating=8.0),
            ]
        )
        recommender = ContentRecommender(candidates)
        p3 = recommender.recommend(
            request(
                candidates,
                profile="P3",
                profile_data={"ranked_genres": ["drama"], "decade_preference": "moderno"},
            )
        )
        p4 = recommender.recommend(
            request(
                candidates,
                profile="P4",
                profile_data={
                    "ranked_genres": ["drama"],
                    "decade_preference": "antes_2000",
                    "popularity_preference": "joia_escondida",
                },
            )
        )
        self.assertEqual(p3.ranked_items[0].movie_id, 2)
        self.assertEqual(p4.ranked_items[0].movie_id, 1)

    def test_metadados_ausentes_nao_interrompem_ranking(self):
        candidates = candidate_set(
            [
                movie(
                    1,
                    year=None,
                    votes="inválido",
                    rating=None,
                    director=None,
                    keywords=None,
                ),
                movie(2, genres=("terror",)),
            ]
        )
        result = ContentRecommender(candidates).recommend(
            request(
                candidates,
                profile="P4",
                profile_data={
                    "ranked_genres": ["drama"],
                    "decade_preference": "moderno",
                    "popularity_preference": "popular",
                },
            )
        )
        self.assertEqual(len(result.ranked_items), 2)

    def test_p5_representa_diretor_e_palavra_chave_estaticos(self):
        candidates = candidate_set(
            [
                movie(1, director="Jane Doe", keywords=("space opera",)),
                movie(2, director="John Doe", keywords=("western",)),
            ]
        )
        result = ContentRecommender(candidates).recommend(
            request(
                candidates,
                profile="P5",
                profile_data={
                    "preferred_directors": ["Jane Doe"],
                    "preferred_keywords": ["space-opera"],
                },
            )
        )
        self.assertEqual(result.ranked_items[0].movie_id, 1)

    def test_feedback_e_historico_nao_afetam_b1(self):
        candidates = candidate_set([movie(1, genres=("drama",)), movie(2, genres=("terror",))])
        recommender = ContentRecommender(candidates)
        first = recommender.recommend(
            request(candidates, profile="P2", profile_data={"ranked_genres": ["drama"]})
        )
        second = recommender.recommend(
            request(
                candidates,
                profile="P2",
                profile_data={"ranked_genres": ["drama"], "feedback": "dislike"},
                history=({"timestamp": "2026-09-15T03:00:00Z", "movie_id": 1, "feedback": "dislike"},),
            )
        )
        self.assertEqual(first.ranked_items, second.ranked_items)

    def test_mudanca_de_perfil_altera_ranking(self):
        candidates = candidate_set([movie(1, genres=("drama",)), movie(2, genres=("terror",))])
        recommender = ContentRecommender(candidates)
        drama = recommender.recommend(
            request(candidates, profile="P1", profile_data={"ranked_genres": ["drama"]})
        )
        terror = recommender.recommend(
            request(candidates, profile="P1", profile_data={"ranked_genres": ["terror"]})
        )
        self.assertNotEqual(drama.ranked_items[0].movie_id, terror.ranked_items[0].movie_id)

    def test_mesma_entrada_produz_ranking_identico(self):
        candidates = candidate_set([movie(2), movie(1)])
        recommender = ContentRecommender(candidates)
        payload = request(candidates, profile="P1", profile_data={"ranked_genres": ["drama"]})
        self.assertEqual(recommender.recommend(payload).ranked_items, recommender.recommend(payload).ranked_items)

    def test_implementa_contrato_e_valida_identidade(self):
        candidates = candidate_set([movie(1)])
        recommender = ContentRecommender(candidates)
        self.assertIsInstance(recommender, Recommender)
        with self.assertRaisesRegex(ContractValidationError, "method=B1"):
            recommender.recommend(request(candidates, method="B0"))
        with self.assertRaisesRegex(ContractValidationError, "CandidateSet"):
            recommender.recommend(request(candidates, candidate_movie_ids=(99,)))

    def test_configuracao_congelada_alterada_e_bloqueada(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b1_content_v1.yaml"
            path.write_bytes(DEFAULT_B1_CONFIG_PATH.read_bytes())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".lock.json").write_text(json.dumps({"sha256": digest}), encoding="utf-8")
            load_content_config(path)
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ContentConfigError, "congelada"):
                load_content_config(path)

    def test_manifesto_registra_configuracao_sem_fit_ou_holdout(self):
        candidates = candidate_set([movie(1)])
        recommender = ContentRecommender(candidates)
        method_manifest = recommender.method_manifest()
        self.assertEqual(method_manifest["transform_fit_partition"], "none")
        self.assertFalse(method_manifest["holdout_used_for_selection"])
        config = load_config()
        with materialized_experiment_inputs(config) as project_root:
            execution = build_execution_manifest(
                config, method="B1", condition="C0", profile="P1", seed=42,
                run=1, method_manifest=method_manifest, project_root=project_root,
            )
        self.assertEqual(execution["method_manifest"], method_manifest)


if __name__ == "__main__":
    unittest.main()
