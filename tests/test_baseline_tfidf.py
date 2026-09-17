import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cinebot_ml.experiment_config import build_execution_manifest, load_config
from tests.experiment_helpers import materialized_experiment_inputs
from cinebot_ml.ranking import (
    CandidateSet,
    EligibilityPolicy,
    RecommendationRequest,
    Recommender,
    TfidfArtifact,
    TfidfConfigError,
    TfidfRecommender,
    fit_tfidf_artifact,
    load_tfidf_config,
)
from cinebot_ml.ranking.contracts import ContractValidationError
from cinebot_ml.ranking.tfidf import DEFAULT_B2_CONFIG_PATH


def movie(movie_id, text="", genre="drama", keywords=()):
    return {
        "id": movie_id,
        "titulo": f"Filme {movie_id}",
        "sinopse": text,
        "perfil": genre,
        "genero": genre,
        "generos_secundarios": [],
        "palavras_chave": list(keywords) if keywords is not None else None,
        "ano": "2020",
    }


def candidate_set(movies):
    return CandidateSet(
        candidate_set_id="candidate-set-tfidf-test",
        version="1.0",
        catalog_path="data/filmes.json",
        catalog_sha256="c" * 64,
        policy=EligibilityPolicy(),
        filters={},
        movies=tuple(movies),
        total_catalog_items=len(movies),
    )


def request(candidates, **changes):
    values = {
        "experiment_id": "experiment-v1.0__b2__c0__p1__s42",
        "protocol_version": "protocol-v1.0",
        "method": "B2",
        "method_version": "1.0",
        "condition": "C0",
        "profile": "P1",
        "seed": 42,
        "logical_timestamp": "2026-09-15T05:00:00Z",
        "candidate_movie_ids": tuple(item["id"] for item in candidates.movies),
        "k": 10,
        "profile_data": {"ranked_genres": ["drama"]},
    }
    values.update(changes)
    return RecommendationRequest(**values)


def recommender(candidates, training=None):
    training = training or list(candidates.movies)
    artifact = fit_tfidf_artifact(training, partition="train")
    return TfidfRecommender(candidates, artifact)


class TfidfBaselineTests(unittest.TestCase):
    def test_perfil_e_filme_com_termo_identico(self):
        candidates = candidate_set(
            [movie(1, text="drama familiar", genre="drama"), movie(2, text="terror", genre="terror")]
        )
        result = recommender(candidates).recommend(request(candidates))
        self.assertEqual(result.ranked_items[0].movie_id, 1)
        self.assertGreater(result.ranked_items[0].score, 0.0)

    def test_textos_completamente_distintos_tem_similaridade_zero(self):
        candidates = candidate_set([movie(1, text="naves galáxia", genre="scifi")])
        result = recommender(candidates).recommend(
            request(candidates, profile_data={"ranked_genres": ["comedia"]})
        )
        self.assertEqual(result.ranked_items[0].score, 0.0)

    def test_documento_de_item_vazio_nao_produz_nan(self):
        candidates = candidate_set([movie(1, text=None, genre="", keywords=None)])
        training = [movie(99, text="drama conhecido")]
        result = recommender(candidates, training).recommend(request(candidates, profile="P0"))
        self.assertEqual(result.ranked_items[0].score, 0.0)

    def test_perfil_vazio_p0_produz_fallback_deterministico(self):
        candidates = candidate_set([movie(2, "drama"), movie(1, "terror")])
        result = recommender(candidates).recommend(
            request(candidates, profile="P0", profile_data={"ranked_genres": ["drama"]})
        )
        self.assertEqual([item.movie_id for item in result.ranked_items], [1, 2])
        self.assertTrue(all(item.score == 0.0 for item in result.ranked_items))

    def test_termo_desconhecido_e_ignorado(self):
        candidates = candidate_set([movie(1, text="drama", genre="drama")])
        model = recommender(candidates)
        result = model.recommend(
            request(candidates, profile_data={"ranked_genres": ["inexistente"]})
        )
        self.assertEqual(result.ranked_items[0].score, 0.0)

    def test_acentuacao_e_caixa_sao_normalizadas(self):
        candidates = candidate_set([movie(1, text="AÇÃO intensa", genre="acao")])
        result = recommender(candidates).recommend(
            request(candidates, profile_data={"ranked_genres": ["ação"]})
        )
        self.assertGreater(result.ranked_items[0].score, 0.0)

    def test_empate_usa_movie_id(self):
        candidates = candidate_set([movie(5, "drama"), movie(2, "drama")])
        result = recommender(candidates).recommend(request(candidates))
        self.assertEqual([item.movie_id for item in result.ranked_items], [2, 5])

    def test_generos_ordenados_alteram_score(self):
        candidates = candidate_set([movie(1, genre="drama"), movie(2, genre="terror")])
        model = recommender(candidates)
        first = model.recommend(
            request(candidates, profile="P2", profile_data={"ranked_genres": ["drama", "terror"]})
        )
        second = model.recommend(
            request(candidates, profile="P2", profile_data={"ranked_genres": ["terror", "drama"]})
        )
        self.assertNotEqual(first.ranked_items[0].movie_id, second.ranked_items[0].movie_id)

    def test_campos_nao_textuais_e_feedback_nao_afetam_b2(self):
        candidates = candidate_set([movie(1, genre="drama"), movie(2, genre="terror")])
        model = recommender(candidates)
        first = model.recommend(request(candidates))
        second = model.recommend(
            request(
                candidates,
                profile_data={
                    "ranked_genres": ["drama"],
                    "decade_preference": "moderno",
                    "profile_text": "terror",
                },
                history=({"timestamp": "2026-09-15T04:00:00Z", "movie_id": 2},),
            )
        )
        self.assertEqual(first.ranked_items, second.ranked_items)

    def test_tentativa_de_ajuste_com_holdout_e_bloqueada(self):
        for partition in ("validation", "test"):
            with self.subTest(partition=partition):
                with self.assertRaisesRegex(TfidfConfigError, "partition=train"):
                    fit_tfidf_artifact([movie(1, "drama")], partition=partition)

    def test_corpus_de_treino_vazio_e_bloqueado(self):
        with self.assertRaisesRegex(TfidfConfigError, "vazio"):
            fit_tfidf_artifact([movie(1, text=None, genre="", keywords=None)], partition="train")

    def test_artefato_persistido_e_reconstruivel(self):
        candidates = candidate_set([movie(1, "drama familiar"), movie(2, "terror")])
        artifact = fit_tfidf_artifact(list(candidates.movies), partition="train")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b2-artifact.json"
            artifact.save(path)
            restored = TfidfArtifact.load(path)
        first = TfidfRecommender(candidates, artifact).recommend(request(candidates))
        second = TfidfRecommender(candidates, restored).recommend(request(candidates))
        self.assertEqual(first.ranked_items, second.ranked_items)
        self.assertEqual(artifact.artifact_id, restored.artifact_id)

    def test_artefato_adulterado_e_bloqueado(self):
        artifact = fit_tfidf_artifact([movie(1, "drama")], partition="train")
        payload = artifact.to_dict()
        payload["idf"][0] += 1
        with self.assertRaisesRegex(TfidfConfigError, "Identidade"):
            TfidfArtifact.from_dict(payload)

    def test_configuracao_congelada_alterada_e_bloqueada(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b2_tfidf_v1.yaml"
            path.write_bytes(DEFAULT_B2_CONFIG_PATH.read_bytes())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".lock.json").write_text(json.dumps({"sha256": digest}), encoding="utf-8")
            load_tfidf_config(path)
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(TfidfConfigError, "congelada"):
                load_tfidf_config(path)

    def test_implementa_contrato_e_valida_identidade(self):
        candidates = candidate_set([movie(1, "drama")])
        model = recommender(candidates)
        self.assertIsInstance(model, Recommender)
        with self.assertRaisesRegex(ContractValidationError, "method=B2"):
            model.recommend(request(candidates, method="B1"))
        with self.assertRaisesRegex(ContractValidationError, "CandidateSet"):
            model.recommend(request(candidates, candidate_movie_ids=(99,)))

    def test_manifesto_registra_parametros_e_origem_do_artefato(self):
        candidates = candidate_set([movie(1, "drama")])
        model = recommender(candidates)
        manifest = model.method_manifest()
        self.assertEqual(manifest["fit_partition"], "train")
        self.assertFalse(manifest["holdout_used_for_fit"])
        self.assertGreater(manifest["vocabulary_size"], 0)
        config = load_config()
        with materialized_experiment_inputs(config) as project_root:
            execution = build_execution_manifest(
                config, method="B2", condition="C0", profile="P1", seed=42,
                run=1, method_manifest=manifest, project_root=project_root,
            )
        self.assertEqual(execution["method_manifest"], manifest)


if __name__ == "__main__":
    unittest.main()
