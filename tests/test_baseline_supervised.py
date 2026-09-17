import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from cinebot_ml.experiment_config import build_execution_manifest, load_config
from tests.experiment_helpers import materialized_experiment_inputs
from cinebot_ml.ranking import (
    CandidateSet,
    EligibilityPolicy,
    RecommendationRequest,
    Recommender,
    SupervisedArtifact,
    SupervisedConfigError,
    SupervisedRecommender,
    assert_supervised_fit_partition,
    assert_supervised_selection_partition,
    build_supervised_frame,
)
from cinebot_ml.ranking.contracts import ContractValidationError
from cinebot_ml.ranking.supervised import DEFAULT_B3_CONFIG_PATH, load_supervised_config
from cinebot_ml.schema import FEATURE_COLUMNS
from cinebot_ml.train import supervised_selection_key


class FeatureScoreModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, frame):
        scores = np.asarray(frame["nota"], dtype=float) / 10.0
        return np.column_stack((1.0 - scores, scores))


class InvalidScoreModel(FeatureScoreModel):
    def predict_proba(self, frame):
        return np.tile(np.asarray([[-0.1, 1.1]]), (len(frame), 1))


class UntrainedModel:
    pass


def movie(movie_id, *, rating=7.0, genre="drama", **changes):
    value = {
        "id": movie_id,
        "titulo": f"Filme {movie_id}",
        "sinopse": "história de teste",
        "perfil": genre,
        "genero": genre,
        "generos_secundarios": [],
        "palavras_chave": [],
        "ano": "2020",
        "nota": rating,
        "votos": 100,
        "duracao": 100,
        "streaming": [],
    }
    value.update(changes)
    return value


def candidate_set(movies):
    return CandidateSet(
        candidate_set_id="candidate-set-supervised-test",
        version="1.0",
        catalog_path="data/filmes.json",
        catalog_sha256="d" * 64,
        policy=EligibilityPolicy(),
        filters={},
        movies=tuple(movies),
        total_catalog_items=len(movies),
    )


def request(candidates, **changes):
    values = {
        "experiment_id": "experiment-v1.0__b3__c0__p4__s42",
        "protocol_version": "protocol-v1.0",
        "method": "B3",
        "method_version": "1.0",
        "condition": "C0",
        "profile": "P4",
        "seed": 42,
        "logical_timestamp": "2026-09-15T06:00:00Z",
        "candidate_movie_ids": tuple(item["id"] for item in candidates.movies),
        "k": 10,
        "profile_data": {
            "ranked_genres": ["drama", "terror", "comedia"],
            "decade_preference": "moderno",
            "popularity_preference": "popular",
        },
    }
    values.update(changes)
    return RecommendationRequest(**values)


def metadata(threshold=0.7, **changes):
    value = {
        "method_version": "1.0",
        "winner_model": "test_probability_model",
        "winner_params": {"constant": False},
        "winner_threshold": threshold,
        "feature_columns": list(FEATURE_COLUMNS),
        "positive_class": 1,
        "label_source": "heuristic_proxy_with_explicit_feedback_overrides",
        "fit_partition": "train",
        "selection_partitions": ["train", "validation"],
        "frozen_before_holdout": True,
    }
    value.update(changes)
    return value


def artifact(model=None, artifact_metadata=None):
    return SupervisedArtifact(
        model=model or FeatureScoreModel(),
        model_path=Path("artifacts/test.joblib"),
        model_sha256="a" * 64,
        metadata_path=Path("artifacts/test.json"),
        metadata_sha256="b" * 64,
        metadata=artifact_metadata or metadata(),
    )


class SupervisedBaselineTests(unittest.TestCase):
    def test_ordena_todos_os_candidatos_por_probabilidade(self):
        candidates = candidate_set([movie(1, rating=6), movie(2, rating=9), movie(3, rating=7)])
        result = SupervisedRecommender(candidates, artifact()).recommend(request(candidates))
        self.assertEqual([item.movie_id for item in result.ranked_items], [2, 3, 1])
        self.assertEqual([item.score for item in result.ranked_items], [0.9, 0.7, 0.6])

    def test_threshold_nao_altera_ordenacao(self):
        candidates = candidate_set([movie(1, rating=4), movie(2, rating=8)])
        low = SupervisedRecommender(candidates, artifact(artifact_metadata=metadata(0.2)))
        high = SupervisedRecommender(candidates, artifact(artifact_metadata=metadata(0.9)))
        first = low.recommend(request(candidates))
        second = high.recommend(request(candidates))
        self.assertEqual(
            [(item.movie_id, item.score) for item in first.ranked_items],
            [(item.movie_id, item.score) for item in second.ranked_items],
        )

    def test_modelo_nao_treinado_e_bloqueado(self):
        candidates = candidate_set([movie(1)])
        with self.assertRaisesRegex(SupervisedConfigError, "classe positiva"):
            SupervisedRecommender(candidates, artifact(model=UntrainedModel()))

    def test_features_ausentes_no_schema_sao_bloqueadas(self):
        candidates = candidate_set([movie(1)])
        bad = metadata(feature_columns=FEATURE_COLUMNS[:-1])
        with self.assertRaisesRegex(SupervisedConfigError, "Schema de features"):
            SupervisedRecommender(candidates, artifact(artifact_metadata=bad))

    def test_categoria_desconhecida_e_preservada_no_frame(self):
        candidates = candidate_set([movie(1, genre="genero_inedito")])
        frame = build_supervised_frame(candidates, request(candidates))
        self.assertEqual(frame.loc[0, "genero"], "generoinedito")
        result = SupervisedRecommender(candidates, artifact()).recommend(request(candidates))
        self.assertEqual(len(result.ranked_items), 1)

    def test_dados_incompletos_recebem_fallback_sem_remover_candidato(self):
        candidates = candidate_set(
            [movie(1, rating=None, votos="inválido", duracao=None, sinopse=None, diretor=None)]
        )
        result = SupervisedRecommender(candidates, artifact()).recommend(request(candidates))
        self.assertEqual(len(result.ranked_items), 1)
        self.assertEqual(result.ranked_items[0].score, 0.0)

    def test_conjunto_vazio_produz_ranking_vazio(self):
        candidates = candidate_set([])
        result = SupervisedRecommender(candidates, artifact()).recommend(request(candidates))
        self.assertEqual(result.ranked_items, ())

    def test_score_fora_do_intervalo_e_bloqueado(self):
        candidates = candidate_set([movie(1)])
        model = SupervisedRecommender(candidates, artifact(model=InvalidScoreModel()))
        with self.assertRaisesRegex(SupervisedConfigError, "intervalo"):
            model.recommend(request(candidates))

    def test_holdout_e_bloqueado_para_fit_e_selecao(self):
        for partition in ("validation", "test"):
            with self.subTest(partition=partition):
                with self.assertRaisesRegex(SupervisedConfigError, "partition=train"):
                    assert_supervised_fit_partition(partition)
        with self.assertRaisesRegex(SupervisedConfigError, "holdout"):
            assert_supervised_selection_partition("test")
        assert_supervised_selection_partition("validation")

    def test_selecao_da_familia_usa_somente_metricas_de_validacao(self):
        strong_validation = {
            "cv_metrics_mean": {
                "f1": 0.9, "precision": 0.8, "average_precision": 0.8, "roc_auc": 0.8,
            },
            "metrics": {"f1": 0.1},
        }
        strong_holdout = {
            "cv_metrics_mean": {
                "f1": 0.8, "precision": 0.9, "average_precision": 0.9, "roc_auc": 0.9,
            },
            "metrics": {"f1": 1.0},
        }
        selected = max([strong_validation, strong_holdout], key=supervised_selection_key)
        self.assertIs(selected, strong_validation)

    def test_perfis_nao_reconstroem_campos_proibidos(self):
        candidates = candidate_set([movie(1)])
        clean = build_supervised_frame(
            candidates,
            request(candidates, profile="P1", profile_data={"ranked_genres": ["drama"]}),
        )
        injected = build_supervised_frame(
            candidates,
            request(
                candidates,
                profile="P1",
                profile_data={
                    "ranked_genres": ["drama", "terror", "comedia"],
                    "decade_preference": "moderno",
                    "popularity_preference": "popular",
                },
            ),
        )
        self.assertTrue(clean[FEATURE_COLUMNS].equals(injected[FEATURE_COLUMNS]))

    def test_feedback_nao_altera_baseline_b3(self):
        candidates = candidate_set([movie(1, rating=6), movie(2, rating=8)])
        model = SupervisedRecommender(candidates, artifact())
        first = model.recommend(request(candidates))
        second = model.recommend(
            request(
                candidates,
                history=({"timestamp": "2026-09-15T05:00:00Z", "movie_id": 2, "feedback": "dislike"},),
            )
        )
        self.assertEqual(first.ranked_items, second.ranked_items)

    def test_ranking_e_deterministico(self):
        candidates = candidate_set([movie(2, rating=8), movie(1, rating=8)])
        model = SupervisedRecommender(candidates, artifact())
        first = model.recommend(request(candidates))
        second = model.recommend(request(candidates))
        self.assertEqual(first.ranked_items, second.ranked_items)
        self.assertEqual([item.movie_id for item in first.ranked_items], [1, 2])

    def test_metadado_legado_nao_conforme_e_rejeitado(self):
        candidates = candidate_set([movie(1)])
        with self.assertRaisesRegex(SupervisedConfigError, "Metadado obrigatório"):
            SupervisedRecommender(candidates, artifact(artifact_metadata={"winner_model": "legacy"}))

    def test_artefato_persistido_registra_hash_do_modelo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            metadata_path = root / "metadata.json"
            joblib.dump(FeatureScoreModel(), model_path)
            metadata_path.write_text(json.dumps(metadata()), encoding="utf-8")
            loaded = SupervisedArtifact.load(model_path, metadata_path)
        self.assertEqual(len(loaded.model_sha256), 64)
        self.assertEqual(len(loaded.metadata_sha256), 64)

    def test_configuracao_congelada_alterada_e_bloqueada(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b3_supervised_v1.yaml"
            path.write_bytes(DEFAULT_B3_CONFIG_PATH.read_bytes())
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".lock.json").write_text(
                json.dumps({"sha256": digest}), encoding="utf-8"
            )
            load_supervised_config(path)
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(SupervisedConfigError, "congelada"):
                load_supervised_config(path)

    def test_implementa_contrato_e_valida_identidade(self):
        candidates = candidate_set([movie(1)])
        model = SupervisedRecommender(candidates, artifact())
        self.assertIsInstance(model, Recommender)
        with self.assertRaisesRegex(ContractValidationError, "method=B3"):
            model.recommend(request(candidates, method="B2"))
        with self.assertRaisesRegex(ContractValidationError, "CandidateSet"):
            model.recommend(request(candidates, candidate_movie_ids=(99,)))

    def test_manifesto_registra_modelo_threshold_proxy_e_splits(self):
        candidates = candidate_set([movie(1)])
        model = SupervisedRecommender(candidates, artifact())
        manifest = model.method_manifest()
        self.assertEqual(len(manifest["model_sha256"]), 64)
        self.assertEqual(manifest["diagnostic_threshold"], 0.7)
        self.assertIn("heuristic_proxy", manifest["label_source"])
        self.assertTrue(manifest["frozen_before_holdout"])
        config = load_config()
        with materialized_experiment_inputs(config) as project_root:
            execution = build_execution_manifest(
                config, method="B3", condition="C0", profile="P4", seed=42,
                run=1, method_manifest=manifest, project_root=project_root,
            )
        self.assertEqual(execution["method_manifest"], manifest)


if __name__ == "__main__":
    unittest.main()
