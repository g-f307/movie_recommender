import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from cinebot_ml.artifacts import ArtifactPreparationError, build_b2_artifact
from cinebot_ml.personalization import StateSnapshot, UserState
from cinebot_ml.ranking import (
    ContentRecommender,
    IncrementalRecommender,
    PopularityRecommender,
    RecommendationRequest,
    StaticPersonalizedRecommender,
    SupervisedArtifact,
    SupervisedConfigError,
    SupervisedRecommender,
    TfidfArtifact,
    TfidfRecommender,
    build_candidate_set,
    fit_tfidf_artifact,
)
from cinebot_ml.readiness import main, run_readiness
from cinebot_ml.schema import FEATURE_COLUMNS
from cinebot_ml.simulation import SimulationScenario, TemporalSimulator, build_agent


class SmokeProbabilityModel:
    classes_ = np.asarray([0, 1])

    def predict_proba(self, frame):
        scores = np.asarray(frame["nota"], dtype=float) / 10.0
        return np.column_stack((1.0 - scores, scores))


def catalog():
    genres = ("drama", "terror", "comedia", "acao", "romance", "scifi")
    return [
        {
            "id": index,
            "titulo": f"Filme {index}",
            "sinopse": "Uma história reproduzível de amizade e aventura.",
            "perfil": genre,
            "genero": genre,
            "generos_secundarios": [],
            "palavras_chave": ["amizade", "aventura"],
            "diretor": f"Diretor {index % 2}",
            "ano": "2020",
            "nota": 6.0 + index / 10,
            "votos": 100 * index,
            "duracao": 100,
            "streaming": [],
        }
        for index, genre in enumerate(genres, 1)
    ]


def supervised_artifact():
    return SupervisedArtifact(
        model=SmokeProbabilityModel(),
        model_path=Path("artifacts/smoke.joblib"),
        model_sha256="a" * 64,
        metadata_path=Path("artifacts/smoke.json"),
        metadata_sha256="b" * 64,
        metadata={
            "method_version": "1.0",
            "winner_model": "smoke_probability_model",
            "winner_params": {},
            "winner_threshold": 0.5,
            "feature_columns": list(FEATURE_COLUMNS),
            "positive_class": 1,
            "label_source": "heuristic_proxy_with_explicit_feedback_overrides",
            "fit_partition": "train",
            "selection_partitions": ["train", "validation"],
            "frozen_before_holdout": True,
        },
    )


class ExperimentReadinessTests(unittest.TestCase):
    def test_configuracoes_congeladas_e_imports(self):
        checks = run_readiness("ci")
        self.assertTrue(checks)
        self.assertEqual({check.status for check in checks}, {"ok"})

    def test_execucao_sem_env_real(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(main(["ci", "--json"]), 0)

    def test_artefato_obrigatorio_ausente_falha_explicitamente(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(SupervisedConfigError, "metadados B3"):
                SupervisedArtifact.load(root / "model.joblib", root / "metadata.json")

    def test_artefato_b2_e_gerado_somente_com_particao_de_treino(self):
        movies = catalog()
        assignments = {
            str(movie["id"]): "train" if index < 3 else "test"
            for index, movie in enumerate(movies)
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "b2.json"
            artifact = build_b2_artifact(movies, assignments, output)
            self.assertTrue(output.is_file())
            self.assertEqual(TfidfArtifact.load(output).artifact_id, artifact.artifact_id)
        with self.assertRaisesRegex(ArtifactPreparationError, "não contém filmes de treino"):
            build_b2_artifact(movies, {str(movie["id"]): "test" for movie in movies})

    def test_smoke_b0_a_b5_em_dados_minimos(self):
        movies = catalog()
        initial = StateSnapshot(
            UserState(
                "synthetic-smoke-001",
                "synthetic",
                "2027-01-01T00:00:00Z",
                initial_profile={
                    "ranked_genres": ["drama", "terror", "comedia"],
                    "decade_preference": "moderno",
                    "popularity_preference": "popular",
                },
            )
        )
        base = RecommendationRequest(
            "smoke-b0-b5",
            "protocol-v1.0",
            "B0",
            "1.0",
            "C0",
            "P4",
            42,
            "2027-01-01T00:00:01Z",
            (),
            5,
            unit_id=initial.state.subject_id,
            profile_data=initial.state.to_dict()["initial_profile"],
        )
        candidates = build_candidate_set(
            movies,
            base,
            catalog_path="fixtures/smoke_catalog.json",
            catalog_sha256="c" * 64,
        )
        tfidf = fit_tfidf_artifact(movies, partition="train")
        recommenders = {
            "B0": PopularityRecommender(candidates),
            "B1": ContentRecommender(candidates),
            "B2": TfidfRecommender(candidates, tfidf),
            "B3": SupervisedRecommender(candidates, supervised_artifact()),
            "B4": StaticPersonalizedRecommender(candidates, initial.state),
            "B5": IncrementalRecommender(candidates, initial),
        }
        for method, recommender in recommenders.items():
            with self.subTest(method=method):
                request = RecommendationRequest.from_dict(
                    {
                        **base.to_dict(),
                        "method": method,
                        "method_version": recommender.method_version,
                        "candidate_movie_ids": list(candidates.movie_ids),
                    }
                )
                result = recommender.recommend(request)
                self.assertEqual(result.status, "completed")
                self.assertGreater(len(result.ranked_items), 0)

    def test_smoke_c0_com_agente_sintetico(self):
        movies = catalog()
        agent = build_agent("consistent", 42, movies)
        initial = StateSnapshot(
            UserState(agent.agent_id, "synthetic", "2027-01-01T00:00:00Z")
        )
        scenario = SimulationScenario(
            "B5", "C0", "P0", 42, 5,
            "fixtures/smoke_catalog.json", "c" * 64,
            initial, "2027-01-01T00:00:00Z",
        )
        result = TemporalSimulator(movies).run(scenario, agent)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.steps, ())
        self.assertGreater(len(result.final_ranking.ranked_items), 0)


if __name__ == "__main__":
    unittest.main()
