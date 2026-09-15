import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cinebot_ml.experiment_config import build_execution_manifest, load_config
from cinebot_ml.personalization import FeedbackEvent, UserState
from cinebot_ml.ranking import (
    CandidateSet,
    EligibilityPolicy,
    RecommendationRequest,
    Recommender,
    StaticPersonalizedConfigError,
    StaticPersonalizedRecommender,
    load_static_personalized_config,
)
from cinebot_ml.ranking.contracts import ContractValidationError
from cinebot_ml.ranking.static_personalized import DEFAULT_B4_CONFIG_PATH


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
        candidate_set_id="candidate-set-b4-test",
        version="1.0",
        catalog_path="data/filmes.json",
        catalog_sha256="e" * 64,
        policy=EligibilityPolicy(),
        filters={},
        movies=tuple(movies),
        total_catalog_items=len(movies),
    )


def user_state(initial_profile=None, *, feedback=None):
    history = ()
    presented = ()
    if feedback is not None:
        history = (
            FeedbackEvent(
                event_id="event-001",
                movie_id=1,
                feedback=feedback,
                timestamp="2026-09-15T03:00:00Z",
                sequence=1,
                source="synthetic_user",
            ),
        )
        presented = (1,)
    return UserState(
        subject_id="synthetic-agent-001",
        identity_kind="synthetic",
        logical_timestamp="2026-09-15T03:00:00Z",
        version=len(history),
        initial_profile=initial_profile or {},
        learned_preferences={"ignored": {"terror": 1.0}} if history else {},
        presented_movie_ids=presented,
        consumed_movie_ids=presented,
        history=history,
    )


def request(candidates, *, profile="P4", history=(), profile_data=None, **changes):
    values = {
        "experiment_id": "experiment-v1.0__b4__c0__p4__s42",
        "protocol_version": "protocol-v1.0",
        "method": "B4",
        "method_version": "1.0",
        "condition": "C0",
        "profile": profile,
        "seed": 42,
        "logical_timestamp": "2026-09-15T04:00:00Z",
        "candidate_movie_ids": candidates.movie_ids,
        "k": 10,
        "unit_id": "synthetic-agent-001",
        "profile_data": profile_data or {},
        "history": history,
    }
    values.update(changes)
    return RecommendationRequest(**values)


class StaticPersonalizedBaselineTests(unittest.TestCase):
    def test_perfil_completo_considera_campos_declarados(self):
        candidates = candidate_set([
            movie(1, genres=("drama",), year="1995", votes=5, rating=8.5, director="Jane Doe", keywords=("espaco",)),
            movie(2, genres=("terror",), year="2020", votes=1000, rating=7.0, director="John Doe", keywords=("cidade",)),
        ])
        initial = {
            "ranked_genres": ["drama", "comedia", "terror"],
            "decade_preference": "antes_2000",
            "popularity_preference": "joia_escondida",
            "preferred_directors": ["Jane Doe"],
            "preferred_keywords": ["espaco"],
        }
        result = StaticPersonalizedRecommender(candidates, user_state(initial)).recommend(
            request(candidates, profile="P5")
        )
        self.assertEqual(result.ranked_items[0].movie_id, 1)

    def test_perfil_sem_genero_usa_demais_campos_permitidos(self):
        candidates = candidate_set([
            movie(1, genres=("drama",), year="1995"),
            movie(2, genres=("terror",), year="2020"),
        ])
        result = StaticPersonalizedRecommender(
            candidates,
            user_state({"decade_preference": "moderno"}),
        ).recommend(request(candidates, profile="P3"))
        self.assertEqual(result.ranked_items[0].movie_id, 2)

    def test_p0_possui_fallback_canonico(self):
        candidates = candidate_set([movie(3), movie(1), movie(2)])
        result = StaticPersonalizedRecommender(candidates, user_state()).recommend(
            request(candidates, profile="P0")
        )
        self.assertEqual([item.movie_id for item in result.ranked_items], [1, 2, 3])
        self.assertTrue(all(item.score == 0.0 for item in result.ranked_items))

    def test_historico_like_dislike_e_preferencias_aprendidas_nao_afetam_b4(self):
        candidates = candidate_set([movie(1, genres=("drama",)), movie(2, genres=("terror",))])
        initial = {"ranked_genres": ["drama"]}
        baseline = StaticPersonalizedRecommender(candidates, user_state(initial)).recommend(
            request(candidates, profile="P1")
        )
        for feedback in ("like", "dislike"):
            observed_state = user_state(initial, feedback=feedback)
            history = ({"timestamp": "2026-09-15T03:00:00Z", "movie_id": 1, "feedback": feedback},)
            result = StaticPersonalizedRecommender(candidates, observed_state).recommend(
                request(
                    candidates,
                    profile="P1",
                    history=history,
                    profile_data={"ranked_genres": ["terror"]},
                )
            )
            self.assertEqual(result.ranked_items, baseline.ranked_items)

    def test_alterar_perfil_inicial_pode_alterar_ranking(self):
        candidates = candidate_set([movie(1, genres=("drama",)), movie(2, genres=("terror",))])
        drama = StaticPersonalizedRecommender(
            candidates, user_state({"ranked_genres": ["drama"]})
        ).recommend(request(candidates, profile="P1"))
        terror = StaticPersonalizedRecommender(
            candidates, user_state({"ranked_genres": ["terror"]})
        ).recommend(request(candidates, profile="P1"))
        self.assertNotEqual(drama.ranked_items[0].movie_id, terror.ranked_items[0].movie_id)

    def test_candidato_sem_metadados_nao_interrompe_ranking(self):
        candidates = candidate_set([{"id": 1}, movie(2, genres=("terror",))])
        result = StaticPersonalizedRecommender(
            candidates, user_state({"ranked_genres": ["terror"]})
        ).recommend(request(candidates, profile="P1"))
        self.assertEqual(len(result.ranked_items), 2)
        self.assertEqual(result.ranked_items[0].movie_id, 2)

    def test_empate_e_reexecucao_sao_deterministicos(self):
        candidates = candidate_set([movie(3), movie(1), movie(2)])
        recommender = StaticPersonalizedRecommender(candidates, user_state())
        payload = request(candidates, profile="P0")
        first = recommender.recommend(payload)
        second = recommender.recommend(payload)
        self.assertEqual(first.ranked_items, second.ranked_items)
        self.assertEqual([item.movie_id for item in first.ranked_items], [1, 2, 3])

    def test_ranking_vazio_e_permitido_sem_candidatos(self):
        candidates = candidate_set([])
        result = StaticPersonalizedRecommender(candidates, user_state()).recommend(
            request(candidates, profile="P0")
        )
        self.assertEqual(result.ranked_items, ())

    def test_implementa_contrato_e_valida_identidade(self):
        candidates = candidate_set([movie(1)])
        recommender = StaticPersonalizedRecommender(candidates, user_state())
        self.assertIsInstance(recommender, Recommender)
        with self.assertRaisesRegex(ContractValidationError, "method=B4"):
            recommender.recommend(request(candidates, method="B1"))
        with self.assertRaisesRegex(ContractValidationError, "CandidateSet"):
            recommender.recommend(request(candidates, candidate_movie_ids=(99,)))
        with self.assertRaisesRegex(ContractValidationError, "unit_id"):
            recommender.recommend(request(candidates, unit_id="synthetic-agent-002"))

    def test_configuracao_invalida_e_lock_alterado_sao_bloqueados(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b4_static_v1.yaml"
            payload = yaml.safe_load(DEFAULT_B4_CONFIG_PATH.read_text(encoding="utf-8"))
            payload["features"]["genres"]["weight"] = -1
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".lock.json").write_text(json.dumps({"sha256": digest}), encoding="utf-8")
            with self.assertRaisesRegex(StaticPersonalizedConfigError, "inválida"):
                load_static_personalized_config(path)

            path.write_bytes(DEFAULT_B4_CONFIG_PATH.read_bytes())
            path.with_suffix(".lock.json").write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
            with self.assertRaisesRegex(StaticPersonalizedConfigError, "congelada"):
                load_static_personalized_config(path)

    def test_manifesto_registra_formula_pesos_estado_e_ausencia_de_holdout(self):
        candidates = candidate_set([movie(1)])
        recommender = StaticPersonalizedRecommender(
            candidates, user_state({"ranked_genres": ["drama"]})
        )
        manifest = recommender.method_manifest()
        self.assertEqual(manifest["method"], "B4")
        self.assertEqual(manifest["state_source"], "immutable_initial_profile")
        self.assertEqual(manifest["config_snapshot"]["selection"]["policy"], "a_priori_protocol_v1")
        self.assertIn("features", manifest["config_snapshot"])
        self.assertFalse(manifest["holdout_used_for_selection"])
        execution = build_execution_manifest(
            load_config(),
            method="B4",
            condition="C0",
            profile="P4",
            seed=42,
            run=1,
            method_manifest=manifest,
        )
        self.assertEqual(execution["method_manifest"], manifest)


if __name__ == "__main__":
    unittest.main()
