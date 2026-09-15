import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import yaml

from cinebot_ml.personalization import FeedbackEvent, ProfileUpdater, StateSnapshot, UserState
from cinebot_ml.ranking import (
    IncrementalConfigError,
    IncrementalRecommender,
    Recommender,
    StaticPersonalizedRecommender,
    load_incremental_config,
)
from cinebot_ml.ranking.contracts import ContractValidationError
from cinebot_ml.ranking.incremental import DEFAULT_B5_CONFIG_PATH
from tests.test_baseline_static import candidate_set, movie, request, user_state


def snapshot(profile=None):
    return StateSnapshot(UserState("synthetic-agent-001", "synthetic", "2026-09-15T02:00:00Z", initial_profile=profile or {}))


def b5_request(candidates, **changes):
    return request(candidates, method="B5", experiment_id="exp-b5", **changes)


class IncrementalBaselineTests(unittest.TestCase):
    def setUp(self):
        self.candidates = candidate_set([movie(1, genres=("drama",)), movie(2, genres=("terror",))])
        self.profile = {"ranked_genres": ["drama"]}

    def updated(self, feedback):
        before = snapshot(self.profile)
        event = FeedbackEvent("event-001", 1, feedback, "2026-09-15T03:00:00Z", 1, "synthetic_user")
        return ProfileUpdater().update(before, event, self.candidates.movies[0], logical_timestamp=event.timestamp)

    def test_sem_feedback_equivale_ao_b4(self):
        state = snapshot(self.profile)
        b5 = IncrementalRecommender(self.candidates, state).recommend(b5_request(self.candidates, profile="P1"))
        b4 = StaticPersonalizedRecommender(self.candidates, state.state).recommend(request(self.candidates, profile="P1"))
        self.assertEqual([(i.movie_id,i.score) for i in b5.ranked_items], [(i.movie_id,i.score) for i in b4.ranked_items])
        self.assertIsInstance(IncrementalRecommender(self.candidates, state), Recommender)

    def test_like_e_dislike_alteram_score_em_direcoes_opostas(self):
        base = IncrementalRecommender(self.candidates, snapshot(self.profile)).recommend(b5_request(self.candidates, profile="P1"))
        scores = {}
        for value in ("like", "dislike"):
            state = self.updated(value)
            remaining = candidate_set([movie(2, genres=("drama",))])
            scores[value] = IncrementalRecommender(remaining, state).recommend(b5_request(remaining, profile="P1" )).ranked_items[0].score
        self.assertGreater(scores["like"], scores["dislike"])
        self.assertTrue(all(0 <= i.score <= 1 for i in base.ranked_items))

    def test_estado_futuro_e_item_consumido_sao_bloqueados(self):
        future = snapshot(self.profile)
        with self.assertRaisesRegex(ContractValidationError, "futuro"):
            IncrementalRecommender(self.candidates, future).recommend(b5_request(self.candidates, logical_timestamp="2026-09-15T01:00:00Z"))
        consumed = self.updated("like")
        with self.assertRaisesRegex(ContractValidationError, "consumido"):
            IncrementalRecommender(self.candidates, consumed).recommend(b5_request(self.candidates))

    def test_determinismo_vazio_identidade_e_manifesto(self):
        state = snapshot()
        recommender = IncrementalRecommender(self.candidates, state)
        payload = b5_request(self.candidates, profile="P0")
        self.assertEqual(recommender.recommend(payload).ranked_items, recommender.recommend(payload).ranked_items)
        self.assertEqual(recommender.method_manifest()["state_version"], 0)
        empty = candidate_set([])
        self.assertEqual(IncrementalRecommender(empty, state).recommend(b5_request(empty, profile="P0")).ranked_items, ())
        with self.assertRaises(ContractValidationError): recommender.recommend(replace(payload, unit_id="other"))

    def test_condicoes_c0_a_c5_usam_o_mesmo_contrato(self):
        recommender = IncrementalRecommender(self.candidates, snapshot(self.profile))
        for condition in ("C0", "C1", "C2", "C3", "C4", "C5"):
            with self.subTest(condition=condition):
                result = recommender.recommend(b5_request(self.candidates, profile="P1", condition=condition))
                self.assertEqual(result.condition, condition)

    def test_sequencia_de_likes_e_feedback_contraditorio(self):
        updater = ProfileUpdater()
        current = snapshot({})
        first_movie = movie(10, genres=("drama",))
        first = FeedbackEvent("event-001", 10, "like", "2026-09-15T03:00:00Z", 1, "synthetic_user")
        current = updater.update(current, first, first_movie, logical_timestamp=first.timestamp)
        second_movie = movie(11, genres=("drama",))
        second = FeedbackEvent("event-002", 11, "like", "2026-09-15T04:00:00Z", 2, "synthetic_user")
        current = updater.update(current, second, second_movie, logical_timestamp=second.timestamp)
        positive = candidate_set([movie(20, genres=("drama",)), movie(21, genres=("terror",))])
        liked = IncrementalRecommender(positive, current).recommend(b5_request(positive, profile="P0"))
        self.assertEqual(liked.ranked_items[0].movie_id, 20)

        third = FeedbackEvent("event-003", 12, "dislike", "2026-09-15T05:00:00Z", 3, "synthetic_user")
        current = updater.update(current, third, movie(12, genres=("drama",)), logical_timestamp=third.timestamp)
        self.assertEqual([event.feedback for event in current.state.history], ["like", "like", "dislike"])

    def test_estado_com_afinidade_malformada_e_rejeitado(self):
        malformed = StateSnapshot(UserState(
            "synthetic-agent-001", "synthetic", "2026-09-15T02:00:00Z",
            learned_preferences={"affinities": {"genres": {"drama": "alto"}}},
        ))
        with self.assertRaisesRegex(ContractValidationError, "não numérica"):
            IncrementalRecommender(self.candidates, malformed).recommend(b5_request(self.candidates))

    def test_configuracao_invalida_e_lock_sao_bloqueados(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "b5_incremental_v1.yaml"
            payload = yaml.safe_load(DEFAULT_B5_CONFIG_PATH.read_text(encoding="utf-8"))
            payload["feedback_adjustment_weight"] = 2
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".lock.json").write_text(json.dumps({"sha256": digest}))
            with self.assertRaisesRegex(IncrementalConfigError, "inválida"):
                load_incremental_config(path)
            path.write_bytes(DEFAULT_B5_CONFIG_PATH.read_bytes())
            path.with_suffix(".lock.json").write_text(json.dumps({"sha256": "0" * 64}))
            with self.assertRaisesRegex(IncrementalConfigError, "congelada"):
                load_incremental_config(path)


if __name__ == "__main__": unittest.main()
