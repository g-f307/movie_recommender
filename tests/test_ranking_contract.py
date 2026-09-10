import json
import math
import unittest

from cinebot_ml.ranking.contracts import (
    ContractValidationError,
    RankingResult,
    RecommendationItem,
    RecommendationRequest,
    Recommender,
    rank_scored_candidates,
)


def request(**changes):
    values = {
        "experiment_id": "experiment-v1.0__b0__c0__p0__s42",
        "protocol_version": "protocol-v1.0",
        "method": "B0",
        "method_version": "1.0",
        "condition": "C0",
        "profile": "P0",
        "seed": 42,
        "logical_timestamp": "2026-09-09T04:00:00Z",
        "candidate_movie_ids": (1, 2, 3),
        "k": 2,
        "unit_id": "synthetic-agent-001",
        "session_id": "session-001",
        "profile_data": {"genres": ["drama"]},
        "history": ({"timestamp": "2026-09-09T03:00:00Z", "movie_id": 9, "feedback": "like"},),
    }
    values.update(changes)
    return RecommendationRequest(**values)


def result(items=(), **changes):
    values = {
        "experiment_id": "experiment-v1.0__b0__c0__p0__s42",
        "method": "B0",
        "method_version": "1.0",
        "condition": "C0",
        "profile": "P0",
        "seed": 42,
        "logical_timestamp": "2026-09-09T04:00:00Z",
        "status": "completed",
        "latency_ms": 1.25,
        "ranked_items": tuple(items),
        "metadata": {"candidate_set_id": "candidate-set-001"},
    }
    values.update(changes)
    return RankingResult(**values)


class RankingContractTests(unittest.TestCase):
    def test_ranking_valido_e_serializavel(self):
        req = request()
        ranked = rank_scored_candidates([(1, 0.9, {"title": "A"}), (2, 0.8, {})], req.k)
        output = result(ranked)
        output.validate_against(req)
        reconstructed = RankingResult.from_json(output.to_json())
        self.assertEqual(reconstructed, output)
        self.assertEqual(RecommendationRequest.from_json(req.to_json()), req)
        json.dumps(output.to_dict(), allow_nan=False)

    def test_ranking_vazio_e_permitido_sem_candidatos(self):
        req = request(candidate_movie_ids=())
        output = result()
        output.validate_against(req)

    def test_ranking_vazio_com_candidatos_exige_falha(self):
        with self.assertRaisesRegex(ContractValidationError, "vazio"):
            result().validate_against(request())

    def test_filme_repetido_e_bloqueado(self):
        items = (RecommendationItem(1, 1, 0.9), RecommendationItem(1, 2, 0.8))
        with self.assertRaisesRegex(ContractValidationError, "duplicado"):
            result(items).validate_against(request())
        with self.assertRaisesRegex(ContractValidationError, "duplicado"):
            rank_scored_candidates([(1, 0.9, {}), (1, 0.8, {})], 2)

    def test_posicao_ausente_ou_descontinua_e_bloqueada(self):
        items = (RecommendationItem(1, 1, 0.9), RecommendationItem(2, 3, 0.8))
        with self.assertRaisesRegex(ContractValidationError, "contínuos"):
            result(items).validate_against(request())

    def test_score_nan_infinito_e_nao_numerico_sao_bloqueados(self):
        for score in (math.nan, math.inf, -math.inf, "0.5", True):
            with self.subTest(score=score), self.assertRaises(ContractValidationError):
                RecommendationItem(1, 1, score)

    def test_item_fora_do_conjunto_candidato_e_bloqueado(self):
        with self.assertRaisesRegex(ContractValidationError, "fora"):
            result((RecommendationItem(99, 1, 0.9),)).validate_against(request())

    def test_ranking_maior_que_k_e_bloqueado(self):
        items = (
            RecommendationItem(1, 1, 0.9),
            RecommendationItem(2, 2, 0.8),
            RecommendationItem(3, 3, 0.7),
        )
        with self.assertRaisesRegex(ContractValidationError, "mais itens que K"):
            result(items).validate_against(request(k=2))

    def test_ranking_rejeita_item_com_tipo_incorreto(self):
        with self.assertRaisesRegex(ContractValidationError, "RecommendationItem"):
            result(({"movie_id": 1, "rank": 1, "score": 0.9},))

    def test_empate_tem_desempate_deterministico_por_movie_id(self):
        ranked = rank_scored_candidates([(3, 0.5, {}), (1, 0.5, {}), (2, 0.5, {})], 3)
        self.assertEqual([item.movie_id for item in ranked], [1, 2, 3])
        self.assertEqual([item.rank for item in ranked], [1, 2, 3])

    def test_ordem_manual_incorreta_e_bloqueada(self):
        items = (RecommendationItem(2, 1, 0.5), RecommendationItem(1, 2, 0.5))
        with self.assertRaisesRegex(ContractValidationError, "ordem"):
            result(items).validate_against(request())

    def test_historico_futuro_igual_ou_sem_timestamp_e_bloqueado(self):
        future = ({"timestamp": "2026-09-09T05:00:00Z", "movie_id": 1},)
        equal = ({"timestamp": "2026-09-09T04:00:00Z", "movie_id": 1},)
        missing = ({"movie_id": 1},)
        for history in (future, equal, missing):
            with self.subTest(history=history), self.assertRaises(ContractValidationError):
                request(history=history)

    def test_timestamp_sem_fuso_e_bloqueado(self):
        with self.assertRaisesRegex(ContractValidationError, "fuso"):
            request(logical_timestamp="2026-09-09T04:00:00")

    def test_profile_data_deve_ser_objeto(self):
        with self.assertRaisesRegex(ContractValidationError, "objeto"):
            request(profile_data=["drama"])

    def test_metadado_desconhecido_pessoal_ou_secreto_e_bloqueado(self):
        with self.assertRaisesRegex(ContractValidationError, "não permitido"):
            RecommendationItem(1, 1, 0.5, {"debug": True})
        with self.assertRaisesRegex(ContractValidationError, "proibido"):
            RecommendationItem(1, 1, 0.5, {"score_components": {"api_token": "sensitive"}})
        with self.assertRaisesRegex(ContractValidationError, "proibido"):
            request(profile_data={"user_email": "person@example.test"})

    def test_resultado_de_falha_e_explicito(self):
        failed = result(status="failed", error_code="NO_ELIGIBLE_ITEMS", error_message="Sem itens")
        failed.validate_against(request())
        with self.assertRaisesRegex(ContractValidationError, "error_code"):
            result(status="failed")

    def test_identidade_do_resultado_deve_corresponder_ao_pedido(self):
        ranked = (RecommendationItem(1, 1, 0.9),)
        with self.assertRaisesRegex(ContractValidationError, "method"):
            result(ranked, method="B1").validate_against(request())

    def test_recommender_e_um_protocolo_estrutural(self):
        class ExampleRecommender:
            method = "B0"
            method_version = "1.0"

            def recommend(self, req):
                ranked = rank_scored_candidates([(1, 1.0, {})], req.k)
                return result(ranked)

        self.assertIsInstance(ExampleRecommender(), Recommender)


if __name__ == "__main__":
    unittest.main()
