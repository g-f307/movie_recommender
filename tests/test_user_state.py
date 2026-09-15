import dataclasses
import json
import math
import unittest

from cinebot_ml.personalization import (
    FeedbackEvent,
    PersonalizationContractError,
    StateSnapshot,
    UserState,
)


def event(**changes):
    values = {
        "event_id": "event-001",
        "movie_id": 101,
        "feedback": "like",
        "timestamp": "2026-09-15T10:00:00Z",
        "sequence": 1,
        "source": "synthetic_user",
        "metadata": {"context": "cold-start"},
    }
    values.update(changes)
    return FeedbackEvent(**values)


def state(**changes):
    history = changes.pop("history", (event(),))
    values = {
        "subject_id": "synthetic-agent-001",
        "identity_kind": "synthetic",
        "logical_timestamp": "2026-09-15T10:00:00Z",
        "version": len(history),
        "initial_profile": {"ranked_genres": ["drama", "comedia"]},
        "learned_preferences": {"genre_affinity": {"drama": 0.75}},
        "presented_movie_ids": (101,),
        "consumed_movie_ids": (101,),
        "history": history,
    }
    values.update(changes)
    return UserState(**values)


class UserStateContractTests(unittest.TestCase):
    def test_estado_inicial_valido_sem_historico_ou_preferencias(self):
        initial = state(
            history=(),
            initial_profile={},
            learned_preferences={},
            presented_movie_ids=(),
            consumed_movie_ids=(),
        )
        self.assertEqual(initial.version, 0)
        self.assertEqual(initial.evaluated_movie_ids, ())
        snapshot = StateSnapshot(initial)
        self.assertEqual(StateSnapshot.from_json(snapshot.to_json()), snapshot)

    def test_like_e_dislike_sao_eventos_explicitos(self):
        liked = event()
        disliked = event(event_id="event-002", feedback="dislike", sequence=2)
        self.assertEqual(liked.feedback, "like")
        self.assertEqual(disliked.feedback, "dislike")
        with self.assertRaisesRegex(PersonalizationContractError, "feedback inválido"):
            event(feedback="missing")

    def test_timestamp_invalido_ou_sem_fuso_e_rejeitado(self):
        for timestamp in ("invalid", "2026-09-15T10:00:00", ""):
            with self.subTest(timestamp=timestamp), self.assertRaises(PersonalizationContractError):
                event(timestamp=timestamp)

    def test_evento_futuro_e_rejeitado(self):
        future = event(timestamp="2026-09-15T10:00:01Z")
        with self.assertRaisesRegex(PersonalizationContractError, "futuro"):
            state(history=(future,))

    def test_evento_no_timestamp_do_estado_e_permitido(self):
        current = state()
        self.assertEqual(current.history[0].timestamp, current.logical_timestamp)

    def test_evento_e_sequencia_duplicados_sao_rejeitados(self):
        duplicate_id = event(event_id="event-001", sequence=2, timestamp="2026-09-15T10:01:00Z")
        with self.assertRaisesRegex(PersonalizationContractError, "event_id duplicado"):
            state(history=(event(), duplicate_id), logical_timestamp="2026-09-15T10:01:00Z")

        duplicate_sequence = event(event_id="event-002", sequence=1)
        with self.assertRaisesRegex(PersonalizationContractError, "sequence duplicada"):
            state(history=(event(), duplicate_sequence))

    def test_historico_fora_de_ordem_e_rejeitado(self):
        first = event()
        second = event(
            event_id="event-002",
            movie_id=102,
            timestamp="2026-09-15T10:01:00Z",
            sequence=2,
        )
        with self.assertRaisesRegex(PersonalizationContractError, "ordem cronológica"):
            state(
                history=(second, first),
                logical_timestamp="2026-09-15T10:01:00Z",
                presented_movie_ids=(101, 102),
            )

    def test_empate_temporal_usa_sequencia_deterministica(self):
        first = event()
        second = event(event_id="event-002", movie_id=102, sequence=2)
        ordered = state(history=(first, second), presented_movie_ids=(101, 102))
        self.assertEqual([item.event_id for item in ordered.history], ["event-001", "event-002"])

    def test_sequencia_deve_ser_continua_e_corresponder_a_versao(self):
        skipped = event(sequence=2)
        with self.assertRaisesRegex(PersonalizationContractError, "começar em 1"):
            state(history=(skipped,))
        with self.assertRaisesRegex(PersonalizationContractError, "quantidade de eventos"):
            state(version=2)

    def test_snapshot_e_estruturas_internas_sao_imutaveis(self):
        current = state()
        snapshot = StateSnapshot(
            current,
            previous_state_id="previous-state-id",
            transition_event_id="event-001",
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.previous_state_id = "changed"
        with self.assertRaises(TypeError):
            current.initial_profile["new"] = True
        with self.assertRaises(TypeError):
            current.learned_preferences["genre_affinity"]["drama"] = 0.1

    def test_serializacao_e_reconstrucao_preservam_estado(self):
        current = state()
        reconstructed = UserState.from_json(current.to_json())
        self.assertEqual(reconstructed, current)
        self.assertEqual(reconstructed.state_id, current.state_id)
        json.dumps(current.to_dict(), allow_nan=False)

    def test_identidade_e_deterministica_e_sensivel_ao_conteudo(self):
        first = state()
        same = state()
        changed = state(initial_profile={"ranked_genres": ["terror"]})
        changed_event = state(history=(event(feedback="dislike"),))
        self.assertEqual(first.state_id, same.state_id)
        self.assertNotEqual(first.state_id, changed.state_id)
        self.assertNotEqual(first.state_id, changed_event.state_id)

    def test_snapshot_valida_cadeia_e_ultimo_evento(self):
        current = state()
        valid = StateSnapshot(current, "initial-state-id", "event-001")
        self.assertEqual(StateSnapshot.from_dict(valid.to_dict()), valid)
        with self.assertRaisesRegex(PersonalizationContractError, "último evento"):
            StateSnapshot(current, "initial-state-id", "wrong-event")
        with self.assertRaisesRegex(PersonalizationContractError, "não pode possuir"):
            StateSnapshot(state(history=(), presented_movie_ids=(), consumed_movie_ids=()), "previous", None)

    def test_itens_avaliados_e_consumidos_devem_ter_sido_apresentados(self):
        with self.assertRaisesRegex(PersonalizationContractError, "avaliado"):
            state(presented_movie_ids=(), consumed_movie_ids=())
        with self.assertRaisesRegex(PersonalizationContractError, "consumido"):
            state(history=(), presented_movie_ids=(), consumed_movie_ids=(202,))

    def test_tuplas_ids_e_tipos_invalidos_sao_rejeitados(self):
        with self.assertRaisesRegex(PersonalizationContractError, "tupla"):
            state(presented_movie_ids=[101])
        with self.assertRaisesRegex(PersonalizationContractError, "duplicidades"):
            state(presented_movie_ids=(101, 101))
        with self.assertRaisesRegex(PersonalizationContractError, "movie_id"):
            event(movie_id=True)

    def test_metadado_secreto_e_json_nao_finito_sao_rejeitados(self):
        with self.assertRaisesRegex(PersonalizationContractError, "proibido"):
            event(metadata={"api_token": "sensitive"})
        with self.assertRaisesRegex(PersonalizationContractError, "JSON finitos"):
            state(learned_preferences={"genre_affinity": math.nan})

    def test_schema_identidade_e_fonte_sao_validados(self):
        with self.assertRaisesRegex(PersonalizationContractError, "schema_version"):
            event(schema_version="2.0")
        with self.assertRaisesRegex(PersonalizationContractError, "identity_kind"):
            state(identity_kind="raw_person")
        with self.assertRaisesRegex(PersonalizationContractError, "source"):
            event(source="unknown")


if __name__ == "__main__":
    unittest.main()
