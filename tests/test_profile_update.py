import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cinebot_ml.personalization import (
    FeedbackEvent,
    ProfileUpdateError,
    ProfileUpdater,
    StateSnapshot,
    UserState,
    load_profile_update_config,
)
from cinebot_ml.personalization.update import DEFAULT_PROFILE_UPDATE_CONFIG_PATH


def initial_snapshot(*, initial_profile=None, learned_preferences=None):
    return StateSnapshot(
        UserState(
            subject_id="synthetic-agent-001",
            identity_kind="synthetic",
            logical_timestamp="2026-09-15T09:00:00Z",
            initial_profile=initial_profile or {},
            learned_preferences=learned_preferences or {},
        )
    )


def feedback(sequence=1, **changes):
    values = {
        "event_id": f"event-{sequence:03d}",
        "movie_id": 101,
        "feedback": "like",
        "timestamp": f"2026-09-15T{9 + sequence:02d}:00:00Z",
        "sequence": sequence,
        "source": "synthetic_user",
    }
    values.update(changes)
    return FeedbackEvent(**values)


def movie(**changes):
    values = {
        "id": 101,
        "perfil": "drama",
        "genero": "drama",
        "generos_secundarios": ["comedia"],
        "diretor": "Jane Doe",
        "palavras_chave": ["viagem espacial", "amizade"],
        "sinopse": "Uma amizade durante uma longa viagem espacial.",
    }
    values.update(changes)
    return values


def apply(updater, snapshot, event, item=None, cutoff=None):
    return updater.update(
        snapshot,
        event,
        movie() if item is None else item,
        logical_timestamp=cutoff or event.timestamp,
    )


class ProfileUpdateTests(unittest.TestCase):
    def test_like_aumenta_afinidades_e_registra_contribuicao(self):
        updated = apply(ProfileUpdater(), initial_snapshot(), feedback())
        preferences = updated.state.to_dict()["learned_preferences"]
        self.assertGreater(preferences["affinities"]["genres"]["drama"], 0)
        self.assertGreater(preferences["affinities"]["directors"]["jane doe"], 0)
        self.assertGreater(preferences["affinities"]["keywords"]["amizade"], 0)
        self.assertGreater(preferences["affinities"]["synopsis_terms"]["amizade"], 0)
        contribution = preferences["event_contributions"]["event-001"]
        self.assertEqual(contribution["feedback"], "like")
        self.assertEqual(contribution["direction"], 1.0)

    def test_dislike_reduz_afinidades(self):
        updated = apply(ProfileUpdater(), initial_snapshot(), feedback(feedback="dislike"))
        affinities = updated.state.to_dict()["learned_preferences"]["affinities"]
        self.assertLess(affinities["genres"]["drama"], 0)
        self.assertLess(affinities["directors"]["jane doe"], 0)

    def test_like_e_dislike_no_mesmo_item_permanecem_no_historico(self):
        updater = ProfileUpdater()
        liked = apply(updater, initial_snapshot(), feedback())
        disliked_event = feedback(
            2,
            feedback="dislike",
            timestamp="2026-09-15T11:00:00Z",
        )
        disliked = apply(updater, liked, disliked_event)
        self.assertEqual([event.feedback for event in disliked.state.history], ["like", "dislike"])
        contributions = disliked.state.to_dict()["learned_preferences"]["event_contributions"]
        self.assertEqual(set(contributions), {"event-001", "event-002"})

    def test_eventos_contraditorios_em_itens_semelhantes_sao_preservados(self):
        updater = ProfileUpdater()
        first = apply(updater, initial_snapshot(), feedback(), movie())
        second_event = feedback(
            2,
            event_id="event-002",
            movie_id=202,
            feedback="dislike",
            timestamp="2026-09-15T11:00:00Z",
        )
        second_movie = movie(id=202, diretor="Outra Pessoa")
        second = apply(updater, first, second_event, second_movie)
        self.assertEqual(len(second.state.history), 2)
        self.assertIn("event-001", second.state.to_dict()["learned_preferences"]["event_contributions"])

    def test_item_sem_diretor_ou_sinopse_nao_gera_valor_invalido(self):
        updated = apply(
            ProfileUpdater(),
            initial_snapshot(),
            feedback(),
            movie(diretor=None, sinopse=None, palavras_chave=None),
        )
        affinities = updated.state.to_dict()["learned_preferences"]["affinities"]
        self.assertEqual(affinities["directors"], {})
        self.assertEqual(affinities["keywords"], {})
        self.assertEqual(affinities["synopsis_terms"], {})
        self.assertGreater(affinities["genres"]["drama"], 0)

    def test_estado_sem_perfil_declarado_e_permitido(self):
        before = initial_snapshot(initial_profile={})
        after = apply(ProfileUpdater(), before, feedback())
        self.assertEqual(after.state.to_dict()["initial_profile"], {})
        self.assertTrue(after.state.to_dict()["learned_preferences"]["affinities"]["genres"])

    def test_evento_futuro_e_rejeitado(self):
        with self.assertRaisesRegex(ProfileUpdateError, "futuro"):
            apply(
                ProfileUpdater(),
                initial_snapshot(),
                feedback(timestamp="2026-09-15T11:00:00Z"),
                cutoff="2026-09-15T10:00:00Z",
            )

    def test_evento_anterior_ao_estado_e_rejeitado(self):
        with self.assertRaisesRegex(ProfileUpdateError, "ordem temporal"):
            apply(
                ProfileUpdater(),
                initial_snapshot(),
                feedback(timestamp="2026-09-15T08:00:00Z"),
            )

    def test_evento_duplicado_e_sequencia_invalida_sao_rejeitados(self):
        updater = ProfileUpdater()
        after = apply(updater, initial_snapshot(), feedback())
        with self.assertRaisesRegex(ProfileUpdateError, "duplicado"):
            apply(
                updater,
                after,
                feedback(2, event_id="event-001", timestamp="2026-09-15T11:00:00Z"),
            )
        with self.assertRaisesRegex(ProfileUpdateError, "sequência"):
            apply(
                updater,
                after,
                feedback(3, timestamp="2026-09-15T12:00:00Z"),
            )

    def test_afinidades_permanecem_nos_limites(self):
        updater = ProfileUpdater()
        current = initial_snapshot()
        for sequence in range(1, 9):
            current = apply(
                updater,
                current,
                feedback(
                    sequence,
                    event_id=f"event-{sequence:03d}",
                    timestamp=f"2026-09-{15 + sequence:02d}T10:00:00Z",
                ),
            )
        affinities = current.state.to_dict()["learned_preferences"]["affinities"]
        scores = [score for values in affinities.values() for score in values.values()]
        self.assertTrue(all(-1.0 <= score <= 1.0 for score in scores))

    def test_snapshot_anterior_e_perfil_inicial_permanecem_immutaveis(self):
        before = initial_snapshot(initial_profile={"ranked_genres": ["drama"]})
        before_payload = before.to_dict()
        after = apply(ProfileUpdater(), before, feedback())
        self.assertEqual(before.to_dict(), before_payload)
        self.assertEqual(after.state.to_dict()["initial_profile"], before_payload["state"]["initial_profile"])
        self.assertEqual(after.previous_state_id, before.state.state_id)
        self.assertNotEqual(after.state.state_id, before.state.state_id)

    def test_reexecucao_e_deterministica(self):
        updater = ProfileUpdater()
        before = initial_snapshot()
        first = apply(updater, before, feedback())
        second = apply(updater, before, feedback())
        self.assertEqual(first, second)
        self.assertEqual(first.snapshot_id, second.snapshot_id)

    def test_ausencia_de_feedback_nao_modifica_estado(self):
        updater = ProfileUpdater()
        before = initial_snapshot()
        after = updater.update(
            before,
            None,
            None,
            logical_timestamp="2026-09-15T10:00:00Z",
        )
        self.assertIs(after, before)

    def test_item_e_evento_devem_corresponder(self):
        with self.assertRaisesRegex(ProfileUpdateError, "diverge"):
            apply(ProfileUpdater(), initial_snapshot(), feedback(), movie(id=999))
        with self.assertRaisesRegex(ProfileUpdateError, "obrigatório"):
            ProfileUpdater().update(
                initial_snapshot(),
                feedback(),
                None,
                logical_timestamp="2026-09-15T10:00:00Z",
            )

    def test_score_preexistente_nao_numerico_e_rejeitado(self):
        before = initial_snapshot(
            learned_preferences={"affinities": {"genres": {"drama": "alto"}}}
        )
        with self.assertRaisesRegex(ProfileUpdateError, "numérica"):
            apply(ProfileUpdater(), before, feedback())

    def test_configuracao_invalida_e_lock_alterado_sao_bloqueados(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile_update_v1.yaml"
            payload = yaml.safe_load(DEFAULT_PROFILE_UPDATE_CONFIG_PATH.read_text(encoding="utf-8"))
            payload["update"]["learning_rate"] = 2.0
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".lock.json").write_text(json.dumps({"sha256": digest}), encoding="utf-8")
            with self.assertRaisesRegex(ProfileUpdateError, "inválida"):
                load_profile_update_config(path)

            path.write_bytes(DEFAULT_PROFILE_UPDATE_CONFIG_PATH.read_bytes())
            path.with_suffix(".lock.json").write_text(json.dumps({"sha256": "0" * 64}), encoding="utf-8")
            with self.assertRaisesRegex(ProfileUpdateError, "congelada"):
                load_profile_update_config(path)

    def test_configuracao_com_parametro_nao_finito_e_bloqueada(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile_update_v1.yaml"
            payload = yaml.safe_load(DEFAULT_PROFILE_UPDATE_CONFIG_PATH.read_text(encoding="utf-8"))
            payload["features"]["genres"]["weight"] = float("nan")
            path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".lock.json").write_text(json.dumps({"sha256": digest}), encoding="utf-8")
            with self.assertRaisesRegex(ProfileUpdateError, "finitos"):
                load_profile_update_config(path)

    def test_manifesto_registra_politica_parametros_e_holdout(self):
        manifest = ProfileUpdater().method_manifest()
        self.assertEqual(manifest["component"], "profile_update")
        self.assertEqual(manifest["policy"], "bounded_additive_decay")
        self.assertEqual(manifest["config_snapshot"]["selection"]["policy"], "a_priori_protocol_v1")
        self.assertFalse(manifest["holdout_used_for_selection"])
        self.assertIn("config_sha256", manifest)


if __name__ == "__main__":
    unittest.main()
