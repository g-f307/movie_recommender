import tempfile
import unittest
from pathlib import Path

from cinebot_ml.analysis.ablation_execution import (
    AblationExecutionError,
    load_execution_config,
    safe_movie,
    validate_pairing,
)


class AblationExecutionTests(unittest.TestCase):
    @staticmethod
    def row(variant):
        return {
            "comparison_id": "pair", "variant": variant, "agent_id": "agent",
            "seed": 11, "condition": "C1", "candidate_set_id": "same",
            "relevance_id": "same-relevance",
        }

    def test_configuracao_oficial_usa_validacao_e_k5(self):
        config = load_execution_config()
        self.assertEqual(config.partition, "validation")
        self.assertEqual(config.k, 5)
        self.assertNotIn(42, config.seeds)

    def test_configuracao_invalida_e_rejeitada(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.yaml"
            path.write_text("version: '1.0'\nfrozen: true\npartition: test\n", encoding="utf-8")
            with self.assertRaisesRegex(AblationExecutionError, "Configuração inválida"):
                load_execution_config(path)

    def test_pareamento_incompleto_e_rejeitado(self):
        with self.assertRaisesRegex(AblationExecutionError, "incompleto"):
            validate_pairing([self.row("A0"), self.row("A1")])

    def test_pareamento_incompativel_e_rejeitado(self):
        rows = [self.row(f"A{index}") for index in range(7)]
        rows[-1] = {**rows[-1], "candidate_set_id": "outro"}
        with self.assertRaisesRegex(AblationExecutionError, "incompatível"):
            validate_pairing(rows)

    def test_pareamento_completo_e_aceito(self):
        validate_pairing([self.row(f"A{index}") for index in range(7)])

    def test_sanitizacao_remove_apenas_colisoes(self):
        movie = {"id": 1, "palavras_chave": ["space", "secret love"],
                 "sinopse": "space secret adventure"}
        sanitized = safe_movie(movie)
        self.assertEqual(sanitized["palavras_chave"], ["space"])
        self.assertEqual(sanitized["sinopse"], "space adventure")
        self.assertEqual(movie["palavras_chave"], ["space", "secret love"])


if __name__ == "__main__":
    unittest.main()
