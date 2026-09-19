import unittest

from cinebot_ml.experiments.matrix import ExperimentCell
from cinebot_ml.experiments.units import _unit


def catalog():
    return [{"id": index, "titulo": f"Filme {index}", "perfil": "acao" if index % 2 else "drama",
             "genero": "Ação" if index % 2 else "Drama", "generos_secundarios": [],
             "ano": "2020", "sinopse": "Uma aventura", "palavras_chave": ["aventura"],
             "diretor": "Diretor", "nota": 7.0, "votos": 100, "streaming": []}
            for index in range(1, 25)]


class UnitGenerationTests(unittest.TestCase):
    def test_historico_neutro_e_relevancia_completa(self):
        cell = ExperimentCell("B0", "C4", "P5", "consistent", 42, 5)
        first = _unit(cell, catalog(), "a" * 64)
        second = _unit(cell, catalog(), "a" * 64)
        self.assertEqual(first, second)
        unit = first["units"][0]
        self.assertEqual(first["relevance_source"], "synthetic_user")
        self.assertEqual(unit["state_version"], 10)
        self.assertEqual(len(unit["request"]["history"]), 10)
        self.assertEqual(len(unit["relevance"]), 14)
        self.assertLess(unit["state_snapshot"]["state"]["logical_timestamp"],
                        unit["request"]["logical_timestamp"])


if __name__ == "__main__":
    unittest.main()
