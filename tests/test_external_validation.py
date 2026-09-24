import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cinebot_ml.analysis.external_validation import (
    ExternalValidationError,
    Rating,
    evaluate_external,
    fetch_archive,
    load_study_config,
    temporal_split,
    transportability,
)


CONFIG_PATH = Path("configs/experiments/external_validation_v1.json")


def synthetic_data():
    movies = {
        movie_id: {
            "movie_id": movie_id,
            "title": f"Movie {movie_id}",
            "genres": ["Drama"] if movie_id % 2 else ["Comedy"],
        }
        for movie_id in range(1, 61)
    }
    ratings = []
    for phase, count, offset, timestamp in (
        ("train", 14, 0, 100),
        ("adapt", 2, 14, 200),
        ("test", 4, 16, 300),
    ):
        for user_id in range(1, 36):
            for index in range(count):
                movie_id = ((user_id * 7 + offset + index) % 60) + 1
                rating = 5 if phase == "test" or index % 3 else 2
                ratings.append(Rating(user_id, movie_id, rating, timestamp + index))
    return ratings, movies


class ProtocolAndSplitTests(unittest.TestCase):
    def test_protocolo_esta_congelado_e_fonte_foi_pre_registrada(self):
        config = load_study_config(CONFIG_PATH)
        self.assertTrue(config["frozen"])
        self.assertTrue(config["selection_registered_before_download"])
        self.assertFalse(config["dataset"]["raw_redistribution"])

    def test_split_global_nao_tem_vazamento_temporal(self):
        config = load_study_config(CONFIG_PATH)
        train, adaptation, test, manifest = temporal_split(
            synthetic_data()[0], config
        )
        self.assertLessEqual(
            max(row.timestamp for row in train),
            min(row.timestamp for row in adaptation),
        )
        self.assertLessEqual(
            max(row.timestamp for row in adaptation),
            min(row.timestamp for row in test),
        )
        self.assertTrue(manifest["temporal_order_valid"])

    def test_checksum_invalido_bloqueia_arquivo_local(self):
        config = load_study_config(CONFIG_PATH)
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "ml-100k.zip"
            archive.write_bytes(b"invalid")
            with self.assertRaisesRegex(ExternalValidationError, "Checksum"):
                fetch_archive(config, archive)


class EvaluationTests(unittest.TestCase):
    def test_avaliacao_mantem_metodos_e_usuarios_pareados(self):
        config = load_study_config(CONFIG_PATH)
        report, rows, split = evaluate_external(
            *synthetic_data(), config
        )
        self.assertEqual(report["eligible_users"], 35)
        self.assertEqual(len(rows), 35)
        self.assertEqual(
            set(report["contrasts"]), {"B5_minus_B4", "B5_minus_B0"}
        )
        self.assertTrue(all("difference_b5_minus_b4" in row for row in rows))
        self.assertEqual(report["coverage"]["rating_mapping_coverage"], 1.0)
        self.assertEqual(split["counts"]["all"], 700)

    def test_efeito_publico_nulo_e_inconclusivo_nao_replicado(self):
        external = {
            "contrasts": {
                "B5_minus_B4": {
                    "valid_pairs": 35,
                    "mean_difference": 0.0,
                    "confidence_interval_95": [0.0, 0.0],
                    "p_value_two_sided": 1.0,
                }
            }
        }
        synthetic_report = {
            "paired_difference_b5_minus_b4": {
                "valid_pairs": 35,
                "mean_difference": -0.02,
                "confidence_interval_95": [-0.03, -0.01],
                "p_value_two_sided": 0.02,
            }
        }
        with patch(
            "cinebot_ml.analysis.external_validation.analyze_official_results",
            return_value=(synthetic_report, []),
        ):
            result = transportability(external, Path("manifest.json"))
        self.assertEqual(result["public"]["direction"], "neutral")
        self.assertEqual(result["direction"], "inconclusive_neutral_external")
        self.assertEqual(result["comparability"], "partial")


if __name__ == "__main__":
    unittest.main()
