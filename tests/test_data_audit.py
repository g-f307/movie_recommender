import csv
import json
import tempfile
import unittest
from pathlib import Path

from cinebot_ml.data_audit import (
    DATASET_COLUMNS,
    FEEDBACK_COLUMNS,
    AuditValidationError,
    audit_catalog,
    audit_dataset,
    audit_feedback,
    render_data_dictionary,
    render_markdown,
    run_audit,
)


def movie(movie_id=1, **overrides):
    data = {
        "id": movie_id,
        "titulo": f"Filme {movie_id}",
        "sinopse": "Uma sinopse de teste.",
        "diretor": "Diretora Teste",
        "ano": 2020,
        "nota": 8.0,
        "votos": 1000,
        "duracao": 120,
        "streaming": ["Teste Play"],
        "poster": "https://example.invalid/poster.jpg",
        "palavras_chave": ["teste"],
    }
    data.update(overrides)
    return data


def dataset_row(movie_id=1, label_source="heuristic_proxy", relevant="1", **overrides):
    row = {column: "1" for column in DATASET_COLUMNS}
    row.update(
        {
            "movie_id": str(movie_id),
            "titulo": f"Filme {movie_id}",
            "sinopse": "Texto",
            "genero": "acao",
            "generos_texto": "Ação Drama",
            "diretor": "Diretora Teste",
            "release_period": "moderno",
            "poster": "https://example.invalid/poster.jpg",
            "pref_1": "acao",
            "pref_2": "drama",
            "pref_3": "comedia",
            "decade_pref": "moderno",
            "popularity_pref": "popular",
            "relevante": relevant,
            "label_source": label_source,
            "feedback_scope": "none" if label_source == "heuristic_proxy" else "exact_context_feedback",
        }
    )
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def feedback_row(user_id="sensitive-user-123", movie_id=1, feedback="like", label="1", **overrides):
    row = {
        "user_id": user_id,
        "movie_id": str(movie_id),
        "pref_1": "acao",
        "pref_2": "drama",
        "pref_3": "comedia",
        "decade_pref": "moderno",
        "popularity_pref": "popular",
        "feedback": feedback,
        "label": label,
        "timestamp": "2026-09-09T00:00:00Z",
    }
    row.update({key: str(value) for key, value in overrides.items()})
    return row


class DataAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.catalog = self.root / "filmes.json"
        self.dataset = self.root / "movie_preferences.csv"
        self.feedback = self.root / "user_feedback.csv"

    def tearDown(self):
        self.temporary.cleanup()

    def write_catalog(self, profiles):
        self.catalog.write_text(json.dumps({"perfis": profiles}, ensure_ascii=False), encoding="utf-8")

    def write_csv(self, path, columns, rows):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def write_valid_inputs(self):
        self.write_catalog({"acao": [movie(1)], "drama": [movie(2, ano=2005, votos=50)]})
        self.write_csv(
            self.dataset,
            DATASET_COLUMNS,
            [dataset_row(1, relevant="1"), dataset_row(2, relevant="0")],
        )
        self.write_csv(self.feedback, FEEDBACK_COLUMNS, [feedback_row()])

    def test_catalogo_valido(self):
        self.write_catalog({"acao": [movie(1)], "drama": [movie(2)]})
        result = audit_catalog(self.catalog)
        self.assertEqual(result["total_occurrences"], 2)
        self.assertEqual(result["unique_movies"], 2)

    def test_catalogo_vazio(self):
        self.write_catalog({})
        result = audit_catalog(self.catalog)
        self.assertEqual(result["unique_movies"], 0)
        self.assertEqual(result["coverage"]["sinopse"]["rate"], 0.0)

    def test_filme_repetido_em_multiplos_perfis(self):
        repeated = movie(7)
        self.write_catalog({"acao": [repeated], "drama": [repeated]})
        result = audit_catalog(self.catalog)
        self.assertEqual(result["total_occurrences"], 2)
        self.assertEqual(result["unique_movies"], 1)
        self.assertEqual(result["duplicates"]["movies_in_multiple_profiles"], 1)
        self.assertEqual(result["duplicates"]["duplicate_occurrences"], 1)

    def test_campos_textuais_e_numericos_ausentes(self):
        self.write_catalog({"acao": [movie(1, sinopse="", diretor=None, ano=None, votos=None)]})
        result = audit_catalog(self.catalog)
        self.assertEqual(result["missing"]["sinopse"]["count"], 1)
        self.assertEqual(result["missing"]["diretor"]["count"], 1)
        self.assertEqual(result["missing"]["ano"]["count"], 1)
        self.assertEqual(result["decade_distribution"]["unknown"], 1)

    def test_dataset_somente_com_rotulos_heuristicos(self):
        self.write_csv(self.dataset, DATASET_COLUMNS, [dataset_row(1), dataset_row(2, relevant="0")])
        result = audit_dataset(self.dataset)
        self.assertEqual(result["distributions"]["label_source"], {"heuristic_proxy": 2})

    def test_dataset_com_feedback_real(self):
        rows = [dataset_row(1), dataset_row(2, label_source="feedback_exact_context")]
        self.write_csv(self.dataset, DATASET_COLUMNS, rows)
        result = audit_dataset(self.dataset)
        self.assertEqual(result["distributions"]["label_source"]["feedback_exact_context"], 1)
        self.assertEqual(result["distributions"]["label_source"]["heuristic_proxy"], 1)

    def test_feedback_vazio(self):
        self.write_csv(self.feedback, FEEDBACK_COLUMNS, [])
        result = audit_feedback(self.feedback)
        self.assertEqual(result["events"], 0)
        self.assertEqual(result["unique_users"], 0)

    def test_colunas_obrigatorias_ausentes(self):
        self.write_csv(self.dataset, ["movie_id", "titulo"], [{"movie_id": "1", "titulo": "Teste"}])
        with self.assertRaises(AuditValidationError):
            audit_dataset(self.dataset)

    def test_determinismo_e_fontes_preservadas(self):
        self.write_valid_inputs()
        before = {path: path.read_bytes() for path in (self.catalog, self.dataset, self.feedback)}
        first = run_audit(self.catalog, self.dataset, self.feedback)
        second = run_audit(self.catalog, self.dataset, self.feedback)
        self.assertEqual(first, second)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_relatorio_nao_exporta_identificador_do_usuario(self):
        self.write_valid_inputs()
        report = run_audit(self.catalog, self.dataset, self.feedback)
        serialized = json.dumps(report, ensure_ascii=False)
        markdown = render_markdown(report)
        self.assertNotIn("sensitive-user-123", serialized)
        self.assertNotIn("sensitive-user-123", markdown)
        self.assertFalse(report["feedback"]["privacy"]["user_identifiers_exported"])

    def test_amostra_limita_dataset_e_feedback(self):
        self.write_catalog({"acao": [movie(1)]})
        self.write_csv(self.dataset, DATASET_COLUMNS, [dataset_row(i) for i in range(1, 5)])
        self.write_csv(self.feedback, FEEDBACK_COLUMNS, [feedback_row(movie_id=i) for i in range(1, 5)])
        report = run_audit(self.catalog, self.dataset, self.feedback, sample_rows=2)
        self.assertEqual(report["dataset"]["rows_audited"], 2)
        self.assertEqual(report["feedback"]["events"], 2)
        self.assertEqual(report["mode"], "sample")

    def test_dicionario_descreve_todas_as_colunas(self):
        dictionary = render_data_dictionary()
        for column in DATASET_COLUMNS:
            self.assertIn(f"`{column}`", dictionary)


if __name__ == "__main__":
    unittest.main()
