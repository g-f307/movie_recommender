import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cinebot_ml.analysis.paper_artifacts import (
    PaperArtifactError,
    build_paper_evidence,
    render_paper_artifacts,
)
from cinebot_ml.analysis.synthesis import DEFAULT_MANIFEST, SPECIALIZED_ROOT


def evidence_fixture():
    return {
        "schema_version": "1.0",
        "matrix_id": "matrix-test",
        "primary": {
            "agent_differences": [-0.03, -0.01, 0.02],
            "mean_difference": -0.006666666666666667,
            "confidence_interval_95": [-0.03, 0.02],
            "p_value": 0.5,
            "rank_biserial": -0.2,
            "n": 3,
        },
        "convergence": [
            {"condition": "C0", "b4_mean": 0.50, "b5_mean": 0.50},
            {"condition": "C1", "b4_mean": 0.51, "b5_mean": 0.48},
            {"condition": "C2", "b4_mean": 0.52, "b5_mean": 0.49},
        ],
        "ablation": [
            {"label": "Sem histórico", "estimate": 0.038, "ci95": [0.019, 0.058]},
            {"label": "Sem gênero", "estimate": -0.166, "ci95": [-0.191, -0.140]},
        ],
        "robustness": [
            {"label": "Contraditório", "estimate": -0.208, "ci95": [-0.289, -0.128]},
        ],
        "transportability": [
            {"source": "Sintético", "estimate": -0.0181, "ci95": [-0.0343, -0.0022], "n": 35},
            {"source": "MovieLens 100K", "estimate": 0.0, "ci95": [0.0, 0.0], "n": 43},
        ],
        "tables": {
            "design": [
                {"method": "B0", "description": "Popularidade", "feedback": "não"},
                {"method": "B5", "description": "Incremental", "feedback": "sim"},
            ],
            "hypotheses": [
                {"rq": "RQ1", "hypothesis": "H1", "estimate": -0.0181,
                 "ci95": [-0.0343, -0.0022], "decision": "não sustentada"},
            ],
            "contrasts": [
                {"contrast": "B5-B4", "estimate": -0.0181,
                 "ci95": [-0.0343, -0.0022], "p_value": 0.0356, "n": 35},
            ],
        },
        "sources": {
            "official_manifest": {"path": "manifest.json", "sha256": "a" * 64},
            "external_report": {"path": "external.json", "sha256": "b" * 64},
        },
    }


class PaperArtifactTests(unittest.TestCase):
    def test_renderiza_quatro_figuras_tres_tabelas_e_manifesto(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "paper"

            files = render_paper_artifacts(evidence_fixture(), output)

            relative = {path.relative_to(output).as_posix() for path in files}
            for stem in ("primary_paired", "feedback_trajectory",
                         "mechanisms_forest", "transportability"):
                self.assertIn(f"figures/{stem}.pdf", relative)
                self.assertIn(f"figures/{stem}.png", relative)
            self.assertIn("tables/design.tex", relative)
            self.assertIn("tables/hypothesis_decisions.tex", relative)
            self.assertIn("tables/contrasts.tex", relative)
            self.assertIn("paper_artifacts.manifest.json", relative)
            self.assertTrue((output / "figures/primary_paired.pdf").read_bytes().startswith(b"%PDF"))
            self.assertTrue((output / "figures/primary_paired.png").read_bytes().startswith(b"\x89PNG"))

    def test_manifesto_registra_fontes_e_hashes_das_saidas(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "paper"

            render_paper_artifacts(evidence_fixture(), output)

            manifest = json.loads((output / "paper_artifacts.manifest.json").read_text())
            self.assertEqual(manifest["matrix_id"], "matrix-test")
            self.assertEqual(manifest["sources"]["official_manifest"]["sha256"], "a" * 64)
            recorded = manifest["outputs"]["figures/primary_paired.png"]
            actual = hashlib.sha256((output / "figures/primary_paired.png").read_bytes()).hexdigest()
            self.assertEqual(recorded["sha256"], actual)
            self.assertGreater(recorded["bytes"], 1000)

    def test_nao_sobrescreve_destino_com_conteudo(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "paper"
            output.mkdir()
            (output / "existing.txt").write_text("preservar", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                render_paper_artifacts(evidence_fixture(), output)

            self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "preservar")

    def test_rejeita_evidencia_sem_identidade_ou_fonte(self):
        cases = [
            ({**evidence_fixture(), "matrix_id": ""}, "matrix_id"),
            ({**evidence_fixture(), "sources": {}}, "sources"),
        ]
        for evidence, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(PaperArtifactError, expected):
                    render_paper_artifacts(evidence, Path(temporary) / "paper")


    def test_execucoes_repetidas_produzem_conteudo_equivalente(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            render_paper_artifacts(evidence_fixture(), root / "first")
            render_paper_artifacts(evidence_fixture(), root / "second")
            first = json.loads((root / "first/paper_artifacts.manifest.json").read_text())
            second = json.loads((root / "second/paper_artifacts.manifest.json").read_text())
            self.assertEqual(first["outputs"], second["outputs"])


    def test_carrega_evidencias_congeladas_sem_divergir_das_conclusoes(self):
        evidence = build_paper_evidence(DEFAULT_MANIFEST, SPECIALIZED_ROOT)

        self.assertEqual(evidence["matrix_id"], "2942e51456add29e4c999307")
        self.assertAlmostEqual(evidence["primary"]["mean_difference"], -0.0180862012)
        self.assertEqual(len(evidence["primary"]["agent_differences"]), 35)
        self.assertEqual([row["condition"] for row in evidence["convergence"]],
                         [f"C{index}" for index in range(6)])
        self.assertEqual(len(evidence["ablation"]), 6)
        self.assertEqual(len(evidence["robustness"]), 12)
        self.assertEqual({row["source"] for row in evidence["transportability"]},
                         {"Sintético", "MovieLens 100K"})
        self.assertGreaterEqual(len(evidence["sources"]), 6)


if __name__ == "__main__":
    unittest.main()
