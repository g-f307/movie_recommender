import hashlib
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

from cinebot_ml.analysis.anonymous_artifact import (
    AnonymousArtifactError,
    build_anonymous_archive,
    build_sac_reanalysis_archive,
    collect_sac_artifact_files,
    verify_anonymous_archive,
)


class AnonymousArtifactTests(unittest.TestCase):
    def test_builds_verifiable_archive_with_relative_hash_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "evidence").mkdir()
            evidence = root / "evidence" / "result.json"
            evidence.write_text('{"effect": 0.0045}\n', encoding="utf-8")
            output = root / "anonymous-artifact.zip"

            manifest = build_anonymous_archive(
                output,
                project_root=root,
                files=(Path("evidence/result.json"),),
                generated_files={"README.md": "Anonymous reproduction instructions.\n"},
                metadata={"study": "movie-feedback", "version": "1.0"},
                forbidden_identifiers=("named-author",),
            )

            expected = hashlib.sha256(evidence.read_bytes()).hexdigest()
            self.assertEqual(manifest["files"]["evidence/result.json"]["sha256"], expected)
            self.assertEqual(manifest["metadata"]["study"], "movie-feedback")
            self.assertEqual(verify_anonymous_archive(output), manifest)
            with zipfile.ZipFile(output) as archive:
                self.assertEqual(
                    sorted(archive.namelist()),
                    ["README.md", "artifact_manifest.json", "evidence/result.json"],
                )
                serialized = archive.read("artifact_manifest.json").decode("utf-8")
                self.assertNotIn(str(root), serialized)

    def test_rejects_sensitive_content_before_writing_archive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "result.json"
            source.write_text(
                '{"author":"named-author","path":"/home/user/project"}\n',
                encoding="utf-8",
            )
            output = root / "anonymous-artifact.zip"

            with self.assertRaisesRegex(AnonymousArtifactError, "sensitive content"):
                build_anonymous_archive(
                    output,
                    project_root=root,
                    files=(Path("result.json"),),
                    generated_files={},
                    metadata={"study": "movie-feedback"},
                    forbidden_identifiers=("named-author",),
                )

            self.assertFalse(output.exists())

    def test_verification_rejects_an_archive_with_modified_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "result.json"
            source.write_text('{"effect": 0.0045}\n', encoding="utf-8")
            output = root / "anonymous-artifact.zip"
            build_anonymous_archive(
                output,
                project_root=root,
                files=(Path("result.json"),),
                generated_files={},
                metadata={"study": "movie-feedback"},
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(output, mode="a") as archive:
                    archive.writestr("result.json", json.dumps({"effect": 999}))

            with self.assertRaisesRegex(AnonymousArtifactError, "integrity mismatch"):
                verify_anonymous_archive(output)

    def test_collects_reanalysis_evidence_without_restricted_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = (
                "LICENSE",
                "requirements-ci.txt",
                "cinebot_ml/__init__.py",
                "cinebot_ml/analysis/v2_analysis.py",
                "cinebot_ml/train.py",
                "configs/experiments/execution.yaml",
                "data/filmes.json",
                "datasets/movie_preferences.csv",
                "results/v2/manifests/execution/matrix.manifest.json",
                "results/v2/manifests/execution/execution.manifest.json",
                "results/v2/manifests/execution/cells/a.json",
                "results/v2/reports/analysis/v2_analysis.json",
                "results/v2/reports/analysis/__pycache__/local.pyc",
                "results/v2/reports/validation/selection.json",
                "results/v1/derived/contrast.csv",
            )
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("anonymous\n", encoding="utf-8")

            selected = collect_sac_artifact_files(
                root,
                execution_manifest=root / "results/v2/manifests/execution/execution.manifest.json",
                analysis_dir=root / "results/v2/reports/analysis",
                selection_path=root / "results/v2/reports/validation/selection.json",
                v1_derived_dir=root / "results/v1/derived",
            )

            selected_names = {path.as_posix() for path in selected}
            self.assertIn("results/v2/manifests/execution/cells/a.json", selected_names)
            self.assertIn("results/v2/reports/analysis/v2_analysis.json", selected_names)
            self.assertIn("results/v1/derived/contrast.csv", selected_names)
            self.assertNotIn("data/filmes.json", selected_names)
            self.assertNotIn("datasets/movie_preferences.csv", selected_names)
            self.assertNotIn("cinebot_ml/train.py", selected_names)
            self.assertNotIn("LICENSE", selected_names)
            self.assertNotIn(
                "results/v2/reports/analysis/__pycache__/local.pyc", selected_names
            )

    def test_builds_sac_package_with_offline_verifier_and_reanalysis_command(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            analysis_payload = json.dumps(
                {
                    "source_manifest": str(
                        root / "results/v2/manifests/execution/execution.manifest.json"
                    ),
                    "effect": 0.0045,
                }
            )
            files = {
                "LICENSE": "license\n",
                "requirements-ci.txt": "numpy\n",
                "cinebot_ml/__init__.py": "",
                "cinebot_ml/analysis/v2_analysis.py": "",
                "configs/experiments/policy.yaml": "version: 1\n",
                "results/v2/manifests/execution/execution.manifest.json": json.dumps(
                    {
                        "status": "complete",
                        "execution": {"protocol_version": "protocol-v2.1"},
                        "matrix": {"matrix_id": "matrix-1", "total_cells": 20160},
                    }
                ),
                "results/v2/manifests/execution/matrix.manifest.json": "{}\n",
                "results/v2/manifests/execution/cells/a.json": "{}\n",
                "results/v2/reports/analysis/v2_analysis.json": analysis_payload,
                "results/v2/reports/analysis/analysis.manifest.json": json.dumps(
                    {
                        "outputs": {
                            "v2_analysis.json": {
                                "bytes": len(analysis_payload.encode("utf-8")),
                                "sha256": hashlib.sha256(
                                    analysis_payload.encode("utf-8")
                                ).hexdigest(),
                            }
                        }
                    }
                ),
                "results/v2/reports/validation/selection.json": "{}\n",
                "results/v1/derived/contrast.csv": "effect\n-0.0181\n",
            }
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            output = root / "sac-artifact.zip"

            manifest = build_sac_reanalysis_archive(
                output,
                project_root=root,
                execution_manifest=root
                / "results/v2/manifests/execution/execution.manifest.json",
                analysis_dir=root / "results/v2/reports/analysis",
                selection_path=root / "results/v2/reports/validation/selection.json",
                v1_derived_dir=root / "results/v1/derived",
            )

            self.assertEqual(manifest["metadata"]["total_cells"], 20160)
            with zipfile.ZipFile(output) as archive:
                readme = archive.read("README.md").decode("utf-8")
                self.assertIn("python verify_artifact.py", readme)
                self.assertIn("cinebot_ml.analysis.v2_analysis", readme)
                self.assertIn("reanalysis", readme.casefold())
                self.assertIn("verify_artifact.py", archive.namelist())
                self.assertNotIn("data/filmes.json", archive.namelist())
                sanitized = archive.read(
                    "results/v2/reports/analysis/v2_analysis.json"
                )
                self.assertNotIn(str(root).encode("utf-8"), sanitized)
                output_manifest = json.loads(
                    archive.read("results/v2/reports/analysis/analysis.manifest.json")
                )
                self.assertEqual(
                    output_manifest["outputs"]["v2_analysis.json"]["sha256"],
                    hashlib.sha256(sanitized).hexdigest(),
                )


if __name__ == "__main__":
    unittest.main()
