import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from cinebot_ml.analysis.submission_package import (
    SubmissionPackageError,
    build_submission_package,
    validate_anonymity,
    validate_pdf_contract,
)


class SubmissionPackageTests(unittest.TestCase):
    def test_accepts_anonymous_letter_pdf_within_page_limit(self):
        findings = validate_pdf_contract(
            page_count=5,
            page_width_points=612.0,
            page_height_points=792.0,
            metadata={"Title": "Anonymous study", "Author": ""},
        )

        self.assertEqual(findings, [])

    def test_rejects_pdf_outside_page_limit_or_letter_size(self):
        findings = validate_pdf_contract(
            page_count=9,
            page_width_points=595.0,
            page_height_points=842.0,
            metadata={"Author": "Anonymous"},
        )

        self.assertIn("page count 9 exceeds the 8-page limit", findings)
        self.assertIn("page size must be US Letter (612 x 792 pt)", findings)

    def test_detects_identity_credentials_private_endpoints_and_local_paths(self):
        findings = validate_anonymity(
            "Author: Gabriel <author@example.org>\n"
            "artifact: /home/gf307/project\n"
            "endpoint: https://account.r2.cloudflarestorage.com\n"
            "token: AWS_SECRET_ACCESS_KEY=secret\n",
            forbidden_identifiers=("Gabriel", "gf307"),
        )

        categories = {finding["category"] for finding in findings}
        self.assertEqual(
            categories,
            {"author_identifier", "email", "local_path", "private_endpoint", "credential"},
        )

    def test_builds_minimal_package_with_relative_inventory_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "paper.tex").write_text("anonymous source\n", encoding="utf-8")
            (source / "refs.bib").write_text("@book{x}\n", encoding="utf-8")
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"%PDF-1.7\nanonymous\n")
            output = root / "package"

            manifest = build_submission_package(
                output,
                pdf_path=pdf,
                source_files=(source / "paper.tex", source / "refs.bib"),
                project_root=root,
                commit="a" * 40,
                checked_at="2026-10-01",
            )

            self.assertEqual(
                sorted(manifest["files"]),
                ["source/paper.tex", "source/refs.bib", "submission/movie_recommender_anonymous.pdf"],
            )
            expected = hashlib.sha256(b"%PDF-1.7\nanonymous\n").hexdigest()
            self.assertEqual(
                manifest["files"]["submission/movie_recommender_anonymous.pdf"]["sha256"],
                expected,
            )
            serialized = (output / "submission_manifest.json").read_text(encoding="utf-8")
            self.assertNotIn(str(root), serialized)
            self.assertEqual(json.loads(serialized), manifest)

    def test_refuses_to_overwrite_existing_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "package"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"%PDF")

            with self.assertRaises(FileExistsError):
                build_submission_package(
                    output,
                    pdf_path=pdf,
                    source_files=(),
                    project_root=root,
                    commit="b" * 40,
                    checked_at="2026-10-01",
                )

            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "keep")

    def test_rejects_source_outside_project_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            root.mkdir()
            outside = Path(temporary) / "secret.txt"
            outside.write_text("secret", encoding="utf-8")
            pdf = root / "paper.pdf"
            pdf.write_bytes(b"%PDF")

            with self.assertRaisesRegex(SubmissionPackageError, "outside project root"):
                build_submission_package(
                    root / "package",
                    pdf_path=pdf,
                    source_files=(outside,),
                    project_root=root,
                    commit="c" * 40,
                    checked_at="2026-10-01",
                )


if __name__ == "__main__":
    unittest.main()
