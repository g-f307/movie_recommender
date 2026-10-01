"""Validate and assemble the anonymous ACM SAC submission package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable, Mapping

from cinebot_ml.config import PROJECT_ROOT


class SubmissionPackageError(ValueError):
    """Raised when a submission artifact violates the package contract."""


_SENSITIVE_PATTERNS = (
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("local_path", re.compile(r"(?:/home/|/Users/|[A-Z]:\\Users\\)")),
    ("private_endpoint", re.compile(r"https?://[^\s}]*?(?:r2\.cloudflarestorage\.com|localhost|127\.0\.0\.1)", re.IGNORECASE)),
    ("credential", re.compile(r"\b(?:AWS_SECRET_ACCESS_KEY|AWS_ACCESS_KEY_ID|R2_SECRET_ACCESS_KEY|TOKEN)\s*[:=]", re.IGNORECASE)),
)


def _digest(path: Path) -> dict[str, int | str]:
    data = path.read_bytes()
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def validate_anonymity(
    text: str,
    *,
    forbidden_identifiers: Iterable[str] = (),
) -> list[dict[str, str]]:
    """Return categorized findings that can identify authors or infrastructure."""
    findings: list[dict[str, str]] = []
    lowered = text.casefold()
    for identifier in forbidden_identifiers:
        normalized = identifier.strip()
        if normalized and normalized.casefold() in lowered:
            findings.append({"category": "author_identifier", "match": normalized})
    for category, pattern in _SENSITIVE_PATTERNS:
        for match in pattern.finditer(text):
            findings.append({"category": category, "match": match.group(0)})
    return findings


def validate_pdf_contract(
    *,
    page_count: int,
    page_width_points: float,
    page_height_points: float,
    metadata: Mapping[str, str],
    page_limit: int = 8,
) -> list[str]:
    """Validate page count, US Letter dimensions, and identifying metadata."""
    findings: list[str] = []
    if page_count < 1:
        findings.append("PDF must contain at least one page")
    if page_count > page_limit:
        findings.append(f"page count {page_count} exceeds the {page_limit}-page limit")
    if abs(page_width_points - 612.0) > 1.0 or abs(page_height_points - 792.0) > 1.0:
        findings.append("page size must be US Letter (612 x 792 pt)")
    author = str(metadata.get("Author", "")).strip()
    if author and author.casefold() not in {"anonymous", "anonymous author(s)"}:
        findings.append("PDF Author metadata must be empty or anonymous")
    return findings


def inspect_pdf(pdf_path: Path, *, page_limit: int = 8) -> dict[str, object]:
    """Inspect a real PDF through Poppler and return its validation report."""
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        raise SubmissionPackageError(f"missing PDF: {pdf_path}")
    info = subprocess.run(
        ["pdfinfo", str(pdf_path)], check=True, capture_output=True, text=True
    ).stdout
    metadata: dict[str, str] = {}
    for line in info.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
    size_match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", info)
    if not size_match or "Pages" not in metadata:
        raise SubmissionPackageError("pdfinfo did not report page count and dimensions")
    text = subprocess.run(
        ["pdftotext", str(pdf_path), "-"], check=True, capture_output=True, text=True
    ).stdout
    contract = validate_pdf_contract(
        page_count=int(metadata["Pages"]),
        page_width_points=float(size_match.group(1)),
        page_height_points=float(size_match.group(2)),
        metadata=metadata,
        page_limit=page_limit,
    )
    return {
        "page_count": int(metadata["Pages"]),
        "page_size_points": [float(size_match.group(1)), float(size_match.group(2))],
        "metadata": {key: metadata.get(key, "") for key in ("Title", "Author", "Creator", "Producer")},
        "contract_findings": contract,
        "extracted_text": text,
    }


def build_submission_package(
    output_dir: Path,
    *,
    pdf_path: Path,
    source_files: Iterable[Path],
    project_root: Path,
    commit: str,
    checked_at: str,
    template_version: str,
) -> dict[str, object]:
    """Build a minimal package atomically and record a relative hash inventory."""
    output_dir = Path(output_dir)
    project_root = Path(project_root).resolve()
    pdf_path = Path(pdf_path).resolve()
    sources = tuple(Path(path).resolve() for path in source_files)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    if not pdf_path.is_file():
        raise SubmissionPackageError(f"missing PDF: {pdf_path}")
    for source in sources:
        if not source.is_file():
            raise SubmissionPackageError(f"missing source: {source}")
        if not source.is_relative_to(project_root):
            raise SubmissionPackageError(f"source outside project root: {source}")

    source_base = Path(os.path.commonpath([str(path.parent) for path in sources])) if sources else project_root
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "submission-package"
        submission = stage / "submission"
        source_root = stage / "source"
        submission.mkdir(parents=True)
        source_root.mkdir()
        shutil.copy2(pdf_path, submission / "movie_recommender_anonymous.pdf")
        destinations: set[Path] = set()
        for source in sources:
            relative = source.relative_to(source_base)
            destination = source_root / relative
            if destination in destinations:
                raise SubmissionPackageError(f"duplicate package destination: {relative}")
            destinations.add(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        files = {
            path.relative_to(stage).as_posix(): _digest(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest: dict[str, object] = {
            "schema_version": "1.0",
            "event": "ACM SAC 2027",
            "track": "SEAI",
            "anonymous": True,
            "checked_at": checked_at,
            "git_commit": commit,
            "template_version": template_version,
            "files": files,
        }
        (stage / "submission_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            output_dir.rmdir()
        shutil.move(str(stage), output_dir)
    return manifest


def _git_commit(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checked-at", required=True)
    parser.add_argument("--forbidden-identifier", action="append", default=[])
    args = parser.parse_args(argv)

    report = inspect_pdf(args.pdf)
    source_text = (PROJECT_ROOT / "reports/ACM_SAC_2027_Article_Template/movie_recommender_draft.tex").read_text(encoding="utf-8")
    anonymity = validate_anonymity(
        source_text + "\n" + str(report["extracted_text"]),
        forbidden_identifiers=args.forbidden_identifier,
    )
    if report["contract_findings"] or anonymity:
        raise SubmissionPackageError(
            json.dumps({"pdf": report["contract_findings"], "anonymity": anonymity}, ensure_ascii=False)
        )

    template = PROJECT_ROOT / "reports/ACM_SAC_2027_Article_Template"
    source_files = [
        template / "movie_recommender_draft.tex",
        template / "movie_recommender_references.bib",
        template / "acmart.cls",
        template / "acmdatamodel.dbx",
        template / "acmnumeric.bbx",
        template / "acmnumeric.cbx",
        *sorted((template / "paper_artifacts/figures").glob("*.pdf")),
        *sorted((template / "paper_artifacts/tables").glob("*.tex")),
    ]
    class_match = re.search(
        r"\\ProvidesClass\{acmart\}\s*\[([^\]]+)\]",
        (template / "acmart.cls").read_text(encoding="utf-8", errors="replace"),
    )
    template_version = class_match.group(1).strip() if class_match else "acmart version not detected"
    manifest = build_submission_package(
        args.output,
        pdf_path=args.pdf,
        source_files=source_files,
        project_root=PROJECT_ROOT,
        commit=_git_commit(PROJECT_ROOT),
        checked_at=args.checked_at,
        template_version=template_version,
    )
    print(json.dumps({"output": str(args.output), "manifest": manifest}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
