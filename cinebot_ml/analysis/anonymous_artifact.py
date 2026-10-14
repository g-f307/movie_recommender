"""Build and verify an anonymous, infrastructure-independent evidence archive."""

from __future__ import annotations

import hashlib
import argparse
import gzip
import json
import os
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping

from .submission_package import validate_anonymity


class AnonymousArtifactError(ValueError):
    """Raised when an anonymous artifact violates its integrity contract."""


_ANONYMOUS_HOST_MAX_FILE_BYTES = 8 * 1024 * 1024
_MANDATORY_FORBIDDEN_IDENTIFIERS = ("g-f307", "movie_recommender")


def _archive_path(value: str | Path) -> str:
    path = PurePosixPath(str(value).replace("\\", "/"))
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise AnonymousArtifactError(f"unsafe archive path: {value}")
    normalized = path.as_posix()
    if normalized == "artifact_manifest.json":
        raise AnonymousArtifactError("artifact_manifest.json is reserved")
    return normalized


def _digest_stream(stream) -> dict[str, int | str]:
    digest = hashlib.sha256()
    size = 0
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
        size += len(chunk)
    return {"sha256": digest.hexdigest(), "bytes": size}


def _digest_path(path: Path) -> dict[str, int | str]:
    with path.open("rb") as stream:
        return _digest_stream(stream)


def _sensitive_findings(data: bytes, forbidden_identifiers: Iterable[str]):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return []
    return validate_anonymity(text, forbidden_identifiers=forbidden_identifiers)


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def _relative_directory_files(project_root: Path, directory: Path) -> set[Path]:
    directory = directory.resolve()
    if directory != project_root and project_root not in directory.parents:
        raise AnonymousArtifactError(f"directory outside project root: {directory}")
    if not directory.is_dir():
        raise AnonymousArtifactError(f"missing artifact directory: {directory}")
    return {
        path.relative_to(project_root)
        for path in directory.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def collect_sac_artifact_files(
    project_root: Path,
    *,
    execution_manifest: Path,
    analysis_dir: Path,
    selection_path: Path,
    v1_derived_dir: Path,
) -> tuple[Path, ...]:
    """Return the explicit SAC reanalysis allowlist, excluding source datasets."""
    project_root = Path(project_root).resolve()
    execution_manifest = Path(execution_manifest).resolve()
    selection_path = Path(selection_path).resolve()
    if not execution_manifest.is_file():
        raise AnonymousArtifactError(f"missing execution manifest: {execution_manifest}")
    if not selection_path.is_file():
        raise AnonymousArtifactError(f"missing validation selection: {selection_path}")

    selected: set[Path] = set()
    selected.update(_relative_directory_files(project_root, execution_manifest.parent))
    selected.update(_relative_directory_files(project_root, Path(analysis_dir)))
    selected.update(_relative_directory_files(project_root, selection_path.parent))
    selected.update(_relative_directory_files(project_root, Path(v1_derived_dir)))

    for relative in (Path("requirements-ci.txt"),):
        if (project_root / relative).is_file():
            selected.add(relative)
    source_root = project_root / "cinebot_ml"
    if source_root.is_dir():
        excluded_modules = {"train.py", "submission_package.py", "anonymous_artifact.py"}
        selected.update(
            path.relative_to(project_root)
            for path in source_root.rglob("*.py")
            if path.name not in excluded_modules and "__pycache__" not in path.parts
        )
    config_root = project_root / "configs"
    if config_root.is_dir():
        selected.update(_relative_directory_files(project_root, config_root))
    return tuple(sorted(selected, key=lambda path: path.as_posix()))


def build_anonymous_archive(
    output_path: Path,
    *,
    project_root: Path,
    files: Iterable[Path],
    generated_files: Mapping[str, str | bytes],
    metadata: Mapping[str, object],
    forbidden_identifiers: Iterable[str] = (),
    max_member_bytes: int | None = None,
) -> dict[str, object]:
    """Create an atomic ZIP archive from an explicit allowlist of project files."""
    output_path = Path(output_path)
    project_root = Path(project_root).resolve()
    forbidden = tuple(forbidden_identifiers)
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")

    sources: dict[str, Path] = {}
    for relative in files:
        archive_name = _archive_path(relative)
        source = (project_root / relative).resolve()
        if source != project_root and project_root not in source.parents:
            raise AnonymousArtifactError(f"source outside project root: {relative}")
        if not source.is_file():
            raise AnonymousArtifactError(f"missing artifact file: {relative}")
        if archive_name in sources:
            raise AnonymousArtifactError(f"duplicate archive path: {archive_name}")
        sources[archive_name] = source

    generated: dict[str, bytes] = {}
    for name, content in generated_files.items():
        archive_name = _archive_path(name)
        if archive_name in sources or archive_name in generated:
            raise AnonymousArtifactError(f"duplicate archive path: {archive_name}")
        generated[archive_name] = content.encode("utf-8") if isinstance(content, str) else content

    findings: dict[str, list[dict[str, str]]] = {}
    for name, source in sources.items():
        matches = _sensitive_findings(source.read_bytes(), forbidden)
        if matches:
            findings[name] = matches
    for name, content in generated.items():
        matches = _sensitive_findings(content, forbidden)
        if matches:
            findings[name] = matches
    metadata_bytes = json.dumps(metadata, ensure_ascii=False, sort_keys=True).encode("utf-8")
    metadata_findings = _sensitive_findings(metadata_bytes, forbidden)
    if metadata_findings:
        findings["artifact_manifest.json:metadata"] = metadata_findings
    if findings:
        raise AnonymousArtifactError(
            "sensitive content detected: " + json.dumps(findings, ensure_ascii=False)
        )

    inventory = {name: _digest_path(source) for name, source in sources.items()}
    inventory.update(
        {
            name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            for name, content in generated.items()
        }
    )
    if max_member_bytes is not None:
        oversized = {
            name: item["bytes"]
            for name, item in inventory.items()
            if int(item["bytes"]) > max_member_bytes
        }
        if oversized:
            raise AnonymousArtifactError(
                "member size limit exceeded: "
                + json.dumps(oversized, ensure_ascii=False, sort_keys=True)
            )
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "anonymous": True,
        "metadata": dict(metadata),
        "files": dict(sorted(inventory.items())),
    }
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if max_member_bytes is not None and len(manifest_bytes) > max_member_bytes:
        raise AnonymousArtifactError(
            "member size limit exceeded: artifact_manifest.json"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with zipfile.ZipFile(temporary, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, source in sorted(sources.items()):
                with source.open("rb") as stream, archive.open(_zip_info(name), "w") as target:
                    while chunk := stream.read(1024 * 1024):
                        target.write(chunk)
            for name, content in sorted(generated.items()):
                archive.writestr(_zip_info(name), content)
            archive.writestr(_zip_info("artifact_manifest.json"), manifest_bytes)
        os.replace(temporary, output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def verify_anonymous_archive(archive_path: Path) -> dict[str, object]:
    """Validate paths, hashes, sizes, duplicates, and undeclared archive members."""
    archive_path = Path(archive_path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            names = [info.filename for info in archive.infolist()]
            if len(names) != len(set(names)):
                raise AnonymousArtifactError("integrity mismatch: duplicate archive member")
            for name in names:
                _archive_path(name) if name != "artifact_manifest.json" else None
            manifest = json.loads(archive.read("artifact_manifest.json"))
            inventory = manifest.get("files")
            if not isinstance(inventory, dict):
                raise AnonymousArtifactError("integrity mismatch: invalid manifest inventory")
            expected_names = set(inventory) | {"artifact_manifest.json"}
            if set(names) != expected_names:
                raise AnonymousArtifactError("integrity mismatch: undeclared or missing member")
            for name, expected in inventory.items():
                with archive.open(name) as stream:
                    observed = _digest_stream(stream)
                if observed != expected:
                    raise AnonymousArtifactError(f"integrity mismatch: {name}")
    except (KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise AnonymousArtifactError(f"integrity mismatch: {exc}") from exc
    return manifest


_OFFLINE_VERIFIER = '''"""Verify the extracted anonymous artifact using only the standard library."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
failures = []
for relative, expected in manifest["files"].items():
    path = root / relative
    if not path.is_file():
        failures.append(f"missing: {relative}")
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != expected["bytes"] or digest != expected["sha256"]:
        failures.append(f"mismatch: {relative}")
if failures:
    raise SystemExit("Artifact verification failed:\\n" + "\\n".join(failures))
print(f"Verified {len(manifest['files'])} files.")
'''


_ANONYMOUS_REVIEW_LICENSE = """# Anonymous Peer-Review License

Copyright (c) 2026 The Authors. All rights reserved.

The authors authorize the temporary hosting of this anonymized artifact by a
peer-review service and permit reviewers and conference organizers to access,
download, execute, and reproduce it locally solely to evaluate the associated
manuscript.

This permission is free of charge and non-exclusive.
Its sole purpose is anonymous peer review. It does not permit commercial use or
redistribution outside the review process. Publication of derivative works and
removal of applicable notices are also prohibited. Third-party components and
external datasets remain subject to their respective terms.
"""


def _artifact_readme(manifest_path: Path, selection_path: Path) -> str:
    manifest = manifest_path.as_posix()
    selection = selection_path.as_posix()
    return f"""# Anonymous SAC 2027 Reanalysis Artifact

This archive supports integrity verification and independent reanalysis of the
reported v2.1 cell-level evidence. It does not redistribute the restricted
source catalog and does not claim a from-scratch rerun of the simulator.

No private object store, credentials, or environment file is required.
Temporary hosting and access for anonymous peer review are authorized under
`LICENSE-REVIEW.md`. Other redistribution rights are not granted.

## Verify integrity

```bash
python verify_artifact.py
```

## Install analysis dependencies

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-ci.txt
```

## Reproduce the v2.1 statistical analysis and figures

```bash
.venv/bin/python -m cinebot_ml.analysis.v2_analysis \\
  --manifest {manifest} \\
  --selection {selection} \\
  --output-dir reproduced/v2_analysis
```

The `results/derived/specialized_v1` directory contains the aggregate evidence
used for the diagnostic v1.1 ablation, robustness, discovery, and external
validation results. Those files are provided for inspection and traceability;
the restricted catalog and large derived training tables are intentionally
excluded.
"""


def _replace_project_paths(value, project_root: Path):
    if isinstance(value, dict):
        return {
            key: _replace_project_paths(item, project_root)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_replace_project_paths(item, project_root) for item in value]
    if isinstance(value, str):
        prefix = str(project_root) + os.sep
        if value.startswith(prefix):
            return Path(value).relative_to(project_root).as_posix()
    return value


def _sanitized_json_overrides(
    project_root: Path, files: Iterable[Path]
) -> dict[str, bytes]:
    """Sanitize workstation paths and keep adjacent hash manifests coherent."""
    relative_files = tuple(files)
    overrides: dict[str, bytes] = {}
    payloads: dict[str, object] = {}
    for relative in relative_files:
        if relative.suffix != ".json":
            continue
        try:
            original = json.loads((project_root / relative).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        sanitized = _replace_project_paths(original, project_root)
        payloads[relative.as_posix()] = sanitized
        if sanitized != original:
            overrides[relative.as_posix()] = (
                json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")

    for relative in relative_files:
        name = relative.as_posix()
        if not name.endswith("manifest.json") or name not in payloads:
            continue
        payload = payloads[name]
        if not isinstance(payload, dict):
            continue
        changed = name in overrides
        for field in ("files", "outputs"):
            inventory = payload.get(field)
            if not isinstance(inventory, dict):
                continue
            for child_name, expected in inventory.items():
                target = (relative.parent / child_name).as_posix()
                content = overrides.get(target)
                if content is None:
                    continue
                digest = hashlib.sha256(content).hexdigest()
                if isinstance(expected, str):
                    inventory[child_name] = digest
                elif isinstance(expected, dict):
                    expected["sha256"] = digest
                    expected["bytes"] = len(content)
                changed = True
        if changed:
            overrides[name] = (
                json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
    return overrides


def build_sac_reanalysis_archive(
    output_path: Path,
    *,
    project_root: Path,
    execution_manifest: Path,
    analysis_dir: Path,
    selection_path: Path,
    v1_derived_dir: Path,
    forbidden_identifiers: Iterable[str] = (),
) -> dict[str, object]:
    """Build the SAC review artifact from frozen cell-level and derived evidence."""
    project_root = Path(project_root).resolve()
    execution_manifest = Path(execution_manifest).resolve()
    selection_path = Path(selection_path).resolve()
    files = collect_sac_artifact_files(
        project_root,
        execution_manifest=execution_manifest,
        analysis_dir=analysis_dir,
        selection_path=selection_path,
        v1_derived_dir=v1_derived_dir,
    )
    try:
        execution = json.loads(execution_manifest.read_text(encoding="utf-8"))
        matrix = execution["matrix"]
        protocol = execution["execution"]["protocol_version"]
        if execution["status"] != "complete":
            raise KeyError("status")
        matrix_id = str(matrix["matrix_id"])
        total_cells = int(matrix["total_cells"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise AnonymousArtifactError("invalid complete execution manifest") from exc
    manifest_relative = execution_manifest.relative_to(project_root)
    selection_relative = selection_path.relative_to(project_root)
    generated_files = _sanitized_json_overrides(project_root, files)
    matrix_relative = execution_manifest.parent.relative_to(project_root) / "matrix.manifest.json"
    matrix_name = matrix_relative.as_posix()
    matrix_content = generated_files.pop(
        matrix_name, (project_root / matrix_relative).read_bytes()
    )
    replaced_sources = set(generated_files) | {matrix_name}
    files = tuple(path for path in files if path.as_posix() not in replaced_sources)
    generated_files[f"{matrix_name}.gz"] = gzip.compress(matrix_content, mtime=0)
    generated_files.update(
        {
            "README.md": _artifact_readme(manifest_relative, selection_relative),
            "LICENSE-REVIEW.md": _ANONYMOUS_REVIEW_LICENSE,
            "verify_artifact.py": _OFFLINE_VERIFIER,
        }
    )
    return build_anonymous_archive(
        output_path,
        project_root=project_root,
        files=files,
        generated_files=generated_files,
        metadata={
            "event": "ACM SAC 2027",
            "scope": "cell-level verification and statistical reanalysis",
            "protocol": protocol,
            "matrix_id": matrix_id,
            "total_cells": total_cells,
            "requires_private_object_store": False,
        },
        forbidden_identifiers=(
            *_MANDATORY_FORBIDDEN_IDENTIFIERS,
            *tuple(forbidden_identifiers),
        ),
        max_member_bytes=_ANONYMOUS_HOST_MAX_FILE_BYTES,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--v1-derived-dir", type=Path, required=True)
    parser.add_argument("--forbidden-identifier", action="append", default=[])
    args = parser.parse_args(list(argv) if argv is not None else None)
    manifest = build_sac_reanalysis_archive(
        args.output,
        project_root=Path.cwd(),
        execution_manifest=args.execution_manifest,
        analysis_dir=args.analysis_dir,
        selection_path=args.selection,
        v1_derived_dir=args.v1_derived_dir,
        forbidden_identifiers=args.forbidden_identifier,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "file_count": len(manifest["files"]),
                "uncompressed_bytes": sum(
                    int(item["bytes"]) for item in manifest["files"].values()
                ),
                "matrix_id": manifest["metadata"]["matrix_id"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
