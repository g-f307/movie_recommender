"""Verificação local do inventário distribuível e dos resultados oficiais."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from cinebot_ml.analysis.loader import load_official_records
from cinebot_ml.config import PROJECT_ROOT


DEFAULT_INVENTORY = PROJECT_ROOT / "configs/artifact_inventory_v1.json"


class ArtifactVerificationError(ValueError):
    """Indica ativo ausente ou incompatível com o inventário congelado."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pointer_md5(path: Path) -> str:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return str(payload["outs"][0]["md5"])
    except (KeyError, IndexError, TypeError) as exc:
        raise ArtifactVerificationError(f"Ponteiro DVC inválido: {path}") from exc


def verify_inventory(inventory_path: Path = DEFAULT_INVENTORY,
                     *, root: Path = PROJECT_ROOT) -> dict[str, Any]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    checks = []
    for asset in inventory["dvc_assets"]:
        target, pointer = root / asset["path"], root / asset["pointer"]
        if not target.exists() or not pointer.is_file():
            raise ArtifactVerificationError(f"Ativo ou ponteiro ausente: {asset['path']}")
        if _pointer_md5(pointer) != asset["dvc_md5"]:
            raise ArtifactVerificationError(f"Hash DVC divergente: {asset['path']}")
        if target.is_file():
            if target.stat().st_size != asset["bytes"] or _sha256(target) != asset["sha256"]:
                raise ArtifactVerificationError(f"Arquivo divergente: {asset['path']}")
        else:
            files = [path for path in target.rglob("*") if path.is_file()]
            if len(files) != asset["files"] or sum(path.stat().st_size for path in files) != asset["bytes"]:
                raise ArtifactVerificationError(f"Diretório divergente: {asset['path']}")
            if "anchor" in asset and _sha256(root / asset["anchor"]) != asset["anchor_sha256"]:
                raise ArtifactVerificationError(f"Âncora divergente: {asset['anchor']}")
        checks.append({"path": asset["path"], "status": "verified", "classification": asset["classification"]})
    official = next(asset for asset in inventory["dvc_assets"] if asset.get("anchor"))
    manifest, records = load_official_records(root / official["anchor"])
    return {
        "schema_version": "1.0", "status": "verified",
        "matrix_id": manifest["matrix_id"], "official_cells": len(records),
        "assets": checks, "excluded_assets": inventory["excluded_assets"],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = verify_inventory(args.inventory, root=args.root.resolve())
    serialized = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
