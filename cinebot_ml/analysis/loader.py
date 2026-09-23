"""Carga estrita de uma execução oficial selada."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class OfficialResultsError(ValueError):
    """Indica que uma execução não satisfaz o contrato oficial congelado."""


@dataclass(frozen=True)
class MetricRecord:
    cell_id: str
    comparison_id: str
    method: str
    condition: str
    profile: str
    persona: str
    seed: int
    run: int
    k: int
    agent_id: str
    status: str
    metric: float | None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise OfficialResultsError(f"JSON oficial inválido em {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise OfficialResultsError(f"Objeto JSON esperado em {path}.")
    return value


def load_official_records(
    manifest_path: Path, *, metric: str = "ndcg_at_k"
) -> tuple[dict[str, Any], list[MetricRecord]]:
    """Valida hashes, identidade e completude antes de carregar as células."""
    manifest_path = manifest_path.resolve()
    manifest = _read_object(manifest_path)
    if manifest.get("status") != "complete" or manifest.get("evidence") != "exploratory_synthetic":
        raise OfficialResultsError("Manifesto não representa evidência sintética oficial completa.")
    if not isinstance(manifest.get("matrix_id"), str) or manifest_path.parent.name != manifest["matrix_id"]:
        raise OfficialResultsError("Diretório e matrix_id do manifesto divergem.")
    if manifest.get("official", {}).get("completed") != manifest.get("total_cells"):
        raise OfficialResultsError("Contagem oficial incompleta no manifesto.")
    dimensions = manifest.get("dimensions", {})
    if not {"B4", "B5"}.issubset(set(dimensions.get("method", []))) or 5 not in dimensions.get("k", []):
        raise OfficialResultsError("Manifesto não contém B4, B5 e K=5.")

    matrix_path = manifest_path.parent / "matrix.manifest.json"
    if not matrix_path.is_file() or _sha256(matrix_path) != manifest.get("matrix_manifest_sha256"):
        raise OfficialResultsError("Hash do manifesto da matriz diverge da execução congelada.")
    matrix = _read_object(matrix_path)
    if matrix.get("matrix_id") != manifest["matrix_id"] or matrix.get("total_cells") != manifest["total_cells"]:
        raise OfficialResultsError("Identidade ou tamanho da matriz incompatível.")

    expected_hashes = manifest.get("cells_sha256")
    if not isinstance(expected_hashes, dict) or len(expected_hashes) != manifest["total_cells"]:
        raise OfficialResultsError("Índice de hashes das células ausente ou incompleto.")
    cells_dir = manifest_path.parent / "cells"
    actual_ids = {path.stem for path in cells_dir.glob("*.json")}
    expected_ids = set(expected_hashes)
    if actual_ids != expected_ids:
        raise OfficialResultsError("Células presentes não correspondem exatamente ao manifesto oficial.")

    records: list[MetricRecord] = []
    for cell_id in sorted(expected_ids):
        path = cells_dir / f"{cell_id}.json"
        if _sha256(path) != expected_hashes[cell_id]:
            raise OfficialResultsError(f"Hash divergente para a célula {cell_id}.")
        payload = _read_object(path)
        cell = payload.get("cell", {})
        individual = payload.get("result", {}).get("individual", {})
        if payload.get("status") != "completed" or cell.get("cell_id") != cell_id:
            raise OfficialResultsError(f"Célula {cell_id} não está completa ou possui identidade divergente.")
        fields = ("method", "condition", "profile", "persona", "seed", "k")
        if any(individual.get(field) != cell.get(field) for field in fields):
            raise OfficialResultsError(f"Metadados divergentes na célula {cell_id}.")
        raw_metric = individual.get("metrics", {}).get(metric)
        records.append(MetricRecord(
            cell_id=cell_id,
            comparison_id=str(cell["comparison_id"]),
            method=str(cell["method"]),
            condition=str(cell["condition"]),
            profile=str(cell["profile"]),
            persona=str(cell["persona"]),
            seed=int(cell["seed"]),
            run=int(cell["run"]),
            k=int(cell["k"]),
            agent_id=str(individual["agent_id"]),
            status=str(individual.get("evaluation_status")),
            metric=float(raw_metric) if isinstance(raw_metric, (int, float)) else None,
        ))
    return manifest, records
