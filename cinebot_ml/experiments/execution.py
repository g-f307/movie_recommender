"""Contrato versionado para isolar execuções experimentais."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT


DEFAULT_V2_EXECUTION_SPEC_PATH = PROJECT_ROOT / "configs/experiments/execution_v2.yaml"
DEFAULT_V2_EXECUTION_SCHEMA_PATH = PROJECT_ROOT / "configs/experiments/execution_v2.schema.json"
OUTPUT_KEYS = ("split_manifest", "catalog", "units", "raw", "manifests", "reports")
PROTECTED_V1_PATHS = (
    "results/derived/test_catalog.json",
    "results/manifests/splits",
    "results/manifests/executions",
    "results/units_v1_1",
    "results/raw/pilot",
    "results/raw/official",
    "results/raw/official_v1_1",
)


class ExecutionSpecError(ValueError):
    """Especificação de execução incompatível com o contrato de isolamento."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionSpecError(f"paths.{field} deve ser um caminho relativo não vazio.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ExecutionSpecError(f"paths.{field} deve permanecer relativo à raiz do projeto.")
    return path.as_posix()


def _overlaps(left: str, right: str) -> bool:
    left_path, right_path = PurePosixPath(left), PurePosixPath(right)
    return left_path == right_path or left_path in right_path.parents or right_path in left_path.parents


@dataclass(frozen=True)
class ExecutionSpec:
    """Especificação imutável e auditável de uma família de artefatos."""

    execution_id: str
    version: str
    protocol_version: str
    exposure_policy: str
    paths: Mapping[str, str]
    profile_observation_config: str | None
    closed_loop_config: str | None
    sha256: str
    source_path: Path
    project_root: Path

    @property
    def output_paths(self) -> dict[str, str]:
        return {key: self.paths[key] for key in OUTPUT_KEYS}

    def resolve_path(self, key: str, project_root: Path | None = None) -> Path:
        if key == "profile_observation_config" and self.profile_observation_config is not None:
            relative = self.profile_observation_config
        elif key == "closed_loop_config" and self.closed_loop_config is not None:
            relative = self.closed_loop_config
        else:
            try:
                relative = self.paths[key]
            except KeyError as exc:
                raise ExecutionSpecError(f"Caminho desconhecido na execução: {key}") from exc
        root = (project_root or self.project_root).resolve()
        resolved = (root / relative).resolve()
        if resolved != root and root not in resolved.parents:
            raise ExecutionSpecError(f"Caminho escapa da raiz do projeto: {relative}")
        return resolved


def validate_execution_spec(payload: Mapping[str, Any], schema_path: Path = DEFAULT_V2_EXECUTION_SCHEMA_PATH) -> None:
    if not isinstance(payload, Mapping):
        raise ExecutionSpecError("A especificação de execução deve ser um objeto.")
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionSpecError(f"Schema da execução indisponível: {exc}") from exc
    Draft202012Validator.check_schema(schema)
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        location = ".".join(str(part) for part in error.absolute_path) or "configuração"
        raise ExecutionSpecError(f"Especificação inválida em {location}: {error.message}")

    execution = payload["execution"]
    if not all(isinstance(execution[key], str) and execution[key].strip()
               for key in ("id", "version", "protocol_version", "exposure_policy")):
        raise ExecutionSpecError("Identidade, versão, protocolo e política devem ser textos não vazios.")
    paths = {key: _relative_path(value, key) for key, value in payload["paths"].items()}
    profile_observation_config = payload.get("profile_observation_config")
    if profile_observation_config is not None:
        _relative_path(profile_observation_config, "profile_observation_config")
    closed_loop_config = payload.get("closed_loop_config")
    if closed_loop_config is not None:
        _relative_path(closed_loop_config, "closed_loop_config")
    outputs = [paths[key] for key in OUTPUT_KEYS]
    if len(outputs) != len(set(outputs)):
        raise ExecutionSpecError("Os caminhos de saída da execução devem ser distintos.")
    for output in outputs:
        protected = next((path for path in PROTECTED_V1_PATHS if _overlaps(output, path)), None)
        if protected:
            raise ExecutionSpecError(
                f"A execução v2 não pode reutilizar o caminho protegido da v1.1: {protected}."
            )
        if not output.startswith("results/v2/"):
            raise ExecutionSpecError("As saídas v2 devem permanecer sob results/v2/.")


def load_execution_spec(
    path: Path = DEFAULT_V2_EXECUTION_SPEC_PATH,
    *,
    project_root: Path = PROJECT_ROOT,
    schema_path: Path = DEFAULT_V2_EXECUTION_SCHEMA_PATH,
) -> ExecutionSpec:
    """Carrega uma execução congelada e verifica o lock antes de qualquer escrita."""
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ExecutionSpecError(f"Não foi possível carregar a execução: {exc}") from exc
    validate_execution_spec(payload, schema_path=schema_path)
    lock_path = path.with_suffix(".lock.json")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionSpecError(f"Lock da execução ausente ou inválido: {lock_path}") from exc
    digest = _sha256(path)
    if lock.get("sha256") != digest:
        raise ExecutionSpecError("A configuração v2 diverge do lock; crie uma nova versão e lock.")
    paths = {key: _relative_path(value, key) for key, value in payload["paths"].items()}
    profile_observation_config = payload.get("profile_observation_config")
    closed_loop_config = payload.get("closed_loop_config")
    spec = ExecutionSpec(
        execution_id=payload["execution"]["id"],
        version=payload["execution"]["version"],
        protocol_version=payload["execution"]["protocol_version"],
        exposure_policy=payload["execution"]["exposure_policy"],
        paths=paths,
        profile_observation_config=(
            _relative_path(profile_observation_config, "profile_observation_config")
            if profile_observation_config is not None
            else None
        ),
        closed_loop_config=(
            _relative_path(closed_loop_config, "closed_loop_config")
            if closed_loop_config is not None
            else None
        ),
        sha256=digest,
        source_path=path,
        project_root=project_root.resolve(),
    )
    for key in spec.paths:
        spec.resolve_path(key, project_root)
    if spec.profile_observation_config is not None:
        spec.resolve_path("profile_observation_config", project_root)
    if spec.closed_loop_config is not None:
        spec.resolve_path("closed_loop_config", project_root)
    return spec


def prepare_execution_directories(spec: ExecutionSpec, project_root: Path | None = None) -> None:
    """Cria somente diretórios próprios da execução, nunca a árvore v1.1."""
    for key in OUTPUT_KEYS:
        path = spec.resolve_path(key, project_root)
        directory = path.parent if key in {"split_manifest", "catalog"} else path
        directory.mkdir(parents=True, exist_ok=True)


def validate_matrix_execution(matrix: Any, spec: ExecutionSpec) -> None:
    """Impede que uma matriz seja escrita ou auditada com outra execução."""
    if getattr(matrix, "execution_id", None) != spec.execution_id:
        raise ExecutionSpecError("A matriz não pertence à execução informada.")
    if getattr(matrix, "execution_config_sha256", None) != spec.sha256:
        raise ExecutionSpecError("O hash da matriz diverge da configuração de execução.")


def main(argv: Sequence[str] | None = None) -> int:
    """Valida, prepara ou audita uma execução v2 sem tocar na v1.1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "prepare", "audit"))
    parser.add_argument("--spec", type=Path, default=DEFAULT_V2_EXECUTION_SPEC_PATH)
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--agents", type=Path)
    parser.add_argument("--matrix-config", type=Path)
    args = parser.parse_args(argv)
    try:
        spec = load_execution_spec(args.spec)
        if args.command == "validate":
            print(json.dumps({"execution_id": spec.execution_id, "sha256": spec.sha256}, sort_keys=True))
            return 0
        if args.command == "prepare":
            prepare_execution_directories(spec)
            print(json.dumps({"prepared": spec.output_paths}, ensure_ascii=False, sort_keys=True))
            return 0
        from cinebot_ml.experiments.freeze import audit_execution
        from cinebot_ml.experiments.matrix import load_experiment_matrix

        matrix = load_experiment_matrix(
            experiment_config_path=args.experiment_config or PROJECT_ROOT / "configs/experiment_v1.yaml",
            agent_config_path=args.agents or PROJECT_ROOT / "configs/agents/personas_v1.yaml",
            matrix_config_path=args.matrix_config or PROJECT_ROOT / "configs/experiments/matrix_v1.yaml",
            execution_spec_path=args.spec,
        )
        report = audit_execution(matrix, spec.resolve_path("raw"), execution_spec=spec)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
        return 0 if report["complete"] else 2
    except (ExecutionSpecError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Erro na execução versionada: {exc}") from exc


if __name__ == "__main__":
    raise SystemExit(main())
