"""Configuração, validação e rastreabilidade dos experimentos científicos."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT


DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment_v1.yaml"
DEFAULT_SCHEMA_PATH = PROJECT_ROOT / "configs" / "experiment_v1.schema.json"
INPUT_PATH_KEYS = ("catalog", "dataset", "feedback")
OUTPUT_PATH_KEYS = ("output_root", "split_manifests", "manifests", "raw", "tables", "figures", "reports")
ALLOWED_METHODS = {f"B{index}" for index in range(7)}
ALLOWED_CONDITIONS = {f"C{index}" for index in range(6)}
ALLOWED_PROFILES = {f"P{index}" for index in range(6)}
SECRET_MARKERS = ("password", "passwd", "secret", "token", "api_key", "apikey", "credential")


class ConfigValidationError(ValueError):
    """Configuração incompatível com o contrato experimental."""


class FrozenConfigError(ConfigValidationError):
    """Configuração congelada foi modificada sem atualização explícita."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigValidationError(f"'{field}' deve ser um objeto.")
    return value


def _require(mapping: Mapping[str, Any], field: str, parent: str = "config") -> Any:
    if field not in mapping:
        raise ConfigValidationError(f"Campo obrigatório ausente: {parent}.{field}")
    return mapping[field]


def _reject_unknown(mapping: Mapping[str, Any], allowed: set[str], parent: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ConfigValidationError(f"Campo desconhecido em {parent}: {unknown[0]}")


def _string_list(value: Any, field: str, allowed: set[str] | None = None) -> list[str]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise ConfigValidationError(f"'{field}' deve ser uma lista não vazia de textos.")
    if len(value) != len(set(value)):
        raise ConfigValidationError(f"'{field}' não pode conter valores duplicados.")
    invalid = sorted(set(value) - allowed) if allowed is not None else []
    if invalid:
        raise ConfigValidationError(f"Valor inválido em '{field}': {invalid[0]}")
    return value


def _contains_secret(value: Any, path: str = "config") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_path = f"{path}.{key}"
            if any(marker in str(key).lower() for marker in SECRET_MARKERS):
                return key_path
            found = _contains_secret(child, key_path)
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _contains_secret(child, f"{path}[{index}]")
            if found:
                return found
    return None


def validate_config(config: Mapping[str, Any]) -> None:
    top_level = {"schema_version", "experiment", "seeds", "k_values", "paths", "data_versions", "methods", "conditions", "profiles", "metrics", "splits", "execution"}
    _reject_unknown(config, top_level, "config")
    for field in top_level:
        _require(config, field)
    if config["schema_version"] != "1.0":
        raise ConfigValidationError("schema_version deve ser '1.0'.")

    experiment = _mapping(config["experiment"], "experiment")
    _reject_unknown(experiment, {"name", "version", "protocol_version", "frozen"}, "experiment")
    for field in ("name", "version", "protocol_version", "frozen"):
        _require(experiment, field, "experiment")
    if any(not isinstance(experiment[field], str) or not experiment[field] for field in ("name", "version", "protocol_version")):
        raise ConfigValidationError("Nome e versões do experimento devem ser textos não vazios.")
    if not isinstance(experiment["frozen"], bool):
        raise ConfigValidationError("experiment.frozen deve ser booleano.")

    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(seed) is not int for seed in seeds):
        raise ConfigValidationError("'seeds' deve ser uma lista não vazia de inteiros.")
    k_values = config["k_values"]
    if not isinstance(k_values, list) or not k_values or any(type(k) is not int or k <= 0 for k in k_values):
        raise ConfigValidationError("'k_values' deve conter apenas inteiros maiores que zero.")

    paths = _mapping(config["paths"], "paths")
    expected_paths = set(INPUT_PATH_KEYS + OUTPUT_PATH_KEYS)
    _reject_unknown(paths, expected_paths, "paths")
    for field in expected_paths:
        value = _require(paths, field, "paths")
        if not isinstance(value, str) or not value.strip():
            raise ConfigValidationError(f"paths.{field} deve ser um texto não vazio.")
        if Path(value).is_absolute():
            raise ConfigValidationError(f"paths.{field} deve ser relativo à raiz do projeto.")

    versions = _mapping(config["data_versions"], "data_versions")
    _reject_unknown(versions, set(INPUT_PATH_KEYS), "data_versions")
    for field in INPUT_PATH_KEYS:
        if not isinstance(_require(versions, field, "data_versions"), str):
            raise ConfigValidationError(f"data_versions.{field} deve ser texto.")

    methods = _mapping(config["methods"], "methods")
    _reject_unknown(methods, {"enabled", "optional"}, "methods")
    _string_list(_require(methods, "enabled", "methods"), "methods.enabled", ALLOWED_METHODS)
    optional = methods.get("optional", [])
    if optional:
        _string_list(optional, "methods.optional", ALLOWED_METHODS)
    _string_list(config["conditions"], "conditions", ALLOWED_CONDITIONS)
    _string_list(config["profiles"], "profiles", ALLOWED_PROFILES)

    metrics = _mapping(config["metrics"], "metrics")
    _reject_unknown(metrics, {"primary", "secondary"}, "metrics")
    _string_list(_require(metrics, "primary", "metrics"), "metrics.primary")
    secondary = _require(metrics, "secondary", "metrics")
    if not isinstance(secondary, list) or any(not isinstance(item, str) for item in secondary):
        raise ConfigValidationError("metrics.secondary deve ser uma lista de textos.")

    splits = _mapping(config["splits"], "splits")
    split_fields = {"strategy", "temporal_strategy", "train_fraction", "validation_fraction", "test_fraction", "attempts"}
    _reject_unknown(splits, split_fields, "splits")
    for field in split_fields:
        _require(splits, field, "splits")
    if splits["strategy"] != "deterministic_group_stratification_search":
        raise ConfigValidationError("Estratégia de split desconhecida.")
    if splits["temporal_strategy"] != "chronological_replay":
        raise ConfigValidationError("Estratégia temporal desconhecida.")
    fractions = [splits[name] for name in ("train_fraction", "validation_fraction", "test_fraction")]
    if any(type(value) not in (int, float) or value <= 0 or value >= 1 for value in fractions) or not math.isclose(sum(fractions), 1.0):
        raise ConfigValidationError("As frações de split devem estar entre zero e um e somar 1.")
    if type(splits["attempts"]) is not int or splits["attempts"] <= 0:
        raise ConfigValidationError("splits.attempts deve ser inteiro maior que zero.")

    execution = _mapping(config["execution"], "execution")
    execution_fields = {"deterministic_repetitions", "stochastic_repetitions", "unit", "fail_on_dirty_worktree"}
    _reject_unknown(execution, execution_fields, "execution")
    for field in execution_fields:
        _require(execution, field, "execution")
    if execution["unit"] not in {"user", "profile"}:
        raise ConfigValidationError("execution.unit deve ser 'user' ou 'profile'.")
    for field in ("deterministic_repetitions", "stochastic_repetitions"):
        if type(execution[field]) is not int or execution[field] <= 0:
            raise ConfigValidationError(f"execution.{field} deve ser inteiro maior que zero.")
    if not isinstance(execution["fail_on_dirty_worktree"], bool):
        raise ConfigValidationError("execution.fail_on_dirty_worktree deve ser booleano.")
    secret = _contains_secret(config)
    if secret:
        raise ConfigValidationError(f"Credencial ou segredo proibido na configuração: {secret}")
    schema = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    error = next(iter(Draft202012Validator(schema).iter_errors(config)), None)
    if error:
        location = ".".join(map(str, error.absolute_path)) or "config"
        raise ConfigValidationError(f"Schema inválido em {location}: {error.message}")


def _lock_path(config_path: Path) -> Path:
    return config_path.with_suffix(".lock.json")


def load_config(config_path: Path = DEFAULT_CONFIG_PATH, *, check_frozen: bool = True) -> dict[str, Any]:
    config_path = config_path.resolve()
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigValidationError(f"Não foi possível carregar {config_path}: {exc}") from exc
    if not isinstance(config, Mapping):
        raise ConfigValidationError("A raiz da configuração deve ser um objeto.")
    result = dict(config)
    validate_config(result)
    if check_frozen and result["experiment"]["frozen"]:
        lock_path = _lock_path(config_path)
        try:
            lock = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise FrozenConfigError(f"Lock da configuração congelada ausente ou inválido: {lock_path}") from exc
        actual = sha256_file(config_path)
        if lock.get("sha256") != actual:
            raise FrozenConfigError("Configuração congelada foi alterada; crie uma nova versão e atualize o lock explicitamente.")
    return result


def resolve_project_path(relative_path: str, project_root: Path = PROJECT_ROOT) -> Path:
    root = project_root.resolve()
    resolved = (root / relative_path).resolve()
    if resolved != root and root not in resolved.parents:
        raise ConfigValidationError(f"Caminho escapa da raiz do projeto: {relative_path}")
    return resolved


def validate_paths(config: Mapping[str, Any], project_root: Path = PROJECT_ROOT) -> None:
    paths = config["paths"]
    for key in INPUT_PATH_KEYS:
        path = resolve_project_path(paths[key], project_root)
        if not path.is_file():
            raise ConfigValidationError(f"Entrada inexistente em paths.{key}: {paths[key]}")
    for key in OUTPUT_PATH_KEYS:
        path = resolve_project_path(paths[key], project_root)
        probe = path if path.exists() else next((parent for parent in path.parents if parent.exists()), None)
        if probe is None or not probe.is_dir() or not os.access(probe, os.W_OK):
            raise ConfigValidationError(f"Saída sem permissão de escrita em paths.{key}: {paths[key]}")


def prepare_output_directories(config: Mapping[str, Any], project_root: Path = PROJECT_ROOT) -> None:
    validate_paths(config, project_root)
    for key in OUTPUT_PATH_KEYS:
        resolve_project_path(config["paths"][key], project_root).mkdir(parents=True, exist_ok=True)


def _git_value(*args: str, project_root: Path = PROJECT_ROOT) -> str:
    process = subprocess.run(["git", *args], cwd=project_root, text=True, capture_output=True, check=False)
    return process.stdout.strip() if process.returncode == 0 else "unavailable"


def environment_diagnostic(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    packages = ("PyYAML", "jsonschema", "pandas", "scikit-learn", "mlflow", "dvc")
    dependencies: dict[str, str] = {}
    for package in packages:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = "not-installed"
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "dependencies": dependencies,
        "git_commit": _git_value("rev-parse", "HEAD", project_root=project_root),
        "worktree_dirty": bool(_git_value("status", "--porcelain", project_root=project_root)),
    }


def build_execution_manifest(
    config: Mapping[str, Any],
    *,
    method: str,
    condition: str,
    profile: str,
    seed: int,
    run: int,
    project_root: Path = PROJECT_ROOT,
    method_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if method not in config["methods"]["enabled"]:
        raise ConfigValidationError(f"Método não habilitado: {method}")
    if condition not in config["conditions"] or profile not in config["profiles"]:
        raise ConfigValidationError("Condição ou perfil fora da configuração.")
    if seed not in config["seeds"] or run <= 0:
        raise ConfigValidationError("Seed ou repetição fora da configuração.")
    inputs = {
        key: {"path": config["paths"][key], "version": config["data_versions"][key], "sha256": sha256_file(resolve_project_path(config["paths"][key], project_root))}
        for key in INPUT_PATH_KEYS
    }
    identity = {
        "config_sha256": _canonical_hash(config), "inputs": inputs, "git_commit": _git_value("rev-parse", "HEAD", project_root=project_root),
        "method": method, "condition": condition, "profile": profile, "seed": seed, "run": run,
    }
    if method_manifest is not None:
        identity["method_manifest"] = dict(method_manifest)
    return {
        "manifest_version": "1.0", "execution_id": _canonical_hash(identity)[:20],
        "created_at_utc": datetime.now(timezone.utc).isoformat(), **identity,
        "config_snapshot": config,
        "protocol_version": config["experiment"]["protocol_version"], "environment": environment_diagnostic(project_root),
        "artifacts": {key: config["paths"][key] for key in ("raw", "tables", "figures", "reports")},
    }


def write_execution_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any], project_root: Path = PROJECT_ROOT) -> Path:
    output = resolve_project_path(config["paths"]["manifests"], project_root)
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"{manifest['execution_id']}.json"
    content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise FileExistsError(f"Manifesto existente e inválido não será sobrescrito: {path}") from exc
        if existing.get("execution_id") == manifest["execution_id"]:
            return path
        raise FileExistsError(f"Manifesto existente não será sobrescrito: {path}")
    path.write_text(content, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "diagnose", "prepare", "manifest"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--method", default="B0")
    parser.add_argument("--condition", default="C0")
    parser.add_argument("--profile", default="P0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        config = load_config(args.config)
        validate_paths(config)
        if args.command == "validate":
            print(f"Configuração válida: {args.config}")
        elif args.command == "diagnose":
            print(json.dumps(environment_diagnostic(), indent=2, sort_keys=True))
        elif args.command == "prepare":
            prepare_output_directories(config)
            print("Diretórios de resultados preparados.")
        else:
            manifest = build_execution_manifest(config, method=args.method, condition=args.condition, profile=args.profile, seed=args.seed, run=args.run)
            print(write_execution_manifest(manifest, config))
    except (ConfigValidationError, FileExistsError) as exc:
        raise SystemExit(f"Erro de reprodutibilidade: {exc}") from exc


if __name__ == "__main__":
    main()
