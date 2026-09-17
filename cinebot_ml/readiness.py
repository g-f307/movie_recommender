"""Diagnóstico reproduzível de prontidão do ambiente experimental."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from cinebot_ml.config import PROJECT_ROOT


SUPPORTED_PYTHON = ((3, 11), (3, 14))


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    detail: str


def _capture(name: str, function: Callable[[], str]) -> ReadinessCheck:
    try:
        return ReadinessCheck(name, "ok", function())
    except Exception as exc:
        return ReadinessCheck(name, "error", f"{type(exc).__name__}: {exc}")


def _python_check() -> str:
    current = sys.version_info[:2]
    if not SUPPORTED_PYTHON[0] <= current <= SUPPORTED_PYTHON[1]:
        raise RuntimeError(
            f"Python {current[0]}.{current[1]} fora do intervalo 3.11–3.14."
        )
    return f"Python {current[0]}.{current[1]} suportado"


def _import_check() -> str:
    modules = (
        "cinebot_ml.dataset",
        "cinebot_ml.experiment_config",
        "cinebot_ml.incremental_benchmark",
        "cinebot_ml.personalization",
        "cinebot_ml.ranking",
        "cinebot_ml.simulation",
    )
    for module in modules:
        importlib.import_module(module)
    return f"{len(modules)} módulos principais importados"


def _config_check() -> str:
    from cinebot_ml.experiment_config import load_config
    from cinebot_ml.personalization import load_profile_update_config
    from cinebot_ml.ranking import (
        load_content_config,
        load_incremental_config,
        load_popularity_config,
        load_static_personalized_config,
        load_supervised_config,
        load_tfidf_config,
    )
    from cinebot_ml.simulation import load_agent_config, load_simulation_config

    loaders = (
        load_config,
        load_popularity_config,
        load_content_config,
        load_tfidf_config,
        load_supervised_config,
        load_static_personalized_config,
        load_incremental_config,
        load_profile_update_config,
        load_agent_config,
        load_simulation_config,
    )
    for loader in loaders:
        loader()
    return f"{len(loaders)} configurações, schemas e locks validados"


def _tracked_inputs_check() -> str:
    required = (
        PROJECT_ROOT / "dvc.yaml",
        PROJECT_ROOT / "datasets" / "movie_preferences.csv.dvc",
    )
    missing = [path.relative_to(PROJECT_ROOT).as_posix() for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Descritor de dados ausente: {missing[0]}")
    return "pipeline DVC e ponteiro do dataset encontrados"


def _full_inputs_check() -> str:
    required = {
        "catalog": PROJECT_ROOT / "data" / "filmes.json",
        "dataset": PROJECT_ROOT / "datasets" / "movie_preferences.csv",
        "feedback": PROJECT_ROOT / "datasets" / "user_feedback.csv",
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Entrada experimental ausente: " + ", ".join(missing))
    return "catálogo, dataset e feedback disponíveis"


def _b2_artifact_check() -> str:
    from cinebot_ml.ranking import TfidfArtifact

    path = PROJECT_ROOT / "artifacts" / "b2_tfidf_v1.json"
    if not path.is_file():
        raise FileNotFoundError(f"Artefato B2 ausente: {path.relative_to(PROJECT_ROOT)}")
    artifact = TfidfArtifact.load(path)
    return f"artefato B2 compatível: {artifact.artifact_id}"


def _b3_artifact_check() -> str:
    from cinebot_ml.ranking import SupervisedArtifact

    model = PROJECT_ROOT / "artifacts" / "production_model.joblib"
    metadata = PROJECT_ROOT / "artifacts" / "model_metadata.json"
    missing = [path.name for path in (model, metadata) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Artefato B3 ausente: " + ", ".join(missing))
    artifact = SupervisedArtifact.load(model, metadata)
    return f"artefato B3 compatível: {artifact.model_sha256[:16]}"


def _dvc_remote_check() -> str:
    config = PROJECT_ROOT / ".dvc" / "config"
    content = config.read_text(encoding="utf-8") if config.is_file() else ""
    if '[remote "' not in content and "remote =" not in content:
        raise RuntimeError("nenhum remoto DVC configurado; dvc pull não recupera os ativos")
    return "remoto DVC configurado"


def run_readiness(mode: str = "ci") -> tuple[ReadinessCheck, ...]:
    if mode not in {"ci", "full"}:
        raise ValueError("mode deve ser ci ou full")
    checks = [
        _capture("python", _python_check),
        _capture("imports", _import_check),
        _capture("configs", _config_check),
        _capture("data_descriptors", _tracked_inputs_check),
    ]
    if mode == "full":
        checks.extend(
            (
                _capture("dvc_remote", _dvc_remote_check),
                _capture("experimental_inputs", _full_inputs_check),
                _capture("b2_artifact", _b2_artifact_check),
                _capture("b3_artifact", _b3_artifact_check),
            )
        )
    return tuple(checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("ci", "full"), nargs="?", default="ci")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    checks = run_readiness(args.mode)
    if args.as_json:
        print(json.dumps([asdict(check) for check in checks], ensure_ascii=False, indent=2))
    else:
        for check in checks:
            print(f"[{check.status.upper()}] {check.name}: {check.detail}")
    return int(any(check.status == "error" for check in checks))


if __name__ == "__main__":
    raise SystemExit(main())
