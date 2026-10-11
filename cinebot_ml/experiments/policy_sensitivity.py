"""Seleção reproduzível de parâmetros B5 somente na validação v2."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml
from jsonschema import Draft202012Validator

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.personalization import ProfileUpdateConfig, ProfileUpdater, load_profile_update_config
from cinebot_ml.ranking.incremental import IncrementalConfig, load_incremental_config


DEFAULT_POLICY_SENSITIVITY_CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/policy_sensitivity_v2.yaml"
)
DEFAULT_POLICY_SENSITIVITY_SCHEMA_PATH = (
    PROJECT_ROOT / "configs/experiments/policy_sensitivity_v2.schema.json"
)
DEFAULT_V2_1_POLICY_SENSITIVITY_CONFIG_PATH = (
    PROJECT_ROOT / "configs/experiments/policy_sensitivity_v2_1.yaml"
)
DEFAULT_V2_1_POLICY_SENSITIVITY_SCHEMA_PATH = (
    PROJECT_ROOT / "configs/experiments/policy_sensitivity_v2_1.schema.json"
)


class PolicySensitivityError(ValueError):
    """Configuração, observação ou seleção de sensibilidade inválida."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PolicyScenario:
    scenario_id: str
    learning_rate: float
    decay: float
    feedback_adjustment_weight: float

    @property
    def scenario_hash(self) -> str:
        return _digest(self.to_dict())[:24]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "learning_rate": self.learning_rate,
            "decay": self.decay,
            "feedback_adjustment_weight": self.feedback_adjustment_weight,
        }


@dataclass(frozen=True)
class PolicySensitivityConfig:
    version: str
    execution_id: str
    partition: str
    validation_output_root: str
    primary_metric: str
    selection_policy: str
    scenarios: tuple[PolicyScenario, ...]
    source_path: Path
    source_sha256: str
    snapshot: Mapping[str, Any]


@dataclass(frozen=True)
class SensitivityRuntime:
    """Configurações em memória de um cenário, sem alterar arquivos v1."""

    profile_updater: ProfileUpdater
    incremental_config: IncrementalConfig


def load_policy_sensitivity_config(
    path: Path = DEFAULT_POLICY_SENSITIVITY_CONFIG_PATH,
    *,
    schema_path: Path = DEFAULT_POLICY_SENSITIVITY_SCHEMA_PATH,
) -> PolicySensitivityConfig:
    """Carrega a grade v2 e confirma que ela está congelada por lock."""
    path = path.resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.resolve().read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise PolicySensitivityError(f"Não foi possível carregar a sensibilidade: {exc}") from exc
    error = next(iter(Draft202012Validator(schema).iter_errors(payload)), None)
    if error:
        raise PolicySensitivityError(f"Configuração de sensibilidade inválida: {error.message}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicySensitivityError("Lock da sensibilidade ausente ou inválido.") from exc
    if lock.get("sha256") != digest:
        raise PolicySensitivityError("A sensibilidade foi alterada sem nova versão.")
    scenarios = tuple(
        PolicyScenario(
            scenario_id,
            float(values["learning_rate"]),
            float(values["decay"]),
            float(values["feedback_adjustment_weight"]),
        )
        for scenario_id, values in sorted(payload["scenarios"].items())
    )
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise PolicySensitivityError("Cenários de sensibilidade duplicados.")
    if any(
        len({getattr(scenario, parameter) for scenario in scenarios}) < 2
        for parameter in ("learning_rate", "decay", "feedback_adjustment_weight")
    ):
        raise PolicySensitivityError("A matriz deve variar os três parâmetros incrementais.")
    return PolicySensitivityConfig(
        str(payload["version"]),
        str(payload["execution_id"]),
        str(payload["partition"]),
        str(payload["validation_output_root"]),
        str(payload["primary_metric"]),
        str(payload["selection"]["policy"]),
        scenarios,
        path,
        digest,
        dict(payload),
    )


def build_sensitivity_runtime(
    config: PolicySensitivityConfig,
    scenario: PolicyScenario,
) -> SensitivityRuntime:
    """Materializa parâmetros de um cenário preservando arquivos v1 congelados."""
    if scenario not in config.scenarios:
        raise PolicySensitivityError("Cenário não pertence à configuração de sensibilidade.")
    profile_base = load_profile_update_config()
    incremental_base = load_incremental_config()
    provenance = {
        "sensitivity_config_sha256": config.source_sha256,
        "scenario": scenario.to_dict(),
        "scenario_hash": scenario.scenario_hash,
    }
    profile_config: ProfileUpdateConfig = replace(
        profile_base,
        version=f"{profile_base.version}-sensitivity-{scenario.scenario_id}",
        learning_rate=scenario.learning_rate,
        decay=scenario.decay,
        source_path=config.source_path,
        source_sha256=config.source_sha256,
        snapshot={"base": dict(profile_base.snapshot), **provenance},
    )
    incremental_config = replace(
        incremental_base,
        version=f"{incremental_base.version}-sensitivity-{scenario.scenario_id}",
        adjustment_weight=scenario.feedback_adjustment_weight,
        source_path=config.source_path,
        source_sha256=config.source_sha256,
        snapshot={"base": dict(incremental_base.snapshot), **provenance},
    )
    return SensitivityRuntime(ProfileUpdater(config=profile_config), incremental_config)


def select_validation_scenario(
    config: PolicySensitivityConfig,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seleciona cenário por média em pares completos, sem consultar holdout."""
    allowed = {scenario.scenario_id: scenario for scenario in config.scenarios}
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    evidence_fields = ("agent_id", "seed", "candidate_set_id")
    if config.execution_id == "execution-v2.1":
        evidence_fields += ("relevance_id", "trajectory_sha256", "state_version")
    identities: dict[str, dict[str, tuple[Any, ...]]] = defaultdict(dict)
    invalid: list[dict[str, Any]] = []
    for observation in observations:
        scenario_id = str(observation.get("scenario_id", ""))
        partition = observation.get("partition")
        status = observation.get("status")
        comparison_id = str(observation.get("comparison_id", ""))
        if partition != config.partition:
            raise PolicySensitivityError("Seleção de sensibilidade aceita exclusivamente validation.")
        if scenario_id not in allowed:
            raise PolicySensitivityError(f"Cenário desconhecido: {scenario_id}")
        if not comparison_id:
            raise PolicySensitivityError("comparison_id é obrigatório na seleção.")
        if status != "evaluated" or observation.get(config.primary_metric) is None:
            invalid.append({"scenario_id": scenario_id, "comparison_id": comparison_id, "status": status})
            continue
        if comparison_id in grouped[scenario_id]:
            raise PolicySensitivityError("Observação duplicada para cenário e comparison_id.")
        identity = tuple(observation.get(key) for key in evidence_fields)
        if any(value is None for value in identity):
            if config.execution_id == "execution-v2.1":
                raise PolicySensitivityError("A evidência compartilhada é obrigatória na seleção v2.1.")
            raise PolicySensitivityError("Agente, seed e conjunto candidato são obrigatórios na seleção.")
        try:
            value = float(observation[config.primary_metric])
        except (TypeError, ValueError) as exc:
            raise PolicySensitivityError("A métrica primária deve ser numérica.") from exc
        if not 0.0 <= value <= 1.0:
            raise PolicySensitivityError("A métrica primária deve estar no intervalo [0, 1].")
        grouped[scenario_id][comparison_id] = value
        identities[scenario_id][comparison_id] = identity
    expected_ids = set.intersection(*(set(values) for values in grouped.values())) if grouped else set()
    if not expected_ids:
        raise PolicySensitivityError("Nenhum par completo de validação foi produzido.")
    summaries = []
    reference_identity: dict[str, tuple[Any, ...]] | None = None
    for scenario in config.scenarios:
        values = grouped.get(scenario.scenario_id, {})
        if set(values) != expected_ids:
            raise PolicySensitivityError("Cenários não possuem o mesmo conjunto de pares de validação.")
        if reference_identity is None:
            reference_identity = identities[scenario.scenario_id]
        elif any(
            identities[scenario.scenario_id][comparison_id] != reference_identity[comparison_id]
            for comparison_id in expected_ids
        ):
            raise PolicySensitivityError(
                "Cenários possuem evidência incompatível."
                if config.execution_id == "execution-v2.1"
                else "Cenários possuem agentes, seeds ou candidatos incompatíveis."
            )
        mean = sum(values[comparison_id] for comparison_id in sorted(expected_ids)) / len(expected_ids)
        summaries.append({
            **scenario.to_dict(),
            "scenario_hash": scenario.scenario_hash,
            "comparison_count": len(expected_ids),
            "mean_ndcg_at_5": mean,
        })
    selected = min(summaries, key=lambda item: (-item["mean_ndcg_at_5"], item["scenario_id"]))
    selection = {
        "selection_version": "1.0",
        "execution_id": config.execution_id,
        "partition": config.partition,
        "primary_metric": config.primary_metric,
        "selection_policy": config.selection_policy,
        "config_sha256": config.source_sha256,
        "selected_scenario_id": selected["scenario_id"],
        "selected_scenario_hash": selected["scenario_hash"],
        "scenario_summaries": summaries,
        "invalid_observations": invalid,
        "holdout_used_for_selection": False,
    }
    return {**selection, "selection_id": _digest(selection)[:24]}


def execute_validation_scenarios(
    config: PolicySensitivityConfig,
    runner: Callable[[PolicyScenario], Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Executa cada cenário e preserva sucessos e falhas da validação v2."""
    if config.partition != "validation":
        raise PolicySensitivityError("A matriz de sensibilidade só pode executar em validation.")
    rows: list[dict[str, Any]] = []
    for scenario in config.scenarios:
        common = {
            "execution_id": config.execution_id,
            "partition": "validation",
            "config_sha256": config.source_sha256,
            **scenario.to_dict(),
            "scenario_hash": scenario.scenario_hash,
        }
        try:
            result = dict(runner(scenario))
            if result.get("partition") not in {None, "validation"}:
                raise PolicySensitivityError("O runner não pode registrar resultado de holdout.")
            required = ("comparison_id", "agent_id", "seed", "candidate_set_id")
            missing = next((key for key in required if result.get(key) is None), None)
            if missing is not None:
                raise PolicySensitivityError(f"Runner não informou {missing}.")
            status = str(result.get("status", "evaluated"))
            if status not in {"evaluated", "failed", "excluded"}:
                raise PolicySensitivityError("Status de cenário inválido.")
            rows.append({**result, **common, "status": status})
        except Exception as exc:
            rows.append({
                **common,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            })
    return tuple(rows)


def write_validation_observations(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    """Grava todas as observações de validação, inclusive falhas e exclusões."""
    if path.exists():
        raise FileExistsError("Observações de sensibilidade não serão sobrescritas.")
    encoded_rows = []
    for row in rows:
        if row.get("partition") != "validation":
            raise PolicySensitivityError("Observações de sensibilidade devem pertencer à validation.")
        if not row.get("scenario_id") or not row.get("status"):
            raise PolicySensitivityError("Observação sem cenário ou status não pode ser gravada.")
        encoded_rows.append(_canonical(dict(row)))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(encoded_rows) + ("\n" if encoded_rows else ""), encoding="utf-8")


def freeze_validation_selection(selection: Mapping[str, Any], path: Path) -> None:
    """Persiste a decisão de validação uma única vez com lock verificável."""
    if selection.get("partition") != "validation" or selection.get("holdout_used_for_selection") is not False:
        raise PolicySensitivityError("Somente uma seleção exclusiva de validação pode ser congelada.")
    if not selection.get("selected_scenario_id") or not selection.get("selection_id"):
        raise PolicySensitivityError("Seleção sem cenário ou identidade não pode ser congelada.")
    encoded = _canonical(dict(selection)) + "\n"
    lock_path = path.with_suffix(".lock.json")
    if path.exists() or lock_path.exists():
        raise FileExistsError("Seleção congelada não será sobrescrita.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    lock_path.write_text(
        json.dumps({"sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}) + "\n",
        encoding="utf-8",
    )


def load_frozen_validation_selection(
    config: PolicySensitivityConfig,
    path: Path,
) -> dict[str, Any]:
    """Valida a decisão que autoriza a futura avaliação no holdout v2."""
    try:
        raw = path.read_bytes()
        selection = json.loads(raw)
        lock = json.loads(path.with_suffix(".lock.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PolicySensitivityError("Seleção congelada ou lock indisponível.") from exc
    if lock.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise PolicySensitivityError("Seleção congelada diverge do lock.")
    if not isinstance(selection, dict):
        raise PolicySensitivityError("Seleção congelada deve ser um objeto.")
    if selection.get("partition") != "validation" or selection.get("holdout_used_for_selection") is not False:
        raise PolicySensitivityError("Seleção congelada não foi produzida exclusivamente na validação.")
    if selection.get("execution_id") != config.execution_id:
        raise PolicySensitivityError("Seleção pertence a outra execução.")
    if selection.get("config_sha256") != config.source_sha256:
        raise PolicySensitivityError("Seleção usa outra configuração de sensibilidade.")
    scenarios = {scenario.scenario_id: scenario for scenario in config.scenarios}
    selected_id = selection.get("selected_scenario_id")
    selected = scenarios.get(selected_id)
    if selected is None or selection.get("selected_scenario_hash") != selected.scenario_hash:
        raise PolicySensitivityError("Cenário selecionado não pertence à configuração congelada.")
    summary_ids = {row.get("scenario_id") for row in selection.get("scenario_summaries", [])}
    if summary_ids != set(scenarios):
        raise PolicySensitivityError("Seleção não contabiliza todos os cenários configurados.")
    return selection
