"""Particionamento experimental determinístico e validações contra leakage."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from cinebot_ml.config import DATASET_PATH, FEEDBACK_PATH, PROJECT_ROOT


SPLIT_VERSION = "1.0"
PARTITIONS = ("train", "validation", "test")
HOLDOUT_FORBIDDEN_PURPOSES = {
    "calibrate",
    "hyperparameter_tuning",
    "model_selection",
    "select_model",
    "select_threshold",
    "threshold_selection",
    "tune",
}
DEFAULT_MANIFEST_DIR = PROJECT_ROOT / "results" / "manifests" / "splits"


class SplitValidationError(ValueError):
    """Indica configuração ou dados incompatíveis com o protocolo de split."""


class ManifestConflictError(FileExistsError):
    """Impede sobrescrita silenciosa de um manifesto existente."""


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    test_fraction: float = 0.20
    attempts: int = 128
    protocol_version: str = "protocol-v1.0"

    def validate(self) -> None:
        fractions = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 or value >= 1 for value in fractions):
            raise SplitValidationError("Todas as frações devem estar entre zero e um.")
        if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise SplitValidationError("As frações de treino, validação e teste devem somar 1.")
        if self.attempts <= 0:
            raise SplitValidationError("attempts deve ser maior que zero.")
        if not isinstance(self.seed, int):
            raise SplitValidationError("seed deve ser um número inteiro definido na configuração.")


@dataclass
class GroupStats:
    group_id: str
    rows: int = 0
    class_counts: Counter | None = None

    def __post_init__(self) -> None:
        if self.class_counts is None:
            self.class_counts = Counter()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_timestamp(value: Any) -> datetime:
    if value is None or not str(value).strip():
        raise SplitValidationError("Evento com timestamp ausente.")
    normalized = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise SplitValidationError(f"Timestamp inválido: {value}") from exc
    if parsed.tzinfo is None:
        raise SplitValidationError(f"Timestamp sem fuso horário: {value}")
    return parsed.astimezone(timezone.utc)


def collect_group_stats(
    rows: Iterable[Mapping[str, Any]],
    group_column: str,
    label_column: str,
) -> dict[str, GroupStats]:
    groups: dict[str, GroupStats] = {}
    observed_classes: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        group = row.get(group_column)
        label = row.get(label_column)
        if group is None or not str(group).strip():
            raise SplitValidationError(f"{group_column} ausente na linha {row_number}.")
        if label is None or not str(label).strip():
            raise SplitValidationError(f"{label_column} ausente na linha {row_number}.")
        group_id = str(group).strip()
        class_id = str(label).strip()
        observed_classes.add(class_id)
        stats = groups.setdefault(group_id, GroupStats(group_id=group_id))
        stats.rows += 1
        stats.class_counts[class_id] += 1

    if not groups:
        raise SplitValidationError("Não existem registros para particionar.")
    if len(groups) < 3:
        raise SplitValidationError("São necessários pelo menos três grupos para treino, validação e teste.")
    if len(observed_classes) < 2:
        raise SplitValidationError("O dataset precisa conter pelo menos duas classes.")
    return groups


def collect_group_stats_from_csv(
    path: Path,
    group_column: str = "movie_id",
    label_column: str = "relevante",
) -> dict[str, GroupStats]:
    if not path.exists():
        raise SplitValidationError(f"Dataset não encontrado: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [column for column in (group_column, label_column) if column not in columns]
        if missing:
            raise SplitValidationError(f"Colunas ausentes em {path.name}: {', '.join(missing)}")
        return collect_group_stats(reader, group_column, label_column)


def _partition_sizes(total_groups: int, config: SplitConfig) -> dict[str, int]:
    raw = {
        "train": total_groups * config.train_fraction,
        "validation": total_groups * config.validation_fraction,
        "test": total_groups * config.test_fraction,
    }
    sizes = {name: max(1, int(math.floor(value))) for name, value in raw.items()}
    while sum(sizes.values()) > total_groups:
        candidate = max(PARTITIONS, key=lambda name: (sizes[name] - raw[name], sizes[name]))
        if sizes[candidate] <= 1:
            raise SplitValidationError("Não foi possível criar três partições não vazias.")
        sizes[candidate] -= 1
    while sum(sizes.values()) < total_groups:
        candidate = max(PARTITIONS, key=lambda name: (raw[name] - sizes[name], -sizes[name]))
        sizes[candidate] += 1
    return sizes


def _candidate_score(
    assignment: dict[str, list[str]],
    groups: Mapping[str, GroupStats],
    config: SplitConfig,
) -> float:
    total_rows = sum(stats.rows for stats in groups.values())
    total_classes = Counter()
    for stats in groups.values():
        total_classes.update(stats.class_counts)
    global_rates = {label: count / total_rows for label, count in total_classes.items()}
    target_fractions = {
        "train": config.train_fraction,
        "validation": config.validation_fraction,
        "test": config.test_fraction,
    }
    score = 0.0
    for partition, group_ids in assignment.items():
        rows = sum(groups[group_id].rows for group_id in group_ids)
        classes = Counter()
        for group_id in group_ids:
            classes.update(groups[group_id].class_counts)
        row_fraction = rows / total_rows
        score += abs(row_fraction - target_fractions[partition]) * 2
        for label, global_rate in global_rates.items():
            partition_rate = classes[label] / rows if rows else 0.0
            score += abs(partition_rate - global_rate)
            if classes[label] == 0:
                score += 10.0
    return score


def split_groups(groups: Mapping[str, GroupStats], config: SplitConfig) -> dict[str, list[str]]:
    config.validate()
    if len(groups) < 3:
        raise SplitValidationError("São necessários pelo menos três grupos.")
    all_classes = Counter()
    for stats in groups.values():
        all_classes.update(stats.class_counts)
    if len([count for count in all_classes.values() if count > 0]) < 2:
        raise SplitValidationError("O dataset precisa conter pelo menos duas classes.")

    sizes = _partition_sizes(len(groups), config)
    group_ids = sorted(groups)
    best_assignment = None
    best_score = math.inf
    for attempt in range(config.attempts):
        candidate = list(group_ids)
        random.Random(config.seed + attempt * 104729).shuffle(candidate)
        train_end = sizes["train"]
        validation_end = train_end + sizes["validation"]
        assignment = {
            "train": sorted(candidate[:train_end]),
            "validation": sorted(candidate[train_end:validation_end]),
            "test": sorted(candidate[validation_end:]),
        }
        score = _candidate_score(assignment, groups, config)
        tie_breaker = tuple(tuple(assignment[name]) for name in PARTITIONS)
        best_tie = tuple(tuple(best_assignment[name]) for name in PARTITIONS) if best_assignment else None
        if score < best_score or (math.isclose(score, best_score) and (best_tie is None or tie_breaker < best_tie)):
            best_assignment = assignment
            best_score = score

    assert best_assignment is not None
    validate_disjoint_partitions(best_assignment)
    return best_assignment


def validate_disjoint_partitions(partitions: Mapping[str, Sequence[str]]) -> None:
    missing = [name for name in PARTITIONS if name not in partitions]
    if missing:
        raise SplitValidationError(f"Partições ausentes: {', '.join(missing)}")
    for name in PARTITIONS:
        if not partitions[name]:
            raise SplitValidationError(f"Partição vazia: {name}")
        normalized = list(map(str, partitions[name]))
        if len(normalized) != len(set(normalized)):
            raise SplitValidationError(f"A partição {name} contém identificadores repetidos.")
    for index, left in enumerate(PARTITIONS):
        for right in PARTITIONS[index + 1 :]:
            overlap = set(map(str, partitions[left])) & set(map(str, partitions[right]))
            if overlap:
                raise SplitValidationError(
                    f"Leakage entre {left} e {right}: {len(overlap)} grupo(s) em comum."
                )


def _partition_summary(group_ids: Sequence[str], groups: Mapping[str, GroupStats]) -> dict[str, Any]:
    classes = Counter()
    rows = 0
    for group_id in group_ids:
        stats = groups[group_id]
        rows += stats.rows
        classes.update(stats.class_counts)
    normalized_ids = sorted(map(str, group_ids))
    return {
        "group_count": len(normalized_ids),
        "rows": rows,
        "class_counts": {key: int(value) for key, value in sorted(classes.items())},
        "group_ids_sha256": _canonical_hash(normalized_ids),
        "group_ids": normalized_ids,
    }


def build_group_manifest(
    groups: Mapping[str, GroupStats],
    partitions: Mapping[str, Sequence[str]],
    config: SplitConfig,
    source_name: str,
    source_sha256: str,
    group_column: str,
    label_column: str,
) -> dict[str, Any]:
    validate_disjoint_partitions(partitions)
    expected = set(groups)
    assigned = {str(group_id) for name in PARTITIONS for group_id in partitions[name]}
    if assigned != expected:
        raise SplitValidationError("As partições não cobrem exatamente todos os grupos da fonte.")
    config_payload = asdict(config)
    partition_payload = {
        name: _partition_summary(partitions[name], groups) for name in PARTITIONS
    }
    identity = {
        "split_version": SPLIT_VERSION,
        "strategy": "deterministic_group_stratification_search",
        "source_sha256": source_sha256,
        "group_column": group_column,
        "label_column": label_column,
        "config": config_payload,
        "partition_hashes": {
            name: partition_payload[name]["group_ids_sha256"] for name in PARTITIONS
        },
    }
    return {
        "manifest_id": _canonical_hash(identity)[:16],
        **identity,
        "source_name": Path(source_name).name,
        "total_groups": len(groups),
        "total_rows": sum(stats.rows for stats in groups.values()),
        "partitions": partition_payload,
    }


def create_group_manifest_from_csv(
    path: Path,
    config: SplitConfig,
    group_column: str = "movie_id",
    label_column: str = "relevante",
) -> dict[str, Any]:
    groups = collect_group_stats_from_csv(path, group_column, label_column)
    partitions = split_groups(groups, config)
    return build_group_manifest(
        groups,
        partitions,
        config,
        source_name=path.name,
        source_sha256=sha256_file(path),
        group_column=group_column,
        label_column=label_column,
    )


def reconstruct_assignments(manifest: Mapping[str, Any]) -> dict[str, str]:
    partitions = manifest.get("partitions")
    if not isinstance(partitions, Mapping):
        raise SplitValidationError("Manifesto sem partições válidas.")
    raw = {
        name: list((partitions.get(name) or {}).get("group_ids") or []) for name in PARTITIONS
    }
    validate_disjoint_partitions(raw)
    assignments = {
        str(group_id): partition for partition, group_ids in raw.items() for group_id in group_ids
    }
    if len(assignments) != int(manifest.get("total_groups", -1)):
        raise SplitValidationError("Quantidade de grupos do manifesto é inconsistente.")
    for name in PARTITIONS:
        expected_hash = partitions[name].get("group_ids_sha256")
        if expected_hash != _canonical_hash(sorted(map(str, raw[name]))):
            raise SplitValidationError(f"Hash inválido na partição {name}.")
    return assignments


def validate_manifest_source(manifest: Mapping[str, Any], source_path: Path) -> None:
    expected = manifest.get("source_sha256")
    current = sha256_file(source_path)
    if expected != current:
        raise SplitValidationError("A fonte mudou depois da geração do manifesto.")


def write_manifest(manifest: Mapping[str, Any], path: Path, overwrite: bool = False) -> None:
    if path.exists() and not overwrite:
        raise ManifestConflictError(f"Manifesto já existe: {path}. Use --overwrite explicitamente.")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SplitValidationError(f"Manifesto inválido: {exc}") from exc
    strategy = manifest.get("strategy")
    if strategy == "per_user_temporal":
        reconstruct_temporal_partitions(manifest)
    else:
        reconstruct_assignments(manifest)
    return manifest


def temporal_split_events(
    events: Sequence[Mapping[str, Any]],
    config: SplitConfig,
    user_column: str = "user_id",
    timestamp_column: str = "timestamp",
) -> dict[str, list[int]]:
    config.validate()
    if not events:
        raise SplitValidationError("Não existem eventos para particionar.")
    by_user: dict[str, list[tuple[int, datetime]]] = defaultdict(list)
    for index, event in enumerate(events):
        user = event.get(user_column)
        if user is None or not str(user).strip():
            raise SplitValidationError(f"{user_column} ausente no evento {index}.")
        by_user[str(user)].append((index, _parse_timestamp(event.get(timestamp_column))))

    result = {name: [] for name in PARTITIONS}
    for _, indexed_events in sorted(by_user.items()):
        timestamps = sorted({timestamp for _, timestamp in indexed_events})
        if len(timestamps) < 3:
            raise SplitValidationError("Cada unidade precisa de pelo menos três timestamps distintos.")
        sizes = _partition_sizes(len(timestamps), config)
        train_end = sizes["train"]
        validation_end = train_end + sizes["validation"]
        time_partition = {
            timestamp: "train" if position < train_end else "validation" if position < validation_end else "test"
            for position, timestamp in enumerate(timestamps)
        }
        for index, timestamp in indexed_events:
            result[time_partition[timestamp]].append(index)

    for name in PARTITIONS:
        result[name].sort()
    validate_temporal_partitions(events, result, user_column, timestamp_column)
    return result


def validate_temporal_partitions(
    events: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, Sequence[int]],
    user_column: str = "user_id",
    timestamp_column: str = "timestamp",
) -> None:
    index_partitions = {name: [str(index) for index in partitions.get(name, [])] for name in PARTITIONS}
    validate_disjoint_partitions(index_partitions)
    expected = set(range(len(events)))
    assigned = {index for name in PARTITIONS for index in partitions[name]}
    if assigned != expected:
        raise SplitValidationError("As partições temporais não cobrem todos os eventos.")

    by_user_partition: dict[str, dict[str, list[datetime]]] = defaultdict(
        lambda: {name: [] for name in PARTITIONS}
    )
    for partition in PARTITIONS:
        for index in partitions[partition]:
            event = events[index]
            user = str(event.get(user_column) or "")
            timestamp = _parse_timestamp(event.get(timestamp_column))
            by_user_partition[user][partition].append(timestamp)

    for values in by_user_partition.values():
        if values["train"] and values["validation"] and max(values["train"]) >= min(values["validation"]):
            raise SplitValidationError("Leakage temporal entre treino e validação.")
        if values["validation"] and values["test"] and max(values["validation"]) >= min(values["test"]):
            raise SplitValidationError("Leakage temporal entre validação e teste.")


def history_before(
    events: Iterable[Mapping[str, Any]],
    recommendation_timestamp: Any,
    timestamp_column: str = "timestamp",
) -> list[Mapping[str, Any]]:
    cutoff = _parse_timestamp(recommendation_timestamp)
    history = []
    for event in events:
        timestamp = _parse_timestamp(event.get(timestamp_column))
        if timestamp >= cutoff:
            continue
        history.append((timestamp, event))
    return [event for _, event in sorted(history, key=lambda item: item[0])]


def assert_partition_allowed(partition: str, purpose: str) -> None:
    normalized_partition = partition.strip().lower()
    normalized_purpose = purpose.strip().lower()
    if normalized_partition not in PARTITIONS:
        raise SplitValidationError(f"Partição desconhecida: {partition}")
    if normalized_partition == "test" and normalized_purpose in HOLDOUT_FORBIDDEN_PURPOSES:
        raise SplitValidationError(
            f"O holdout final não pode ser usado para {purpose}; use treino/validação."
        )


def read_feedback_events(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SplitValidationError(f"Feedback não encontrado: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        missing = [column for column in ("user_id", "timestamp") if column not in columns]
        if missing:
            raise SplitValidationError(f"Colunas ausentes em {path.name}: {', '.join(missing)}")
        return list(reader)


def build_temporal_manifest(
    events: Sequence[Mapping[str, Any]],
    partitions: Mapping[str, Sequence[int]],
    config: SplitConfig,
    source_name: str,
    source_sha256: str,
) -> dict[str, Any]:
    validate_temporal_partitions(events, partitions)
    summaries = {}
    for name in PARTITIONS:
        indexes = sorted(map(int, partitions[name]))
        timestamps = [_parse_timestamp(events[index]["timestamp"]).isoformat() for index in indexes]
        summaries[name] = {
            "event_count": len(indexes),
            "event_indexes": indexes,
            "event_indexes_sha256": _canonical_hash(indexes),
            "minimum_timestamp": min(timestamps) if timestamps else None,
            "maximum_timestamp": max(timestamps) if timestamps else None,
        }
    identity = {
        "split_version": SPLIT_VERSION,
        "strategy": "per_user_temporal",
        "source_sha256": source_sha256,
        "config": asdict(config),
        "partition_hashes": {
            name: summaries[name]["event_indexes_sha256"] for name in PARTITIONS
        },
    }
    return {
        "manifest_id": _canonical_hash(identity)[:16],
        **identity,
        "source_name": Path(source_name).name,
        "total_events": len(events),
        "partitions": summaries,
    }


def reconstruct_temporal_partitions(manifest: Mapping[str, Any]) -> dict[str, list[int]]:
    partitions = manifest.get("partitions")
    if not isinstance(partitions, Mapping):
        raise SplitValidationError("Manifesto temporal sem partições válidas.")
    result: dict[str, list[int]] = {}
    for name in PARTITIONS:
        payload = partitions.get(name)
        if not isinstance(payload, Mapping):
            raise SplitValidationError(f"Partição temporal ausente: {name}")
        indexes = sorted(map(int, payload.get("event_indexes") or []))
        if payload.get("event_indexes_sha256") != _canonical_hash(indexes):
            raise SplitValidationError(f"Hash inválido na partição temporal {name}.")
        result[name] = indexes
    validate_disjoint_partitions({name: list(map(str, indexes)) for name, indexes in result.items()})
    assigned = {index for indexes in result.values() for index in indexes}
    if assigned != set(range(int(manifest.get("total_events", -1)))):
        raise SplitValidationError("Índices temporais não cobrem exatamente os eventos do manifesto.")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--feedback", type=Path, default=FEEDBACK_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attempts", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = SplitConfig(seed=args.seed, attempts=args.attempts)
    try:
        group_manifest = create_group_manifest_from_csv(args.dataset, config)
        feedback_events = read_feedback_events(args.feedback)
        temporal_partitions = temporal_split_events(feedback_events, config)
        temporal_manifest = build_temporal_manifest(
            feedback_events,
            temporal_partitions,
            config,
            source_name=args.feedback.name,
            source_sha256=sha256_file(args.feedback),
        )
        write_manifest(
            group_manifest,
            args.output_dir / "movie_id_split.json",
            overwrite=args.overwrite,
        )
        write_manifest(
            temporal_manifest,
            args.output_dir / "feedback_temporal_split.json",
            overwrite=args.overwrite,
        )
    except (SplitValidationError, ManifestConflictError) as exc:
        raise SystemExit(f"Erro de particionamento: {exc}") from exc
    print(
        f"Splits concluídos: {group_manifest['total_groups']} filmes e "
        f"{temporal_manifest['total_events']} eventos."
    )


if __name__ == "__main__":
    main()
