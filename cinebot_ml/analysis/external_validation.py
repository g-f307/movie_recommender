"""Validação externa limitada e transportabilidade com MovieLens 100K."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from cinebot_ml.config import PROJECT_ROOT
from cinebot_ml.ranking.metrics import ndcg_at_k

from .inference import describe, paired_inference
from .statistical import analyze_official_results


DEFAULT_CONFIG = PROJECT_ROOT / "configs/experiments/external_validation_v1.json"
DEFAULT_ARCHIVE = PROJECT_ROOT / "data/external/ml-100k.zip"
DEFAULT_SYNTHETIC_MANIFEST = (
    PROJECT_ROOT
    / "results/raw/official_v1_1/2942e51456add29e4c999307/experiment.manifest.json"
)
GENRES = (
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
)


class ExternalValidationError(ValueError):
    """Indica violação do protocolo externo congelado."""


@dataclass(frozen=True)
class Rating:
    user_id: int
    movie_id: int
    rating: int
    timestamp: int


def _hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_study_config(path: Path) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("frozen") is not True:
        raise ExternalValidationError("Protocolo externo deve estar congelado.")
    if config.get("selection_registered_before_download") is not True:
        raise ExternalValidationError("Seleção da fonte não foi pré-registrada.")
    fractions = config["split"]
    total = sum(float(fractions[key]) for key in (
        "train_fraction", "adaptation_fraction", "test_fraction"
    ))
    if not math.isclose(total, 1.0):
        raise ExternalValidationError("Frações do split devem somar um.")
    return config


def fetch_archive(config: Mapping[str, Any], destination: Path) -> Path:
    destination = Path(destination)
    expected = str(config["dataset"]["md5"])
    if destination.exists():
        if _hash(destination, "md5") != expected:
            raise ExternalValidationError("Checksum MD5 divergente no arquivo existente.")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    urllib.request.urlretrieve(str(config["dataset"]["url"]), temporary)
    if _hash(temporary, "md5") != expected:
        temporary.unlink(missing_ok=True)
        raise ExternalValidationError("Checksum MD5 divergente após download.")
    temporary.replace(destination)
    return destination


def load_movielens(archive: Path) -> tuple[list[Rating], dict[int, dict[str, Any]], str]:
    try:
        with zipfile.ZipFile(archive) as package:
            readme = package.read("ml-100k/README").decode("latin-1")
            ratings = []
            for line in package.read("ml-100k/u.data").decode("ascii").splitlines():
                user_id, movie_id, rating, timestamp = map(int, line.split("\t"))
                ratings.append(Rating(user_id, movie_id, rating, timestamp))
            movies = {}
            for line in package.read("ml-100k/u.item").decode("latin-1").splitlines():
                fields = line.split("|")
                movie_id = int(fields[0])
                genres = [
                    genre for genre, active in zip(GENRES, fields[5:24])
                    if active == "1" and genre != "unknown"
                ]
                movies[movie_id] = {
                    "movie_id": movie_id,
                    "title": fields[1],
                    "release_date": fields[2] or None,
                    "genres": genres,
                }
    except (OSError, KeyError, zipfile.BadZipFile, ValueError) as exc:
        raise ExternalValidationError(f"Pacote MovieLens inválido: {exc}") from exc
    if len(ratings) != 100_000 or len(movies) != 1_682:
        raise ExternalValidationError("Dimensões do MovieLens 100K divergentes.")
    return ratings, movies, readme


def temporal_split(
    ratings: Sequence[Rating], config: Mapping[str, Any]
) -> tuple[list[Rating], list[Rating], list[Rating], dict[str, Any]]:
    ordered = sorted(ratings, key=lambda row: (
        row.timestamp, row.user_id, row.movie_id
    ))
    train_end = int(len(ordered) * float(config["split"]["train_fraction"]))
    adapt_end = train_end + int(
        len(ordered) * float(config["split"]["adaptation_fraction"])
    )
    train, adaptation, test = (
        ordered[:train_end], ordered[train_end:adapt_end], ordered[adapt_end:]
    )
    if not (max(row.timestamp for row in train) <= min(row.timestamp for row in adaptation)
            <= max(row.timestamp for row in adaptation) <= min(row.timestamp for row in test)):
        raise ExternalValidationError("Split temporal contém sobreposição.")
    manifest = {
        "strategy": config["split"]["strategy"],
        "tie_breakers": config["split"]["tie_breakers"],
        "counts": {
            "all": len(ordered), "train": len(train),
            "adaptation": len(adaptation), "test": len(test),
        },
        "timestamp_ranges": {
            name: [min(row.timestamp for row in values), max(row.timestamp for row in values)]
            for name, values in (
                ("train", train), ("adaptation", adaptation), ("test", test)
            )
        },
        "temporal_order_valid": True,
    }
    return train, adaptation, test, manifest


def _signed(rating: int, config: Mapping[str, Any]) -> float:
    positive = int(config["semantics"]["positive_rating_minimum"])
    negative = int(config["semantics"]["negative_rating_maximum"])
    neutral = int(config["semantics"]["neutral_rating"])
    if rating >= positive:
        return float(rating - neutral)
    if rating <= negative:
        return float(rating - neutral)
    return 0.0


def _genre_profile(
    events: Sequence[Rating],
    movies: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, float]:
    profile: dict[str, float] = defaultdict(float)
    for event in events:
        signal = _signed(event.rating, config)
        for genre in movies[event.movie_id]["genres"]:
            profile[genre] += signal
    return dict(profile)


def _bayesian_popularity(train: Sequence[Rating]) -> dict[int, float]:
    by_movie: dict[int, list[int]] = defaultdict(list)
    for row in train:
        by_movie[row.movie_id].append(row.rating)
    global_mean = sum(row.rating for row in train) / len(train)
    counts = sorted(len(values) for values in by_movie.values())
    prior = counts[int(0.60 * (len(counts) - 1))]
    return {
        movie_id: (len(values) * (sum(values) / len(values)) + prior * global_mean)
        / (len(values) + prior)
        for movie_id, values in by_movie.items()
    }


def _profile_score(movie: Mapping[str, Any], profile: Mapping[str, float]) -> float:
    genres = movie["genres"]
    if not genres:
        return 0.0
    return sum(float(profile.get(genre, 0.0)) for genre in genres) / len(genres)


def _rank(
    method: str,
    candidates: Sequence[int],
    movies: Mapping[int, Mapping[str, Any]],
    popularity: Mapping[int, float],
    static_profile: Mapping[str, float],
    incremental_profile: Mapping[str, float],
    k: int,
) -> list[int]:
    if method == "B0":
        score = lambda movie_id: float(popularity.get(movie_id, 0.0))
    elif method == "B4":
        score = lambda movie_id: _profile_score(movies[movie_id], static_profile)
    elif method == "B5":
        score = lambda movie_id: _profile_score(movies[movie_id], incremental_profile)
    else:
        raise ExternalValidationError(f"Método externo desconhecido: {method}.")
    return sorted(candidates, key=lambda movie_id: (-score(movie_id), movie_id))[:k]


def evaluate_external(
    ratings: Sequence[Rating],
    movies: Mapping[int, Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    train, adaptation, test, split_manifest = temporal_split(ratings, config)
    train_by_user: dict[int, list[Rating]] = defaultdict(list)
    adapt_by_user: dict[int, list[Rating]] = defaultdict(list)
    test_by_user: dict[int, list[Rating]] = defaultdict(list)
    for collection, target in (
        (train, train_by_user), (adaptation, adapt_by_user), (test, test_by_user)
    ):
        for row in collection:
            target[row.user_id].append(row)

    minimums = config["split"]
    relevance_threshold = int(config["semantics"]["positive_rating_minimum"])
    eligible = []
    for user_id in sorted(train_by_user):
        relevant = [
            row for row in test_by_user[user_id]
            if row.rating >= relevance_threshold
        ]
        if (
            len(train_by_user[user_id]) >= int(minimums["minimum_train_events_per_user"])
            and len(adapt_by_user[user_id]) >= int(minimums["minimum_adaptation_events_per_user"])
            and len(relevant) >= int(minimums["minimum_relevant_test_items_per_user"])
        ):
            eligible.append(user_id)
    if len(eligible) < 30:
        raise ExternalValidationError("Menos de 30 usuários elegíveis no split externo.")

    popularity = _bayesian_popularity(train)
    all_movies = set(movies)
    rows = []
    recommended: dict[str, set[int]] = defaultdict(set)
    for user_id in eligible:
        initial = _genre_profile(train_by_user[user_id], movies, config)
        update = _genre_profile(adapt_by_user[user_id], movies, config)
        incremental = {
            genre: initial.get(genre, 0.0) + update.get(genre, 0.0)
            for genre in set(initial) | set(update)
        }
        seen = {
            row.movie_id
            for row in train_by_user[user_id] + adapt_by_user[user_id]
        }
        candidates = sorted(all_movies - seen)
        relevance = {
            row.movie_id: float(row.rating)
            for row in test_by_user[user_id]
            if row.rating >= relevance_threshold and row.movie_id in candidates
        }
        result = {"user_id_hash": hashlib.sha256(
            f"movielens-100k:{user_id}".encode()
        ).hexdigest()[:16], "candidate_count": len(candidates),
            "relevant_test_items": len(relevance)}
        for method in ("B0", "B4", "B5"):
            ranking = _rank(
                method, candidates, movies, popularity, initial, incremental,
                int(config["semantics"]["k"]),
            )
            result[f"{method.lower()}_ndcg_at_5"] = ndcg_at_k(ranking, relevance, 5)
            recommended[method].update(ranking)
        result["difference_b5_minus_b4"] = (
            result["b5_ndcg_at_5"] - result["b4_ndcg_at_5"]
        )
        result["difference_b5_minus_b0"] = (
            result["b5_ndcg_at_5"] - result["b0_ndcg_at_5"]
        )
        rows.append(result)

    contrasts = {}
    for left, right, name in (
        ("b4_ndcg_at_5", "b5_ndcg_at_5", "B5_minus_B4"),
        ("b0_ndcg_at_5", "b5_ndcg_at_5", "B5_minus_B0"),
    ):
        inference = paired_inference(
            [row[left] for row in rows],
            [row[right] for row in rows],
            resamples=int(config["inference"]["bootstrap_resamples"]),
            seed=int(config["inference"]["bootstrap_seed"]),
        )
        contrasts[name] = {
            **inference,
            "left": describe([row[left] for row in rows]),
            "right": describe([row[right] for row in rows]),
        }
    primary = contrasts["B5_minus_B4"]
    decision = (
        "supported" if primary["mean_difference"] > 0
        and primary["confidence_interval_95"][0] > 0
        and primary["p_value_two_sided"] < float(config["inference"]["alpha"])
        else "not_supported"
    )
    coverage = {
        "catalog_movies": len(movies),
        "ratings_with_known_movie": sum(row.movie_id in movies for row in ratings),
        "ratings_total": len(ratings),
        "rating_mapping_coverage": sum(row.movie_id in movies for row in ratings) / len(ratings),
        "eligible_users": len(eligible),
        "all_users": len({row.user_id for row in ratings}),
        "eligible_user_coverage": len(eligible) / len({row.user_id for row in ratings}),
        "recommendation_catalog_coverage": {
            method: len(values) / len(movies) for method, values in recommended.items()
        },
    }
    report = {
        "unit_of_analysis": "anonymous_movielens_user",
        "k": 5,
        "eligible_users": len(rows),
        "contrasts": contrasts,
        "external_H1_decision": decision,
        "split": split_manifest,
        "coverage": coverage,
    }
    return report, rows, split_manifest


def transportability(
    external: Mapping[str, Any], synthetic_manifest: Path
) -> dict[str, Any]:
    synthetic = analyze_official_results(synthetic_manifest)[0][
        "paired_difference_b5_minus_b4"
    ]
    public = external["contrasts"]["B5_minus_B4"]
    def direction(value: float) -> str:
        if value > 0:
            return "positive"
        if value < 0:
            return "negative"
        return "neutral"

    synthetic_direction = direction(float(synthetic["mean_difference"]))
    public_direction = direction(float(public["mean_difference"]))
    return {
        "contrast": "B5_minus_B4_ndcg_at_5",
        "synthetic": {
            "n": synthetic["valid_pairs"],
            "mean_difference": synthetic["mean_difference"],
            "confidence_interval_95": synthetic["confidence_interval_95"],
            "p_value": synthetic["p_value_two_sided"],
            "direction": synthetic_direction,
        },
        "public": {
            "n": public["valid_pairs"],
            "mean_difference": public["mean_difference"],
            "confidence_interval_95": public["confidence_interval_95"],
            "p_value": public["p_value_two_sided"],
            "direction": public_direction,
        },
        "direction": (
            "replicated"
            if synthetic_direction == public_direction and public_direction != "neutral"
            else "divergent"
            if "neutral" not in {synthetic_direction, public_direction}
            else "inconclusive_neutral_external"
        ),
        "comparability": "partial",
        "limitations": [
            "MovieLens ratings are observed human interactions, not declared profiles.",
            "B4/B5 are genre-only adaptations and not identical to the synthetic methods.",
            "MovieLens 100K is historical and may not represent current consumption.",
            "Samples remain separate and are never pooled.",
        ],
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot(report: Mapping[str, Any], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    values = [
        report["contrasts"]["B5_minus_B4"]["left"]["mean"],
        report["contrasts"]["B5_minus_B4"]["right"]["mean"],
        report["contrasts"]["B5_minus_B0"]["left"]["mean"],
    ]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.bar(("B4", "B5", "B0"), values, color=("#4c78a8", "#f58518", "#54a24b"))
    axis.set(ylabel="NDCG@5", title="Validação externa — MovieLens 100K")
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def execute(
    config_path: Path,
    archive_path: Path,
    synthetic_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Saída existente não será sobrescrita: {output_dir}")
    config = load_study_config(config_path)
    archive = fetch_archive(config, archive_path)
    ratings, movies, readme = load_movielens(archive)
    if "100,000 ratings" not in readme and "100000 ratings" not in readme:
        raise ExternalValidationError("README não confirma a versão MovieLens 100K.")
    external, rows, split = evaluate_external(ratings, movies, config)
    comparison = transportability(external, synthetic_manifest)
    output_dir.mkdir(parents=True)
    audit = {
        "dataset": config["dataset"],
        "archive_md5": _hash(archive, "md5"),
        "archive_sha256": _hash(archive),
        "ratings": len(ratings),
        "users": len({row.user_id for row in ratings}),
        "movies": len(movies),
        "minimum_timestamp": min(row.timestamp for row in ratings),
        "maximum_timestamp": max(row.timestamp for row in ratings),
        "raw_data_redistributed": False,
    }
    for name, value in (
        ("dataset_audit.json", audit),
        ("split_manifest.json", split),
        ("coverage_report.json", external["coverage"]),
        ("external_report.json", external),
        ("transportability.json", comparison),
    ):
        (output_dir / name).write_text(
            json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
    _write_csv(output_dir / "external_user_results.csv", rows)
    summaries = []
    for method in ("B0", "B4", "B5"):
        values = [row[f"{method.lower()}_ndcg_at_5"] for row in rows]
        summaries.append({"method": method, **describe(values)})
    _write_csv(output_dir / "external_method_summary.csv", summaries)
    _write_csv(output_dir / "synthetic_vs_public.csv", [
        {"source": source, **value}
        for source, value in (
            ("synthetic", comparison["synthetic"]),
            ("public", comparison["public"]),
        )
    ])
    _plot(external, output_dir / "external_ndcg.png")
    files = {
        path.name: _hash(path) for path in sorted(output_dir.iterdir()) if path.is_file()
    }
    (output_dir / "evidence.manifest.json").write_text(
        json.dumps({
            "schema_version": "1.0",
            "study": config["study"],
            "config_sha256": _hash(config_path),
            "synthetic_manifest_sha256": _hash(synthetic_manifest),
            "raw_archive_sha256": _hash(archive),
            "raw_archive_redistributed": False,
            "files": files,
        }, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "external": external,
        "transportability": comparison,
        "output_dir": str(output_dir),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--synthetic-manifest", type=Path, default=DEFAULT_SYNTHETIC_MANIFEST)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_study_config(args.config)
    output = args.output_dir or (
        PROJECT_ROOT / "results/derived/specialized_v1/external_validation"
        / config["study"]
    )
    report = execute(args.config, args.archive, args.synthetic_manifest, output)
    print(json.dumps({
        "output_dir": report["output_dir"],
        "external_H1_decision": report["external"]["external_H1_decision"],
        "transportability": report["transportability"]["direction"],
        "public_effect": report["transportability"]["public"]["mean_difference"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
