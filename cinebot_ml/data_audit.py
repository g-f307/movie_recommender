"""Auditoria reproduzível e somente leitura dos dados do Movie Recommender."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from cinebot_ml.config import DATASET_PATH, DEFAULT_DATA_PATH, FEEDBACK_PATH, PROJECT_ROOT


AUDIT_VERSION = "1.0"
DEFAULT_JSON_OUTPUT = PROJECT_ROOT / "results" / "reports" / "data_audit.json"
DEFAULT_MARKDOWN_OUTPUT = PROJECT_ROOT / "docs" / "research" / "data_audit.md"
DEFAULT_DICTIONARY_OUTPUT = PROJECT_ROOT / "docs" / "research" / "data_dictionary.md"

DATASET_COLUMNS = [
    "movie_id",
    "titulo",
    "sinopse",
    "genero",
    "generos_texto",
    "diretor",
    "ano",
    "nota",
    "votos",
    "duracao",
    "streaming_count",
    "release_period",
    "poster",
    "url",
    "sinopse_word_count",
    "keyword_count",
    "secondary_genre_count",
    "has_streaming",
    "has_poster",
    "pref_1",
    "pref_2",
    "pref_3",
    "decade_pref",
    "popularity_pref",
    "ranked_profile_text",
    "primary_match_pref_1",
    "secondary_match_pref_2",
    "secondary_match_pref_3",
    "ranked_match_count",
    "genre_overlap_count",
    "matches_decade_pref",
    "popular_score",
    "hidden_gem_score",
    "matches_popularity_pref",
    "popularity_match_score",
    "score_heuristico",
    "relevante",
    "label_source",
    "feedback_applied",
    "feedback_scope",
    "feedback_votes",
]

FEEDBACK_COLUMNS = [
    "user_id",
    "movie_id",
    "pref_1",
    "pref_2",
    "pref_3",
    "decade_pref",
    "popularity_pref",
    "feedback",
    "label",
    "timestamp",
]

CATALOG_FIELDS = [
    "id",
    "titulo",
    "sinopse",
    "diretor",
    "ano",
    "nota",
    "votos",
    "duracao",
    "streaming",
    "poster",
    "palavras_chave",
]

DATA_DICTIONARY: dict[str, tuple[str, str, str]] = {
    "movie_id": ("identificador", "catálogo TMDB", "Identificar o filme de forma estável."),
    "titulo": ("texto", "catálogo", "Título de exibição e auditoria do item."),
    "sinopse": ("texto", "catálogo enriquecido", "Representação textual do conteúdo."),
    "genero": ("categoria", "catálogo normalizado", "Gênero principal do filme."),
    "generos_texto": ("texto", "catálogo derivado", "Gêneros concatenados para vetorização."),
    "diretor": ("categoria/texto", "catálogo", "Representar afinidade com direção."),
    "ano": ("inteiro", "catálogo", "Ano de lançamento."),
    "nota": ("decimal", "TMDB", "Nota agregada do filme."),
    "votos": ("inteiro", "TMDB", "Quantidade de votos usada em popularidade."),
    "duracao": ("inteiro", "catálogo", "Duração em minutos."),
    "streaming_count": ("inteiro", "catálogo derivado", "Quantidade de provedores de streaming."),
    "release_period": ("categoria", "ano derivado", "Período usado no filtro de década."),
    "poster": ("URL/texto", "catálogo", "Imagem de apresentação do filme."),
    "url": ("URL/texto", "catálogo", "Referência pública do item."),
    "sinopse_word_count": ("inteiro", "sinopse derivada", "Medir disponibilidade textual."),
    "keyword_count": ("inteiro", "palavras-chave derivadas", "Medir densidade de metadados."),
    "secondary_genre_count": ("inteiro", "gêneros derivados", "Quantidade de gêneros secundários."),
    "has_streaming": ("binário", "streaming derivado", "Indicar disponibilidade de streaming."),
    "has_poster": ("binário", "poster derivado", "Indicar disponibilidade de poster."),
    "pref_1": ("categoria", "perfil gerado", "Primeiro gênero declarado."),
    "pref_2": ("categoria", "perfil gerado", "Segundo gênero declarado."),
    "pref_3": ("categoria", "perfil gerado", "Terceiro gênero declarado."),
    "decade_pref": ("categoria", "perfil gerado", "Preferência de período de lançamento."),
    "popularity_pref": ("categoria", "perfil gerado", "Preferência por popularidade."),
    "ranked_profile_text": ("texto", "perfil derivado", "Representação textual do perfil."),
    "primary_match_pref_1": ("binário", "item e perfil", "Aderência ao primeiro gênero."),
    "secondary_match_pref_2": ("binário", "item e perfil", "Aderência ao segundo gênero."),
    "secondary_match_pref_3": ("binário", "item e perfil", "Aderência ao terceiro gênero."),
    "ranked_match_count": ("inteiro", "item e perfil", "Quantidade de correspondências ranqueadas."),
    "genre_overlap_count": ("inteiro", "item e perfil", "Interseção entre gêneros do item e perfil."),
    "matches_decade_pref": ("binário", "item e perfil", "Aderência ao filtro de década."),
    "popular_score": ("decimal", "metadados derivados", "Score heurístico de popularidade."),
    "hidden_gem_score": ("decimal", "metadados derivados", "Score heurístico de joia escondida."),
    "matches_popularity_pref": ("binário", "item e perfil", "Aderência ao perfil de popularidade."),
    "popularity_match_score": ("decimal", "item e perfil", "Intensidade da aderência de popularidade."),
    "score_heuristico": ("decimal", "regra de supervisão", "Proxy que origina o rótulo inicial."),
    "relevante": ("binário", "proxy ou feedback", "Alvo supervisionado; origem em label_source."),
    "label_source": ("categoria", "pipeline de supervisão", "Distinguir proxy heurístico e feedback."),
    "feedback_applied": ("binário", "feedback derivado", "Indicar sobrescrita por feedback."),
    "feedback_scope": ("categoria", "feedback derivado", "Escopo contextual da sobrescrita."),
    "feedback_votes": ("inteiro", "feedback derivado", "Eventos agregados no rótulo."),
}


class AuditValidationError(ValueError):
    """Indica entrada ausente ou incompatível com o contrato de auditoria."""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) == 0
    return str(value).strip().lower() in {"", "nan", "none", "null", "<na>"}


def _counter(counter: Counter) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items(), key=lambda item: str(item[0]))}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_metadata(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _release_period(movie: dict[str, Any]) -> str:
    try:
        year = int(movie.get("ano") or 0)
    except (TypeError, ValueError):
        return "unknown"
    if year >= 2010:
        return "moderno"
    if year >= 2000:
        return "anos_2000"
    if year > 0:
        return "antes_2000"
    return "unknown"


def _popularity_bucket(movie: dict[str, Any], votes: list[int]) -> str:
    try:
        rating = float(movie.get("nota") or 0)
        movie_votes = max(int(movie.get("votos") or 0), 0)
    except (TypeError, ValueError):
        return "unknown"
    if not votes:
        return "unknown"
    rank = sum(value <= movie_votes for value in votes) / len(votes)
    if rating >= 7.4 and rank <= 0.45:
        return "joia_escondida"
    if rank >= 0.60:
        return "popular"
    max_votes = max(max(votes), 1)
    vote_scale = math.log1p(movie_votes) / math.log1p(max_votes)
    popular_score = 0.45 * vote_scale + 0.25 * rank + 0.30 * min(max(rating / 10, 0), 1)
    hidden_score = 0.45 * min(max(rating / 10, 0), 1) + 0.30 * (1 - vote_scale) + 0.25 * (1 - rank)
    return "popular" if popular_score >= hidden_score else "joia_escondida"


def audit_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AuditValidationError(f"Catálogo não encontrado: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AuditValidationError(f"Catálogo inválido: {exc}") from exc

    profiles = payload.get("perfis", {}) if isinstance(payload, dict) else {}
    if not isinstance(profiles, dict):
        raise AuditValidationError("O campo 'perfis' do catálogo deve ser um objeto.")

    occurrences: list[tuple[str, dict[str, Any]]] = []
    for profile, movies in profiles.items():
        if not isinstance(movies, list):
            raise AuditValidationError(f"O perfil '{profile}' deve conter uma lista de filmes.")
        for movie in movies:
            if not isinstance(movie, dict):
                raise AuditValidationError(f"O perfil '{profile}' contém um filme inválido.")
            occurrences.append((str(profile), movie))

    profile_counts = Counter(profile for profile, _ in occurrences)
    movies_by_id: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    missing_ids = 0
    for profile, movie in occurrences:
        movie_id = movie.get("id")
        if _is_missing(movie_id):
            missing_ids += 1
            continue
        movies_by_id[str(movie_id)].append((profile, movie))

    unique_movies = [items[0][1] for _, items in sorted(movies_by_id.items())]
    duplicate_profiles = {
        movie_id: sorted({profile for profile, _ in items})
        for movie_id, items in movies_by_id.items()
        if len({profile for profile, _ in items}) > 1
    }
    conflicting_duplicates = []
    for movie_id, items in sorted(movies_by_id.items()):
        fingerprints = {
            json.dumps(movie, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for _, movie in items
        }
        if len(fingerprints) > 1:
            conflicting_duplicates.append(movie_id)

    missing_counts = Counter()
    for movie in unique_movies:
        for field in CATALOG_FIELDS:
            if _is_missing(movie.get(field)):
                missing_counts[field] += 1

    unique_count = len(unique_movies)
    coverage_fields = {
        "sinopse": "sinopse",
        "diretor": "diretor",
        "streaming": "streaming",
        "poster": "poster",
        "palavras_chave": "palavras_chave",
    }
    coverage = {
        label: {
            "present": unique_count - int(missing_counts[field]),
            "total": unique_count,
            "rate": round((unique_count - int(missing_counts[field])) / unique_count, 6)
            if unique_count
            else 0.0,
        }
        for label, field in coverage_fields.items()
    }

    decade_distribution = Counter(_release_period(movie) for movie in unique_movies)
    votes = []
    for movie in unique_movies:
        try:
            votes.append(max(int(movie.get("votos") or 0), 0))
        except (TypeError, ValueError):
            continue
    votes_by_period: dict[str, list[int]] = defaultdict(list)
    for movie in unique_movies:
        try:
            votes_by_period[_release_period(movie)].append(max(int(movie.get("votos") or 0), 0))
        except (TypeError, ValueError):
            continue
    popularity_distribution = Counter(
        _popularity_bucket(movie, votes_by_period[_release_period(movie)]) for movie in unique_movies
    )

    genre_distribution = Counter()
    for items in movies_by_id.values():
        for profile in {profile for profile, _ in items}:
            genre_distribution[profile] += 1

    top_decile_share = 0.0
    if votes:
        ordered_votes = sorted(votes, reverse=True)
        top_size = max(1, math.ceil(len(ordered_votes) * 0.10))
        total_votes = sum(ordered_votes)
        top_decile_share = sum(ordered_votes[:top_size]) / total_votes if total_votes else 0.0

    return {
        "total_occurrences": len(occurrences),
        "unique_movies": unique_count,
        "occurrences_without_id": missing_ids,
        "profile_occurrences": _counter(profile_counts),
        "unique_movies_by_profile": _counter(genre_distribution),
        "decade_distribution": _counter(decade_distribution),
        "popularity_distribution": _counter(popularity_distribution),
        "duplicates": {
            "movies_in_multiple_profiles": len(duplicate_profiles),
            "duplicate_occurrences": sum(len(items) - 1 for items in movies_by_id.values()),
            "conflicting_movie_records": len(conflicting_duplicates),
            "movie_ids_in_multiple_profiles": sorted(duplicate_profiles),
            "conflicting_movie_ids": conflicting_duplicates,
        },
        "missing": {
            field: {
                "count": int(missing_counts[field]),
                "rate": round(int(missing_counts[field]) / unique_count, 6) if unique_count else 0.0,
            }
            for field in CATALOG_FIELDS
        },
        "coverage": coverage,
        "popularity_bias": {
            "top_10_percent_vote_share": round(top_decile_share, 6),
            "interpretation": "Participação dos 10% de filmes com mais votos no total de votos do catálogo.",
        },
    }


def _validated_reader(path: Path, required_columns: list[str]) -> tuple[Any, csv.DictReader]:
    if not path.exists():
        raise AuditValidationError(f"Arquivo não encontrado: {path}")
    handle = path.open("r", encoding="utf-8", newline="")
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames or []
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        handle.close()
        raise AuditValidationError(f"Colunas obrigatórias ausentes em {path.name}: {', '.join(missing)}")
    return handle, reader


def _limited_rows(reader: Iterable[dict[str, str]], sample_rows: int | None) -> Iterable[dict[str, str]]:
    for index, row in enumerate(reader):
        if sample_rows is not None and index >= sample_rows:
            break
        yield row


def audit_dataset(path: Path, sample_rows: int | None = None) -> dict[str, Any]:
    handle, reader = _validated_reader(path, DATASET_COLUMNS)
    total_rows = 0
    movie_ids: set[str] = set()
    missing_counts = Counter()
    distributions = {
        "genre": Counter(),
        "decade": Counter(),
        "popularity_preference": Counter(),
        "class": Counter(),
        "label_source": Counter(),
        "feedback_scope": Counter(),
    }
    coverage_present = Counter()

    try:
        for row in _limited_rows(reader, sample_rows):
            total_rows += 1
            movie_id = row.get("movie_id")
            if not _is_missing(movie_id):
                movie_ids.add(str(movie_id))
            for column in DATASET_COLUMNS:
                if _is_missing(row.get(column)):
                    missing_counts[column] += 1
            distributions["genre"][row.get("genero") or "<missing>"] += 1
            distributions["decade"][row.get("release_period") or "<missing>"] += 1
            distributions["popularity_preference"][row.get("popularity_pref") or "<missing>"] += 1
            distributions["class"][row.get("relevante") or "<missing>"] += 1
            distributions["label_source"][row.get("label_source") or "<missing>"] += 1
            distributions["feedback_scope"][row.get("feedback_scope") or "<missing>"] += 1

            for field in ("sinopse", "diretor", "poster"):
                if not _is_missing(row.get(field)):
                    coverage_present[field] += 1
            for field in ("streaming_count", "keyword_count"):
                try:
                    if float(row.get(field) or 0) > 0:
                        coverage_present[field] += 1
                except (TypeError, ValueError):
                    pass
    finally:
        handle.close()

    class_counts = distributions["class"]
    numeric_class_counts = [class_counts[key] for key in ("0", "1") if class_counts[key] > 0]
    imbalance_ratio = None
    if len(numeric_class_counts) == 2:
        imbalance_ratio = max(numeric_class_counts) / min(numeric_class_counts)

    mode = "sample" if sample_rows is not None else "full"
    return {
        "mode": mode,
        "requested_sample_rows": sample_rows,
        "rows_audited": total_rows,
        "unique_movies_in_rows": len(movie_ids),
        "distributions": {name: _counter(values) for name, values in distributions.items()},
        "missing": {
            column: {
                "count": int(missing_counts[column]),
                "rate": round(int(missing_counts[column]) / total_rows, 6) if total_rows else 0.0,
            }
            for column in DATASET_COLUMNS
        },
        "coverage": {
            field: {
                "present": int(coverage_present[field]),
                "total": total_rows,
                "rate": round(int(coverage_present[field]) / total_rows, 6) if total_rows else 0.0,
            }
            for field in ("sinopse", "diretor", "streaming_count", "poster", "keyword_count")
        },
        "class_balance": {
            "positive_rate": round(class_counts["1"] / total_rows, 6) if total_rows else 0.0,
            "majority_to_minority_ratio": round(imbalance_ratio, 6) if imbalance_ratio is not None else None,
            "is_imbalanced_over_3_to_1": bool(imbalance_ratio and imbalance_ratio >= 3),
        },
    }


def audit_feedback(path: Path, sample_rows: int | None = None) -> dict[str, Any]:
    handle, reader = _validated_reader(path, FEEDBACK_COLUMNS)
    total_rows = 0
    user_ids: set[str] = set()
    movie_ids: set[str] = set()
    missing_counts = Counter()
    feedback_distribution = Counter()
    label_distribution = Counter()

    try:
        for row in _limited_rows(reader, sample_rows):
            total_rows += 1
            user_id = row.get("user_id")
            movie_id = row.get("movie_id")
            if not _is_missing(user_id):
                user_ids.add(str(user_id))
            if not _is_missing(movie_id):
                movie_ids.add(str(movie_id))
            for column in FEEDBACK_COLUMNS:
                if _is_missing(row.get(column)):
                    missing_counts[column] += 1
            feedback_distribution[row.get("feedback") or "<missing>"] += 1
            label_distribution[row.get("label") or "<missing>"] += 1
    finally:
        handle.close()

    return {
        "mode": "sample" if sample_rows is not None else "full",
        "requested_sample_rows": sample_rows,
        "events": total_rows,
        "unique_users": len(user_ids),
        "unique_movies": len(movie_ids),
        "feedback_distribution": _counter(feedback_distribution),
        "label_distribution": _counter(label_distribution),
        "missing": {
            column: {
                "count": int(missing_counts[column]),
                "rate": round(int(missing_counts[column]) / total_rows, 6) if total_rows else 0.0,
            }
            for column in FEEDBACK_COLUMNS
        },
        "privacy": {
            "user_identifiers_exported": False,
            "note": "O relatório contém apenas contagem agregada de usuários.",
        },
    }


def run_audit(
    catalog_path: Path = DEFAULT_DATA_PATH,
    dataset_path: Path = DATASET_PATH,
    feedback_path: Path = FEEDBACK_PATH,
    sample_rows: int | None = None,
) -> dict[str, Any]:
    if sample_rows is not None and sample_rows <= 0:
        raise AuditValidationError("sample_rows deve ser maior que zero.")
    inputs = [catalog_path, dataset_path, feedback_path]
    for path in inputs:
        if not path.exists():
            raise AuditValidationError(f"Arquivo não encontrado: {path}")

    return {
        "audit_version": AUDIT_VERSION,
        "mode": "sample" if sample_rows is not None else "full",
        "sample_rows": sample_rows,
        "inputs": {
            "catalog": _input_metadata(catalog_path),
            "dataset": _input_metadata(dataset_path),
            "feedback": _input_metadata(feedback_path),
        },
        "catalog": audit_catalog(catalog_path),
        "dataset": audit_dataset(dataset_path, sample_rows),
        "feedback": audit_feedback(feedback_path, sample_rows),
        "interpretation": {
            "heuristic_labels_are_user_judgments": False,
            "sample_is_full_dataset": sample_rows is None,
            "notes": [
                "Rótulos heuristic_proxy medem aderência ao proxy e não satisfação do usuário.",
                "Filmes repetidos entre perfis são ocorrências do mesmo movie_id, não itens únicos.",
                "Relatórios de amostra descrevem somente as primeiras linhas auditadas.",
            ],
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    catalog = report["catalog"]
    dataset = report["dataset"]
    feedback = report["feedback"]
    sources = dataset["distributions"]["label_source"]
    classes = dataset["distributions"]["class"]

    lines = [
        "# Auditoria dos dados",
        "",
        f"**Versão da auditoria:** {report['audit_version']}",
        f"**Modo:** {report['mode']}",
        "",
        "Este relatório é gerado automaticamente por `python -m cinebot_ml.data_audit`.",
        "Não contém identificadores de usuários nem valores da `.env`.",
        "",
        "## Como reproduzir",
        "",
        "Auditoria completa:",
        "",
        "```bash",
        "python -m cinebot_ml.data_audit",
        "```",
        "",
        "Auditoria rápida para CI:",
        "",
        "```bash",
        "python -m cinebot_ml.data_audit --sample-rows 1000 \\",
        "  --json-output /tmp/data_audit.json \\",
        "  --markdown-output /tmp/data_audit.md \\",
        "  --dictionary-output /tmp/data_dictionary.md",
        "```",
        "",
        "## Resumo",
        "",
        "| Indicador | Valor |",
        "|---|---:|",
        f"| Ocorrências no catálogo | {catalog['total_occurrences']} |",
        f"| Filmes únicos | {catalog['unique_movies']} |",
        f"| Filmes em múltiplos perfis | {catalog['duplicates']['movies_in_multiple_profiles']} |",
        f"| Linhas auditadas do dataset | {dataset['rows_audited']} |",
        f"| Eventos de feedback | {feedback['events']} |",
        f"| Usuários distintos, somente contagem | {feedback['unique_users']} |",
        "",
        "## Distribuições do catálogo",
        "",
        _markdown_table(catalog["unique_movies_by_profile"], "Gênero/perfil", "Filmes"),
        "",
        _markdown_table(catalog["decade_distribution"], "Década", "Filmes"),
        "",
        _markdown_table(catalog["popularity_distribution"], "Popularidade", "Filmes"),
        "",
        "## Cobertura do catálogo",
        "",
        "| Campo | Presentes | Total | Cobertura |",
        "|---|---:|---:|---:|",
    ]
    for field, values in catalog["coverage"].items():
        lines.append(f"| {field} | {values['present']} | {values['total']} | {values['rate']:.2%} |")

    lines.extend(
        [
            "",
            "## Valores ausentes no catálogo",
            "",
            "| Campo | Ausentes | Taxa |",
            "|---|---:|---:|",
        ]
    )
    for field, values in catalog["missing"].items():
        lines.append(f"| {field} | {values['count']} | {values['rate']:.2%} |")

    lines.extend(
        [
            "",
            "## Cobertura do dataset",
            "",
            "| Campo | Presentes | Total | Cobertura |",
            "|---|---:|---:|---:|",
        ]
    )
    for field, values in dataset["coverage"].items():
        lines.append(f"| {field} | {values['present']} | {values['total']} | {values['rate']:.2%} |")

    lines.extend(
        [
            "",
            "## Supervisão e classes",
            "",
            _markdown_table(sources, "Origem do rótulo", "Linhas"),
            "",
            _markdown_table(classes, "Classe", "Linhas"),
            "",
            f"Taxa positiva: **{dataset['class_balance']['positive_rate']:.2%}**.",
            f"Razão maioria/minoria: **{dataset['class_balance']['majority_to_minority_ratio']}**.",
            "",
            "> `heuristic_proxy` é supervisão calculada por regra e não representa julgamento de usuário.",
            "",
            "## Feedback anonimizado",
            "",
            _markdown_table(feedback["feedback_distribution"], "Feedback", "Eventos"),
            "",
            "O relatório exporta somente contagens agregadas; nenhum `user_id` é incluído.",
            "",
            "## Duplicidade e viés",
            "",
            f"- Ocorrências duplicadas entre perfis: {catalog['duplicates']['duplicate_occurrences']}.",
            f"- Registros conflitantes do mesmo filme: {catalog['duplicates']['conflicting_movie_records']}.",
            f"- Participação do decil superior nos votos: {catalog['popularity_bias']['top_10_percent_vote_share']:.2%}.",
            "",
            "## Limitações",
            "",
            "- O catálogo é filtrado na coleta e não representa todo o universo de filmes.",
            "- Distribuições em modo `sample` não substituem a auditoria completa.",
            "- Métricas sobre rótulos heurísticos avaliam reprodução do proxy.",
            "- A auditoria descreve os dados; ela não corrige valores automaticamente.",
        ]
    )
    return "\n".join(lines) + "\n"


def _markdown_table(values: dict[str, int], first_header: str, second_header: str) -> str:
    lines = [f"| {first_header} | {second_header} |", "|---|---:|"]
    if not values:
        lines.append("| Sem dados | 0 |")
    else:
        lines.extend(f"| {key} | {value} |" for key, value in values.items())
    return "\n".join(lines)


def render_data_dictionary() -> str:
    lines = [
        "# Dicionário de dados",
        "",
        "## Dataset supervisionado",
        "",
        "| Campo | Tipo lógico | Origem | Finalidade |",
        "|---|---|---|---|",
    ]
    for column in DATASET_COLUMNS:
        logical_type, source, purpose = DATA_DICTIONARY[column]
        lines.append(f"| `{column}` | {logical_type} | {source} | {purpose} |")

    lines.extend(
        [
            "",
            "## Catálogo",
            "",
            "| Campo | Tipo lógico | Origem | Finalidade |",
            "|---|---|---|---|",
            "| `id` | identificador | TMDB | Identificar o filme único entre perfis. |",
            "| `titulo` | texto | TMDB/BotCity | Apresentar e auditar o item. |",
            "| `sinopse` | texto | TMDB/BotCity | Representar o conteúdo textual. |",
            "| `diretor` | texto | créditos do TMDB | Representar direção e personalização. |",
            "| `ano` | inteiro | data de lançamento | Derivar a década do filme. |",
            "| `nota` | decimal | TMDB | Apoiar qualidade e popularidade. |",
            "| `votos` | inteiro | TMDB | Calcular popularidade relativa. |",
            "| `duracao` | inteiro | TMDB | Registrar duração em minutos. |",
            "| `streaming` | lista de textos | provedores TMDB/BR | Informar disponibilidade. |",
            "| `poster` | URL/texto | TMDB | Apresentar imagem do item. |",
            "| `palavras_chave` | lista de textos | TMDB | Enriquecer conteúdo e personalização. |",
            "| `perfil` | categoria | coleta | Registrar o perfil de origem da ocorrência. |",
            "| `generos_secundarios` | lista de categorias | TMDB normalizado | Representar gêneros adicionais. |",
            "",
            "## Feedback",
            "",
            "| Campo | Tipo lógico | Origem | Finalidade |",
            "|---|---|---|---|",
            "| `user_id` | identificador pseudonimizável | Telegram | Relacionar eventos do mesmo usuário; nunca exportado na auditoria. |",
            "| `movie_id` | identificador | recomendação | Identificar o filme avaliado. |",
            "| `pref_1`–`pref_3` | categorias | usuário | Registrar os gêneros do contexto. |",
            "| `decade_pref` | categoria | usuário | Registrar o filtro de década. |",
            "| `popularity_pref` | categoria | usuário | Registrar o filtro de popularidade. |",
            "| `feedback` | categoria | usuário | `like` ou `dislike`. |",
            "| `label` | binário | feedback derivado | Representar feedback positivo ou negativo. |",
            "| `timestamp` | data/hora UTC | sistema | Ordenar eventos e prevenir uso de informação futura. |",
            "",
            "O catálogo preserva os campos originais por filme dentro de `perfis`. O `id` do TMDB define o item único; aparições em diferentes perfis são ocorrências do mesmo filme.",
            "",
            "## Regras de interpretação",
            "",
            "- `label_source=heuristic_proxy` não é julgamento humano.",
            "- `feedback_exact_context` e `feedback_profile` indicam sobrescrita por feedback agregado.",
            "- Campos derivados devem ser recalculados somente pelo pipeline versionado.",
            "- Identificadores de usuários não podem aparecer em relatórios compartilháveis.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_outputs(
    report: dict[str, Any],
    json_output: Path,
    markdown_output: Path,
    dictionary_output: Path,
) -> None:
    payloads = {
        json_output: json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        markdown_output: render_markdown(report),
        dictionary_output: render_data_dictionary(),
    }
    for path, content in payloads.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--feedback", type=Path, default=FEEDBACK_PATH)
    parser.add_argument("--sample-rows", type=int, default=None)
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON_OUTPUT)
    parser.add_argument("--markdown-output", type=Path, default=DEFAULT_MARKDOWN_OUTPUT)
    parser.add_argument("--dictionary-output", type=Path, default=DEFAULT_DICTIONARY_OUTPUT)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        report = run_audit(args.catalog, args.dataset, args.feedback, args.sample_rows)
        write_outputs(report, args.json_output, args.markdown_output, args.dictionary_output)
    except AuditValidationError as exc:
        raise SystemExit(f"Erro de validação: {exc}") from exc
    print(
        f"Auditoria {report['mode']} concluída: "
        f"{report['dataset']['rows_audited']} linhas, "
        f"{report['catalog']['unique_movies']} filmes únicos."
    )


if __name__ == "__main__":
    main()
