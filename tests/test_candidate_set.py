import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cinebot_ml.ranking.candidates import (
    CandidateSetValidationError,
    EligibilityPolicy,
    build_candidate_set,
    build_candidate_set_from_config,
)
from cinebot_ml.ranking.contracts import ContractValidationError, RecommendationRequest


def movie(movie_id, **changes):
    value = {
        "id": movie_id,
        "titulo": f"Filme {movie_id}",
        "ano": "2020",
        "perfil": "drama",
        "genero": "Drama",
        "generos_secundarios": [],
        "nota": 7.0,
        "votos": 100,
    }
    value.update(changes)
    return value


def request(**changes):
    value = {
        "experiment_id": "experiment-v1.0__b0__c0__p0__s42",
        "protocol_version": "protocol-v1.0",
        "method": "B0",
        "method_version": "1.0",
        "condition": "C0",
        "profile": "P0",
        "seed": 42,
        "logical_timestamp": "2026-09-12T20:00:00Z",
        "candidate_movie_ids": (),
        "k": 5,
    }
    value.update(changes)
    return RecommendationRequest(**value)


def candidate_set(catalog, req=None, policy=None):
    return build_candidate_set(
        catalog,
        req or request(),
        catalog_path="data/filmes.json",
        catalog_sha256="a" * 64,
        policy=policy,
    )


class CandidateSetTests(unittest.TestCase):
    def test_catalogo_vazio_e_bloqueado(self):
        with self.assertRaisesRegex(CandidateSetValidationError, "vazio"):
            candidate_set([])

    def test_ids_duplicados_sao_bloqueados(self):
        with self.assertRaisesRegex(CandidateSetValidationError, "duplicado"):
            candidate_set([movie(1), movie(1)])

    def test_itens_invalidos_sao_excluidos_com_motivo(self):
        result = candidate_set(
            [movie(1), movie(None), movie(2, titulo=""), movie(3, ano="inválido"), movie(4, perfil="", genero="")]
        )
        self.assertEqual(result.movie_ids, (1,))
        self.assertEqual(
            result.excluded_counts,
            {"invalid_id": 1, "missing_title": 1, "invalid_year": 1, "missing_genre": 1},
        )

    def test_ids_sao_produzidos_em_ordem_canonica(self):
        result = candidate_set([movie(20), movie(3), movie(11)])
        self.assertEqual(result.movie_ids, (3, 11, 20))

    def test_candidate_set_preenche_requisicao_final(self):
        initial = request()
        result = candidate_set([movie(2), movie(1)], initial)
        final = result.attach_to_request(initial)
        self.assertEqual(final.candidate_movie_ids, (1, 2))

    def test_candidatos_preexistentes_divergentes_sao_bloqueados(self):
        with self.assertRaisesRegex(CandidateSetValidationError, "diverge"):
            candidate_set([movie(1), movie(2)], request(candidate_movie_ids=(2, 1)))

    def test_filme_inexistente_no_historico_e_bloqueado(self):
        req = request(history=({"timestamp": "2026-09-12T19:00:00Z", "movie_id": 99},))
        with self.assertRaisesRegex(CandidateSetValidationError, "inexistente"):
            candidate_set([movie(1)], req)

    def test_item_consumido_e_excluido(self):
        req = request(history=({"timestamp": "2026-09-12T19:00:00Z", "movie_id": 2},))
        result = candidate_set([movie(1), movie(2)], req)
        self.assertEqual(result.movie_ids, (1,))
        self.assertEqual(result.excluded_counts["already_presented_or_rated"], 1)

    def test_politica_pode_preservar_item_consumido_explicitamente(self):
        req = request(history=({"timestamp": "2026-09-12T19:00:00Z", "movie_id": 2},))
        policy = EligibilityPolicy(exclude_history_items=False)
        self.assertEqual(candidate_set([movie(1), movie(2)], req, policy).movie_ids, (1, 2))

    def test_feedback_futuro_e_bloqueado_pelo_contrato(self):
        with self.assertRaisesRegex(ContractValidationError, "não é anterior"):
            request(history=({"timestamp": "2026-09-12T21:00:00Z", "movie_id": 1},))

    def test_filtros_explicitos_sao_aplicados_em_conjunto(self):
        catalog = [
            movie(1, perfil="drama", genero="Drama", ano="2022", votos=10000),
            movie(2, perfil="acao", genero="Ação", ano="2005", votos=20),
            movie(3, perfil="acao", genero="Ação", ano="2021", votos=1),
        ]
        req = request(
            profile_data={
                "candidate_filters": {"genres": ["ação"], "decade": "moderno", "popularity": "joia escondida"}
            }
        )
        result = candidate_set(catalog, req)
        self.assertEqual(result.movie_ids, (3,))
        self.assertEqual(result.filters, {"genres": ["acao"], "decade": "moderno", "popularity": "joia_escondida"})

    def test_filtro_de_popularidade_exclui_metadado_invalido(self):
        req = request(profile_data={"candidate_filters": {"popularity": "popular"}})
        result = candidate_set([movie(1), movie(2, votos="inválido")], req)
        self.assertEqual(result.movie_ids, (1,))
        self.assertEqual(result.excluded_counts["invalid_popularity_metadata"], 1)

    def test_filtros_ausentes_nao_sao_inferidos_do_perfil(self):
        req = request(profile_data={"ranked_genres": ["drama"]})
        result = candidate_set([movie(1, perfil="drama"), movie(2, perfil="acao")], req)
        self.assertEqual(result.movie_ids, (1, 2))
        self.assertEqual(result.filters, {})

    def test_filtro_sem_resultado_retorna_conjunto_vazio_auditavel(self):
        req = request(profile_data={"candidate_filters": {"genres": ["terror"]}})
        result = candidate_set([movie(1, perfil="drama")], req)
        self.assertEqual(result.movie_ids, ())
        self.assertEqual(result.excluded_counts["genre_filter"], 1)

    def test_filtro_desconhecido_e_bloqueado(self):
        req = request(profile_data={"candidate_filters": {"director": "Pessoa"}})
        with self.assertRaisesRegex(CandidateSetValidationError, "desconhecido"):
            candidate_set([movie(1)], req)

    def test_mesmo_contexto_produz_mesmo_hash(self):
        catalog = [movie(2), movie(1)]
        first = candidate_set(catalog)
        second = candidate_set(list(reversed(catalog)))
        self.assertEqual(first.candidate_set_id, second.candidate_set_id)
        self.assertEqual(first.to_manifest(), second.to_manifest())

    def test_alteracao_de_filtro_ou_elegibilidade_muda_hash(self):
        catalog = [movie(1, perfil="drama"), movie(2, perfil="acao")]
        unfiltered = candidate_set(catalog)
        filtered = candidate_set(
            catalog,
            request(profile_data={"candidate_filters": {"genres": ["drama"]}}),
        )
        changed_policy = candidate_set(catalog, policy=EligibilityPolicy(require_year=False))
        self.assertNotEqual(unfiltered.candidate_set_id, filtered.candidate_set_id)
        self.assertNotEqual(unfiltered.candidate_set_id, changed_policy.candidate_set_id)

    def test_manifesto_resume_catalogo_filtros_e_exclusoes(self):
        req = request(profile_data={"candidate_filters": {"genres": ["drama"]}})
        manifest = candidate_set([movie(1), movie(2, perfil="acao", genero="Ação")], req).to_manifest()
        self.assertEqual(manifest["candidate_count"], 1)
        self.assertEqual(manifest["total_catalog_items"], 2)
        self.assertEqual(manifest["catalog"]["path"], "data/filmes.json")
        self.assertEqual(manifest["excluded_counts"], {"genre_filter": 1})

    def test_carrega_catalogo_pelo_caminho_relativo_da_configuracao(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_path = root / "data" / "catalog.json"
            catalog_path.parent.mkdir()
            payload = {"perfis": {"drama": [movie(7)]}}
            catalog_path.write_text(json.dumps(payload), encoding="utf-8")
            for relative in ("datasets/data.csv", "datasets/feedback.csv"):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("header\n", encoding="utf-8")
            config = {
                "schema_version": "1.0",
                "experiment": {"name": "test", "version": "1", "protocol_version": "protocol-v1.0", "frozen": False},
                "seeds": [42], "k_values": [5],
                "paths": {"catalog": "data/catalog.json", "dataset": "datasets/data.csv", "feedback": "datasets/feedback.csv", "output_root": "results", "split_manifests": "results/manifests/splits", "manifests": "results/manifests/executions", "raw": "results/raw", "tables": "results/tables", "figures": "results/figures", "reports": "results/reports"},
                "data_versions": {"catalog": "v1", "dataset": "v1", "feedback": "v1"},
                "methods": {"enabled": ["B0"], "optional": ["B6"]},
                "conditions": ["C0"], "profiles": ["P0"],
                "metrics": {"primary": ["ndcg_at_k"], "secondary": []},
                "splits": {"strategy": "deterministic_group_stratification_search", "temporal_strategy": "chronological_replay", "train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "attempts": 1},
                "execution": {"deterministic_repetitions": 1, "stochastic_repetitions": 1, "unit": "profile", "fail_on_dirty_worktree": False},
            }
            config_path = root / "configs" / "experiment.yaml"
            config_path.parent.mkdir()
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            result = build_candidate_set_from_config(request(), config_path, project_root=root)
            self.assertEqual(result.movie_ids, (7,))
            self.assertEqual(result.catalog_sha256, hashlib.sha256(catalog_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
