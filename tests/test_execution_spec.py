import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from cinebot_ml.experiments.execution import (
    ExecutionSpecError,
    load_execution_spec,
    prepare_execution_directories,
    validate_execution_spec,
)
from cinebot_ml.experiments.finalize import finalize_versioned_execution
from cinebot_ml.experiments.freeze import audit_execution
from cinebot_ml.experiments.matrix import ExperimentCell, ExperimentMatrix, execute_matrix
from cinebot_ml.experiments.units import write_units


def valid_spec():
    return {
        "schema_version": "1.0",
        "execution": {
            "id": "study-v2",
            "version": "2.0.0",
            "protocol_version": "protocol-v2.0",
            "frozen": True,
            "exposure_policy": "closed-loop-v2",
        },
        "paths": {
            "source_catalog": "data/filmes.json",
            "split_manifest": "results/v2/manifests/splits/movie_id_split.json",
            "catalog": "results/v2/derived/test_catalog.json",
            "units": "results/v2/units",
            "raw": "results/v2/raw",
            "manifests": "results/v2/manifests/executions",
            "reports": "results/v2/reports",
        },
    }


class ExecutionSpecTests(unittest.TestCase):
    def write_spec(self, root: Path, payload=None) -> Path:
        path = root / "configs" / "execution_v2.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(yaml.safe_dump(payload or valid_spec(), sort_keys=False), encoding="utf-8")
        path.with_suffix(".lock.json").write_text(
            json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), encoding="utf-8"
        )
        return path

    def test_configuracao_v2_congelada_identifica_execucao_e_separa_saidas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_spec(root)
            spec = load_execution_spec(path, project_root=root)

        self.assertEqual(spec.execution_id, "study-v2")
        self.assertEqual(spec.version, "2.0.0")
        self.assertEqual(spec.protocol_version, "protocol-v2.0")
        self.assertEqual(len(spec.sha256), 64)
        self.assertTrue(all(path.startswith("results/v2/") for path in spec.output_paths.values()))

    def test_configuracao_de_observacao_e_referenciada_sem_virar_saida(self):
        payload = valid_spec()
        payload["profile_observation_config"] = "configs/agents/profile_observation_v2.yaml"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_execution_spec(self.write_spec(root, payload), project_root=root)

        self.assertEqual(
            spec.profile_observation_config,
            "configs/agents/profile_observation_v2.yaml",
        )
        self.assertNotIn("profile_observation_config", spec.output_paths)

    def test_configuracao_de_exposicao_fechada_e_referenciada_sem_virar_saida(self):
        payload = valid_spec()
        payload["closed_loop_config"] = "configs/experiments/closed_loop_v2.yaml"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_execution_spec(self.write_spec(root, payload), project_root=root)

        self.assertEqual(
            spec.closed_loop_config,
            "configs/experiments/closed_loop_v2.yaml",
        )
        self.assertNotIn("closed_loop_config", spec.output_paths)

    def test_lock_desatualizado_e_rejeitado_antes_de_criar_artefatos(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = self.write_spec(root)
            path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ExecutionSpecError, "lock"):
                load_execution_spec(path, project_root=root)
            self.assertFalse((root / "results").exists())

    def test_preparo_respeita_a_raiz_associada_a_especificacao(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_execution_spec(self.write_spec(root), project_root=root)
            prepare_execution_directories(spec)
            self.assertTrue(spec.resolve_path("raw").is_dir())

    def test_caminhos_oficiais_v1_1_nao_podem_ser_reutilizados_por_v2(self):
        payload = valid_spec()
        payload["paths"]["raw"] = "results/raw/official_v1_1"
        with self.assertRaisesRegex(ExecutionSpecError, "v1.1"):
            validate_execution_spec(payload)

    def test_caminhos_de_saida_devem_ser_distintos(self):
        payload = valid_spec()
        payload["paths"]["reports"] = payload["paths"]["raw"]
        with self.assertRaisesRegex(ExecutionSpecError, "distintos"):
            validate_execution_spec(payload)

    def test_matriz_v2_carrega_execucao_na_identidade_e_rejeita_saida_estranha(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_execution_spec(self.write_spec(root), project_root=root)
            cell = ExperimentCell(
                "B4", "C1", "P5", "consistent", 42, 5,
                execution_id=spec.execution_id,
                execution_config_sha256=spec.sha256,
            )
            matrix = ExperimentMatrix(
                "2.0", spec.protocol_version, "a" * 64, "b" * 64, "c" * 64,
                (cell,), (), execution_id=spec.execution_id,
                execution_config_sha256=spec.sha256,
            )
            self.assertEqual(matrix.to_manifest()["execution_id"], spec.execution_id)
            self.assertEqual(cell.to_dict()["execution_config_sha256"], spec.sha256)
            with self.assertRaisesRegex(ExecutionSpecError, "saída"):
                execute_matrix(matrix, root / "results/raw/official_v1_1", lambda _: {}, execution_spec=spec)

    def test_auditoria_e_finalizacao_rejeitam_mistura_e_preservam_manifesto(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_execution_spec(self.write_spec(root), project_root=root)
            for key in ("source_catalog", "split_manifest", "catalog"):
                path = spec.resolve_path(key, root)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n", encoding="utf-8")
            cell = ExperimentCell(
                "B4", "C1", "P5", "consistent", 42, 5,
                execution_id=spec.execution_id,
                execution_config_sha256=spec.sha256,
            )
            matrix = ExperimentMatrix(
                "2.0", spec.protocol_version, "a" * 64, "b" * 64, "c" * 64,
                (cell,), (), execution_id=spec.execution_id,
                execution_config_sha256=spec.sha256,
            )
            output = spec.resolve_path("raw", root)

            def runner(current):
                return {"individual": {
                    "method": current.method, "ranking_status": "completed",
                    "evaluation_status": "evaluated", "metrics": {"ndcg_at_k": 0.5},
                    "candidate_set_id": "candidates", "relevance_id": "relevance",
                    "agent_id": "agent", "persona": current.persona, "state_version": 0,
                    "benchmark_manifest": {"git_commit": "a" * 40},
                }}

            execute_matrix(matrix, output, runner, execution_spec=spec)
            self.assertTrue(audit_execution(matrix, output, execution_spec=spec)["complete"])

            checkpoint = output / matrix.matrix_id / "cells" / f"{cell.cell_id}.json"
            mixed = json.loads(checkpoint.read_text(encoding="utf-8"))
            mixed["cell"]["execution_id"] = "execution-v1.1"
            checkpoint.write_text(json.dumps(mixed), encoding="utf-8")
            report = audit_execution(matrix, output, execution_spec=spec)
            self.assertFalse(report["complete"])
            self.assertEqual(report["failures"][0]["reason"], "cell_mismatch")

            mixed["cell"] = cell.to_dict()
            checkpoint.write_text(json.dumps(mixed), encoding="utf-8")
            manifest = finalize_versioned_execution(matrix, spec, output, project_root=root)
            self.assertEqual(manifest["execution"]["id"], spec.execution_id)
            with self.assertRaisesRegex(FileExistsError, "não será sobrescrito"):
                finalize_versioned_execution(matrix, spec, output, project_root=root)

    def test_unidades_v2_rejeitam_metodo_fora_do_contrato_pareado(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_execution_spec(self.write_spec(root), project_root=root)
            cell = ExperimentCell(
                "B0", "C1", "P5", "consistent", 42, 5,
                execution_id=spec.execution_id,
                execution_config_sha256=spec.sha256,
            )
            matrix = ExperimentMatrix(
                "2.0", spec.protocol_version, "a" * 64, "b" * 64, "c" * 64,
                (cell,), (), execution_id=spec.execution_id,
                execution_config_sha256=spec.sha256,
            )
            units = spec.resolve_path("units", root)
            with self.assertRaisesRegex(ExecutionSpecError, "B4 e B5"):
                write_units(matrix, [], "a" * 64, units, execution_spec=spec)
            self.assertFalse(units.exists())

    def test_unidades_v2_geram_trilha_fechada_por_metodo_e_auditavel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = load_execution_spec(self.write_spec(root), project_root=root)
            cell = ExperimentCell(
                "B5", "C1", "P2", "consistent", 42, 3,
                execution_id=spec.execution_id,
                execution_config_sha256=spec.sha256,
            )
            paired = ExperimentCell(
                "B4", "C1", "P2", "consistent", 42, 3,
                execution_id=spec.execution_id,
                execution_config_sha256=spec.sha256,
            )
            matrix = ExperimentMatrix(
                "2.0", spec.protocol_version, "a" * 64, "b" * 64, "c" * 64,
                (cell, paired), (), execution_id=spec.execution_id,
                execution_config_sha256=spec.sha256,
            )
            catalog = [
                {
                    "id": index,
                    "titulo": f"Filme {index}",
                    "perfil": "drama",
                    "genero": "drama",
                    "generos_secundarios": [],
                    "ano": 2000 + index,
                    "nota": 7.0,
                    "votos": index * 100,
                    "diretor": "Diretor",
                    "palavras_chave": ["tema"],
                    "sinopse": "História de teste.",
                }
                for index in range(1, 12)
            ]

            try:
                count = write_units(
                    matrix, catalog, "a" * 64, spec.resolve_path("units"),
                    execution_spec=spec,
                )
            except ExecutionSpecError as exc:
                self.fail(f"A política fechada v2 ainda foi bloqueada: {exc}")

            payload = json.loads((spec.resolve_path("units") / f"{cell.cell_id}.json").read_text())
            unit = payload["units"][0]
            paired_payload = json.loads(
                (spec.resolve_path("units") / f"{paired.cell_id}.json").read_text()
            )
            self.assertEqual(count, 2)
            self.assertEqual(payload["exposure_policy"], "closed-loop-v2")
            self.assertEqual(payload["pairing"]["pairing_id"], paired_payload["pairing"]["pairing_id"])
            self.assertEqual(len(payload["trajectory"]["steps"]), 1)
            self.assertEqual(
                payload["trajectory"]["steps"][0]["feedback_event"]["movie_id"],
                payload["trajectory"]["steps"][0]["presented_movie_ids"][0],
            )
            self.assertEqual(unit["state_version"], 1)

    def test_comando_de_validacao_nao_emite_aviso_de_modulo_precarregado(self):
        result = subprocess.run(
            [
                sys.executable, "-W", "error::RuntimeWarning", "-m",
                "cinebot_ml.experiments.execution", "validate", "--spec",
                "configs/experiments/execution_v2.yaml",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_exportacoes_publicas_dos_experimentos_continuam_compativeis(self):
        from cinebot_ml.experiments import ExperimentCell as PublicExperimentCell
        from cinebot_ml.experiments import load_experiment_matrix

        self.assertIs(PublicExperimentCell, ExperimentCell)
        self.assertTrue(callable(load_experiment_matrix))


if __name__ == "__main__":
    unittest.main()
