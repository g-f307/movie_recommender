import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from cinebot_ml.experiment_config import (
    ConfigValidationError,
    FrozenConfigError,
    build_execution_manifest,
    environment_diagnostic,
    load_config,
    prepare_output_directories,
    validate_config,
    validate_paths,
    write_execution_manifest,
)


def valid_config():
    return {
        "schema_version": "1.0",
        "experiment": {"name": "test", "version": "1", "protocol_version": "protocol-v1.0", "frozen": False},
        "seeds": [42], "k_values": [5, 10],
        "paths": {"catalog": "data/catalog.json", "dataset": "datasets/data.csv", "feedback": "datasets/feedback.csv", "output_root": "results", "split_manifests": "results/manifests/splits", "manifests": "results/manifests/executions", "raw": "results/raw", "tables": "results/tables", "figures": "results/figures", "reports": "results/reports"},
        "data_versions": {"catalog": "v1", "dataset": "v1", "feedback": "v1"},
        "methods": {"enabled": ["B0", "B5"], "optional": ["B6"]},
        "conditions": ["C0", "C3"], "profiles": ["P0", "P5"],
        "metrics": {"primary": ["ndcg_at_k"], "secondary": ["precision_at_k"]},
        "splits": {"strategy": "deterministic_group_stratification_search", "temporal_strategy": "chronological_replay", "train_fraction": 0.6, "validation_fraction": 0.2, "test_fraction": 0.2, "attempts": 8},
        "execution": {"deterministic_repetitions": 1, "stochastic_repetitions": 5, "unit": "profile", "fail_on_dirty_worktree": False},
    }


class ExperimentConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for relative in ("data/catalog.json", "datasets/data.csv", "datasets/feedback.csv"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("test\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, config=None, frozen=False):
        config = config or valid_config()
        config["experiment"]["frozen"] = frozen
        path = self.root / "configs" / "experiment_v1.yaml"
        path.parent.mkdir()
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        if frozen:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            path.with_suffix(".lock.json").write_text(json.dumps({"sha256": digest}), encoding="utf-8")
        return path

    def test_configuracao_valida_e_carregada_fora_do_cwd(self):
        path = self.write_config()
        previous = Path.cwd()
        try:
            os.chdir("/")
            loaded = load_config(path)
        finally:
            os.chdir(previous)
        validate_paths(loaded, self.root)

    def test_campo_obrigatorio_ausente_tem_erro_claro(self):
        config = valid_config(); del config["seeds"]
        with self.assertRaisesRegex(ConfigValidationError, "config.seeds"):
            validate_config(config)

    def test_tipo_incorreto_e_seed_ausente_falham(self):
        config = valid_config(); config["seeds"] = "42"
        with self.assertRaisesRegex(ConfigValidationError, "seeds"):
            validate_config(config)
        config = valid_config(); config["seeds"] = []
        with self.assertRaisesRegex(ConfigValidationError, "seeds"):
            validate_config(config)

    def test_metodo_desconhecido_e_k_invalido_falham(self):
        config = valid_config(); config["methods"]["enabled"] = ["B99"]
        with self.assertRaisesRegex(ConfigValidationError, "B99"):
            validate_config(config)
        config = valid_config(); config["k_values"] = [0]
        with self.assertRaisesRegex(ConfigValidationError, "k_values"):
            validate_config(config)

    def test_campo_desconhecido_falha(self):
        config = valid_config(); config["unexpected"] = True
        with self.assertRaisesRegex(ConfigValidationError, "unexpected"):
            validate_config(config)

    def test_entrada_inexistente_e_saida_sem_escrita_falham(self):
        config = valid_config(); (self.root / "data/catalog.json").unlink()
        with self.assertRaisesRegex(ConfigValidationError, "Entrada inexistente"):
            validate_paths(config, self.root)
        (self.root / "data/catalog.json").write_text("test", encoding="utf-8")
        with patch("cinebot_ml.experiment_config.os.access", return_value=False):
            with self.assertRaisesRegex(ConfigValidationError, "Saída sem permissão"):
                validate_paths(config, self.root)

    def test_caminho_absoluto_e_segredo_sao_proibidos(self):
        config = valid_config(); config["paths"]["catalog"] = "/tmp/catalog.json"
        with self.assertRaisesRegex(ConfigValidationError, "relativo"):
            validate_config(config)
        config = valid_config(); config["api_token"] = "sensitive-value"
        stream = io.StringIO()
        with contextlib.redirect_stderr(stream), self.assertRaises(ConfigValidationError):
            validate_config(config)
        self.assertNotIn("sensitive-value", stream.getvalue())

    def test_configuracao_congelada_alterada_falha(self):
        path = self.write_config(frozen=True)
        load_config(path)
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(FrozenConfigError, "congelada"):
            load_config(path)

    def test_prepara_estrutura_padrao(self):
        config = valid_config()
        prepare_output_directories(config, self.root)
        for key in ("manifests", "raw", "tables", "figures", "reports"):
            self.assertTrue((self.root / config["paths"][key]).is_dir())

    def test_manifesto_tem_id_deterministico_hashes_e_caminhos_relativos(self):
        config = valid_config()
        first = build_execution_manifest(config, method="B5", condition="C3", profile="P5", seed=42, run=1, project_root=self.root)
        second = build_execution_manifest(config, method="B5", condition="C3", profile="P5", seed=42, run=1, project_root=self.root)
        self.assertEqual(first["execution_id"], second["execution_id"])
        self.assertEqual(first["config_snapshot"], config)
        self.assertEqual(len(first["inputs"]["dataset"]["sha256"]), 64)
        self.assertTrue(all(not Path(path).is_absolute() for path in first["artifacts"].values()))
        path = write_execution_manifest(first, config, self.root)
        self.assertEqual(path, write_execution_manifest(second, config, self.root))

    def test_diagnostico_e_sanitizado(self):
        diagnostic = environment_diagnostic(self.root)
        payload = json.dumps(diagnostic).lower()
        self.assertIn("python", diagnostic)
        self.assertNotIn("token", payload)
        self.assertNotIn("password", payload)


if __name__ == "__main__":
    unittest.main()
