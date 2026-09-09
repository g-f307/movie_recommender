import csv
import json
import tempfile
import unittest
from pathlib import Path

from cinebot_ml.experimental_splits import (
    GroupStats,
    ManifestConflictError,
    SplitConfig,
    SplitValidationError,
    assert_partition_allowed,
    build_group_manifest,
    build_temporal_manifest,
    collect_group_stats,
    create_group_manifest_from_csv,
    history_before,
    load_manifest,
    reconstruct_assignments,
    reconstruct_temporal_partitions,
    sha256_file,
    split_groups,
    temporal_split_events,
    validate_disjoint_partitions,
    validate_manifest_source,
    validate_temporal_partitions,
    write_manifest,
)


def group_rows(group_count=12):
    rows = []
    for group in range(group_count):
        rows.append({"movie_id": f"m{group:02d}", "relevante": "0"})
        rows.extend(
            {"movie_id": f"m{group:02d}", "relevante": "1"}
            for _ in range(1 + group % 3)
        )
    return rows


def events(users=("u1", "u2"), count=5):
    return [
        {
            "user_id": user,
            "timestamp": f"2026-09-{day:02d}T00:00:00Z",
            "movie_id": f"{user}-m{day}",
        }
        for user in users
        for day in range(1, count + 1)
    ]


class ExperimentalSplitsTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def write_dataset(self, rows):
        path = self.root / "dataset.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["movie_id", "relevante"])
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_mesma_seed_reproduz_split_e_manifesto(self):
        path = self.write_dataset(group_rows())
        config = SplitConfig(seed=42, attempts=16)
        first = create_group_manifest_from_csv(path, config)
        second = create_group_manifest_from_csv(path, config)
        self.assertEqual(first, second)

    def test_seeds_diferentes_podem_gerar_splits_diferentes(self):
        groups = collect_group_stats(group_rows(), "movie_id", "relevante")
        first = split_groups(groups, SplitConfig(seed=42, attempts=1))
        second = split_groups(groups, SplitConfig(seed=137, attempts=1))
        self.assertNotEqual(first, second)

    def test_intersecao_de_movie_id_e_bloqueada(self):
        partitions = {"train": ["m1", "m2"], "validation": ["m3"], "test": ["m2"]}
        with self.assertRaisesRegex(SplitValidationError, "Leakage"):
            validate_disjoint_partitions(partitions)

    def test_intersecao_de_usuario_e_bloqueada(self):
        partitions = {"train": ["u1"], "validation": ["u2"], "test": ["u1"]}
        with self.assertRaisesRegex(SplitValidationError, "Leakage"):
            validate_disjoint_partitions(partitions)

    def test_split_agrupado_por_usuario_pseudonimizado(self):
        rows = [
            {"user_id": f"anon-{user}", "relevante": str(label)}
            for user in range(9)
            for label in (0, 1)
        ]
        groups = collect_group_stats(rows, "user_id", "relevante")
        partitions = split_groups(groups, SplitConfig(seed=42, attempts=8))
        assigned = [group_id for values in partitions.values() for group_id in values]
        self.assertEqual(len(assigned), 9)
        self.assertEqual(len(set(assigned)), 9)

    def test_identificador_repetido_na_mesma_particao_e_bloqueado(self):
        partitions = {"train": ["m1", "m1"], "validation": ["m2"], "test": ["m3"]}
        with self.assertRaisesRegex(SplitValidationError, "identificadores repetidos"):
            validate_disjoint_partitions(partitions)

    def test_feedback_futuro_nao_entra_no_historico(self):
        source = events(users=("u1",), count=4)
        history = history_before(source, "2026-09-03T00:00:00Z")
        self.assertEqual([item["movie_id"] for item in history], ["u1-m1", "u1-m2"])

    def test_timestamp_ausente_ou_invalido_e_bloqueado(self):
        missing = [{"user_id": "u1", "timestamp": ""}]
        invalid = [{"user_id": "u1", "timestamp": "ontem"}]
        with self.assertRaisesRegex(SplitValidationError, "timestamp ausente"):
            temporal_split_events(missing, SplitConfig())
        with self.assertRaisesRegex(SplitValidationError, "Timestamp inválido"):
            temporal_split_events(invalid, SplitConfig())

    def test_timestamp_sem_fuso_e_bloqueado(self):
        source = [{"user_id": "u1", "timestamp": "2026-09-01T00:00:00"}]
        with self.assertRaisesRegex(SplitValidationError, "sem fuso"):
            temporal_split_events(source, SplitConfig())

    def test_dataset_com_uma_classe_e_bloqueado(self):
        rows = [{"movie_id": f"m{i}", "relevante": "1"} for i in range(5)]
        with self.assertRaisesRegex(SplitValidationError, "duas classes"):
            collect_group_stats(rows, "movie_id", "relevante")

    def test_fonte_alterada_invalida_manifesto(self):
        path = self.write_dataset(group_rows())
        manifest = create_group_manifest_from_csv(path, SplitConfig(attempts=4))
        validate_manifest_source(manifest, path)
        with path.open("a", encoding="utf-8") as handle:
            handle.write("m99,1\n")
        with self.assertRaisesRegex(SplitValidationError, "fonte mudou"):
            validate_manifest_source(manifest, path)

    def test_holdout_nao_pode_selecionar_modelo_threshold_ou_calibrar(self):
        for purpose in ("select_model", "select_threshold", "calibrate", "hyperparameter_tuning"):
            with self.subTest(purpose=purpose):
                with self.assertRaisesRegex(SplitValidationError, "holdout final"):
                    assert_partition_allowed("test", purpose)
        assert_partition_allowed("validation", "select_threshold")
        assert_partition_allowed("test", "final_evaluation")

    def test_reconstrucao_do_split_a_partir_do_manifesto(self):
        path = self.write_dataset(group_rows())
        manifest = create_group_manifest_from_csv(path, SplitConfig(attempts=8))
        manifest_path = self.root / "manifest.json"
        write_manifest(manifest, manifest_path)
        loaded = load_manifest(manifest_path)
        assignments = reconstruct_assignments(loaded)
        self.assertEqual(len(assignments), manifest["total_groups"])
        self.assertEqual(set(assignments.values()), {"train", "validation", "test"})

    def test_manifesto_nao_e_sobrescrito_silenciosamente(self):
        path = self.root / "manifest.json"
        manifest = {"example": True}
        write_manifest(manifest, path)
        with self.assertRaises(ManifestConflictError):
            write_manifest(manifest, path)
        write_manifest({"example": False}, path, overwrite=True)
        self.assertFalse(json.loads(path.read_text(encoding="utf-8"))["example"])

    def test_split_temporal_respeita_ordem_por_usuario(self):
        source = events(count=5)
        partitions = temporal_split_events(source, SplitConfig())
        validate_temporal_partitions(source, partitions)
        for user in ("u1", "u2"):
            times = {
                name: [source[index]["timestamp"] for index in partitions[name] if source[index]["user_id"] == user]
                for name in ("train", "validation", "test")
            }
            self.assertLess(max(times["train"]), min(times["validation"]))
            self.assertLess(max(times["validation"]), min(times["test"]))

    def test_timestamp_igual_permanece_na_mesma_particao(self):
        source = events(users=("u1",), count=3)
        source.append(dict(source[0], movie_id="duplicated-time"))
        partitions = temporal_split_events(source, SplitConfig())
        locations = {
            name for name, indexes in partitions.items() if 0 in indexes or 3 in indexes
        }
        self.assertEqual(len(locations), 1)

    def test_manifesto_temporal_e_reconstruivel(self):
        source = events(count=5)
        partitions = temporal_split_events(source, SplitConfig())
        manifest = build_temporal_manifest(
            source,
            partitions,
            SplitConfig(),
            source_name="feedback.csv",
            source_sha256="abc",
        )
        self.assertEqual(reconstruct_temporal_partitions(manifest), partitions)

    def test_configuracao_invalida_produz_erro_explicito(self):
        config = SplitConfig(train_fraction=0.7, validation_fraction=0.2, test_fraction=0.2)
        with self.assertRaisesRegex(SplitValidationError, "somar 1"):
            config.validate()

    def test_manifesto_registra_seed_estrategia_tamanho_e_versao(self):
        groups = collect_group_stats(group_rows(), "movie_id", "relevante")
        config = SplitConfig(seed=2027, attempts=4)
        partitions = split_groups(groups, config)
        manifest = build_group_manifest(
            groups,
            partitions,
            config,
            source_name="dataset.csv",
            source_sha256="abc",
            group_column="movie_id",
            label_column="relevante",
        )
        self.assertEqual(manifest["config"]["seed"], 2027)
        self.assertEqual(manifest["split_version"], "1.0")
        self.assertEqual(manifest["strategy"], "deterministic_group_stratification_search")
        self.assertEqual(sum(item["group_count"] for item in manifest["partitions"].values()), 12)


if __name__ == "__main__":
    unittest.main()
