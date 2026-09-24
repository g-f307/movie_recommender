import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cinebot_ml.reproducibility import ArtifactVerificationError, verify_inventory


class ArtifactInventoryTests(unittest.TestCase):
    def _fixture(self, root: Path):
        target = root / "asset.bin"; target.write_bytes(b"frozen")
        pointer = root / "asset.bin.dvc"
        pointer.write_text("outs:\n- md5: abc123\n  path: asset.bin\n", encoding="utf-8")
        inventory = root / "inventory.json"
        inventory.write_text(json.dumps({"dvc_assets": [{
            "path": "asset.bin", "pointer": "asset.bin.dvc", "classification": "public",
            "bytes": 6, "sha256": hashlib.sha256(b"frozen").hexdigest(), "dvc_md5": "abc123",
            "anchor": "asset.bin", "anchor_sha256": hashlib.sha256(b"frozen").hexdigest(),
        }], "excluded_assets": []}), encoding="utf-8")
        return inventory, target

    @patch("cinebot_ml.reproducibility.load_official_records")
    def test_verifica_hash_tamanho_ponteiro_e_manifesto(self, loader):
        loader.return_value = ({"matrix_id": "matrix"}, [object(), object()])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); inventory, _ = self._fixture(root)
            report = verify_inventory(inventory, root=root)
            self.assertEqual(report["status"], "verified")
            self.assertEqual(report["official_cells"], 2)

    @patch("cinebot_ml.reproducibility.load_official_records")
    def test_rejeita_conteudo_alterado(self, loader):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); inventory, target = self._fixture(root)
            target.write_bytes(b"changed")
            with self.assertRaises(ArtifactVerificationError):
                verify_inventory(inventory, root=root)


if __name__ == "__main__":
    unittest.main()
