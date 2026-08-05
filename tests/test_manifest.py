import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ManifestTest(unittest.TestCase):
    def test_manifest_checksums_match_files(self):
        manifest = json.loads(
            (ROOT / "baseline-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["baseline"]["release_version"], "0.1.0")
        self.assertEqual(manifest["baseline"]["template_revision"], 3)
        for entry in manifest["files"]:
            content = (ROOT / entry["path"]).read_bytes()
            self.assertEqual(hashlib.sha256(content).hexdigest(), entry["sha256"])
            self.assertEqual(len(content), entry["size"])


if __name__ == "__main__":
    unittest.main()
