"""Exercise CUTLASS staging safeguards without reaching the network."""

import importlib.util
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[2] / ".github/scripts/install-cutlass.py"
SPEC = importlib.util.spec_from_file_location("install_cutlass", SCRIPT)
install_cutlass = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(install_cutlass)


class InstallCutlassTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.destination = Path(temporary.name) / "headers"
        self.destination.mkdir()

    def test_nonempty_destination_preserves_existing_data(self):
        existing = self.destination / "existing.txt"
        existing.write_bytes(b"user data")
        with patch.object(install_cutlass.urllib.request, "urlopen") as download:
            with self.assertRaisesRegex(ValueError, "empty destination"):
                install_cutlass.stage(self.destination)
        download.assert_not_called()
        self.assertEqual(existing.read_bytes(), b"user data")

    def test_reuse_pinned_headers_without_downloading(self):
        headers = self.destination / "include/cutlass"
        gemm = headers / "gemm/device/gemm.h"
        gemm.parent.mkdir(parents=True)
        gemm.write_text("// Cached header\n", encoding="utf-8")
        (headers / "version.h").write_text(
            "#define CUTLASS_MAJOR 4\n#define CUTLASS_MINOR 4\n#define CUTLASS_PATCH 2\n",
            encoding="utf-8",
        )
        with patch.object(install_cutlass.urllib.request, "urlopen") as download:
            self.assertEqual(install_cutlass.stage(self.destination), self.destination.resolve())
        download.assert_not_called()

    def test_checksum_failure_leaves_destination_empty(self):
        with patch.object(install_cutlass.urllib.request, "urlopen", return_value=BytesIO(b"wrong")):
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                install_cutlass.stage(self.destination)
        self.assertEqual(list(self.destination.iterdir()), [])
        self.assertEqual(list(self.destination.parent.iterdir()), [self.destination])


if __name__ == "__main__":
    unittest.main()
