import contextlib
import io
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import check_training_backend as backend
from src.training_collection import read_json


class TrainingBackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_missing_backend_blocks_before_creating_an_attempt(self):
        versions = {name: "installed" for name in backend.PACKAGES}
        versions["bitsandbytes"] = None
        with patch.object(backend, "package_versions", return_value=versions), patch.object(backend, "supervise") as supervise:
            result = backend.start_probe(self.root)
        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["missing_packages"], ["bitsandbytes"])
        self.assertFalse((self.root / "outputs").exists())
        supervise.assert_not_called()

    def test_unapproved_version_blocks_without_modifying_packages(self):
        versions = {name: "installed" for name in backend.PACKAGES}
        versions["bitsandbytes"] = "0.0.0"
        with patch.object(backend, "package_versions", return_value=versions), patch.object(backend, "supervise") as supervise:
            result = backend.start_probe(self.root)
        self.assertEqual(result["status"], "blocked")
        self.assertFalse((self.root / "outputs").exists())
        supervise.assert_not_called()

    def test_existing_attempt_is_preserved(self):
        directory = self.root / "outputs/cpu_4bit_backend_v1"
        directory.mkdir(parents=True)
        marker = directory / "summary.json"
        marker.write_text('{"saved": true}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "attempt already exists"):
            backend.start_probe(self.root)
        self.assertEqual(marker.read_text(encoding="utf-8"), '{"saved": true}')

    def test_changed_script_is_rejected_before_operations(self):
        directory = self.root / "worker"
        directory.mkdir()
        from src.runtime import write_json
        write_json(directory / "run.json", {"probe_id": backend.PROBE_ID, "script_sha256": "changed"})
        with patch.object(backend, "run_operations") as operations:
            with self.assertRaisesRegex(ValueError, "implementation or dependencies changed"):
                backend.worker(directory)
        operations.assert_not_called()
        self.assertFalse((directory / "attempt.json").exists())

    @unittest.skipUnless(backend.package_versions()["bitsandbytes"] == backend.BNB_VERSION,
                         "Optional pinned bitsandbytes backend is not installed")
    def test_real_cpu_nf4_update_and_packed_base_adapter_reload(self):
        # Uses actual CPU operators in an offline subprocess, not mocked training.
        with contextlib.redirect_stdout(io.StringIO()):
            result = backend.start_probe(self.root)
        self.assertEqual(result["status"], "passed", result)
        self.assertTrue(result["quantized_cpu_base_verified"])
        self.assertTrue(result["frozen_base_unchanged"])
        self.assertTrue(result["finite_input_gradient"])
        self.assertTrue(result["adapter_output_changed"])
        self.assertTrue(result["packed_base_reload_passed"])
        self.assertTrue(result["adapter_reload_passed"])
        self.assertEqual(result["trainable_parameters"], 512)
        self.assertEqual(result["packed_weight_bytes"], 2048)
        self.assertLessEqual(result["reload_max_abs_error"], 1e-5)
        self.assertEqual(result["synthetic_optimizer_steps"], 1)
        self.assertEqual(result["medical_optimizer_steps"], 0)
        self.assertFalse(result["weights_downloaded"])
        self.assertFalse(result["full_model_checked"])
        self.assertIsNone(result["full_model_training_feasible"])
        summary = read_json(self.root / "outputs/cpu_4bit_backend_v1/summary.json")
        self.assertEqual(summary, result)


if __name__ == "__main__":
    unittest.main()
