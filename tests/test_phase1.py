import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

from src.auditor import auditor_probe, endpoint_url, validate_audit
from src.model import supervised_batch, training_step
from src.runtime import read_settings, safe_error, supervise, write_json


ROOT = Path(__file__).resolve().parents[1]


class Phase1Tests(unittest.TestCase):
    def setUp(self):
        self.settings = read_settings(ROOT / "settings.json")

    def test_invalid_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            for key, value in [("revision", "main"), ("cpu_threads", True),
                               ("max_new_tokens", 0), ("learning_rate", float("nan"))]:
                settings = dict(self.settings, **{key: value})
                path.write_text(json.dumps(settings))
                with self.assertRaises(ValueError):
                    read_settings(path)

    def test_report_write_preserves_previous_file_on_invalid_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            write_json(path, {"status": "passed"})
            with self.assertRaises(ValueError):
                write_json(path, {"value": float("nan")})
            self.assertEqual(json.loads(path.read_text()), {"status": "passed"})

    def test_secrets_are_redacted_from_errors(self):
        with patch.dict(os.environ, {"AUDITOR_API_KEY": "fake-secret"}):
            self.assertNotIn("fake-secret", safe_error(ValueError("bad fake-secret")))

    def test_auditor_endpoint_trust_boundary(self):
        self.assertEqual(endpoint_url("http://localhost:8080/v1"),
                         "http://localhost:8080/v1/chat/completions")
        for endpoint in ["http://example.com/v1", "https://user:pass@example.com/v1",
                         "https://example.com/v1?key=secret", "file:///tmp/test"]:
            with self.assertRaises(ValueError):
                endpoint_url(endpoint)

    def test_auditor_configuration_is_not_a_false_pass(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(auditor_probe()["status"], "blocked")

    def test_auditor_schema_and_answer_free_practice_hint(self):
        valid = {"action": "repair", "prefix_end": 0,
                 "hint": "Reinspect the required color and outline."}
        self.assertEqual(validate_audit(valid), valid)
        for replacement in [{"prefix_end": True}, {"prefix_end": -1},
                            {"hint": "It is a red circle."}, {"hint": ""},
                            {"action": "unknown"}, {"extra": "unexpected"}]:
            with self.assertRaises(ValueError):
                validate_audit(dict(valid, **replacement))

    def test_auditor_rejects_reported_string_boundary(self):
        with self.assertRaisesRegex(ValueError, "character boundary"):
            validate_audit({"action": "repair", "prefix_end": "color",
                            "hint": "The central shape is red. Its outline is square."})

    def test_auditor_rejects_wrong_shape_in_hint(self):
        with self.assertRaisesRegex(ValueError, "states a color or shape"):
            validate_audit({"action": "repair", "prefix_end": 0,
                            "hint": "Its outline is square."})

    def test_auditor_rejects_preserved_wrong_practice_finding(self):
        with self.assertRaisesRegex(ValueError, "unsupported finding"):
            validate_audit({"action": "repair", "prefix_end": 49,
                            "hint": "Reinspect the color and outline."})

    def test_auditor_http_contract(self):
        received = []

        class Handler(BaseHTTPRequestHandler):
            audit = {"action": "repair", "prefix_end": 0,
                     "hint": "Reinspect the color and outline."}

            def log_message(self, *args):
                pass

            def do_POST(self):
                received.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"choices": [{"message": {
                    "content": json.dumps(self.audit)
                }}]}).encode()
                self.send_response(200)
                self.end_headers()
                self.wfile.write(body)

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                    "AUDITOR_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                    "AUDITOR_MODEL": "local-test-auditor", "AUDITOR_API_KEY": "fake-secret"
                }):
                    result = auditor_probe(directory)
                    self.assertTrue((Path(directory) / "auditor_request.json").is_file())
                    self.assertTrue((Path(directory) / "auditor_response.json").is_file())
                    Handler.audit = {"action": "repair", "prefix_end": "color",
                                     "hint": "The central shape is red. Its outline is square."}
                    with self.assertRaisesRegex(ValueError, "character boundary"):
                        auditor_probe(directory)
                    saved = json.loads((Path(directory) / "auditor_response.json").read_text())
                    envelope = json.loads(saved["response_text"])
                    invalid = json.loads(envelope["choices"][0]["message"]["content"])
                    self.assertEqual(invalid["prefix_end"], "color")
                self.assertEqual(result["status"], "passed")
                request = received[0]
                self.assertEqual(request["response_format"]["type"], "json_schema")
                contract = request["response_format"]["json_schema"]["schema"]
                self.assertEqual(contract["properties"]["prefix_end"], {
                    "type": "integer", "minimum": 0, "maximum": 49})
                self.assertFalse(contract["additionalProperties"])
                self.assertEqual(request["temperature"], 0)
                self.assertNotIn("fake-secret", json.dumps(request))
                self.assertNotIn("reference_answer", json.dumps(request))
                text = request["messages"][1]["content"][1]["text"]
                self.assertNotIn("red circle", text)
                self.assertTrue(request["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"))
            finally:
                server.shutdown()
                thread.join()

    def test_auditor_does_not_follow_redirects(self):
        class Redirect(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.send_response(302)
                self.send_header("Location", "https://example.com/should-not-be-contacted")
                self.end_headers()

        with ThreadingHTTPServer(("127.0.0.1", 0), Redirect) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with patch.dict(os.environ, {
                    "AUDITOR_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
                    "AUDITOR_MODEL": "local-test-auditor"
                }):
                    with self.assertRaisesRegex(RuntimeError, "HTTP 302"):
                        auditor_probe()
            finally:
                server.shutdown()
                thread.join()

    def test_supervisor_stops_a_stalled_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            settings = dict(self.settings, timeout_seconds=0.3)
            result = supervise([sys.executable, "-c", "import time; time.sleep(10)"],
                               settings, Path(directory) / "console.log")
            self.assertEqual(result["limit_reached"], "time_limit")
            self.assertLess(result["elapsed_seconds"], 5)

    def test_supervisor_stops_when_sample_exceeds_limit(self):
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as directory:
            settings = dict(self.settings, memory_limit_gib=0.03)
            command = [sys.executable, "-c", "import time; time.sleep(10)"]
            with patch("psutil.Process") as process:
                process.return_value.children.return_value = []
                process.return_value.memory_info.return_value = SimpleNamespace(rss=80 * 1024**2)
                result = supervise(command, settings, Path(directory) / "console.log")
            self.assertEqual(result["limit_reached"], "memory_limit")

    def test_supervised_mask_and_boundary(self):
        import torch

        class Processor:
            mismatch = False

            def apply_chat_template(self, messages, **kwargs):
                ids = [1, 2, 3] if len(messages) == 1 else [1, 2, 3, 4, 5]
                if self.mismatch and len(messages) > 1:
                    ids[0] = 7
                return {"input_ids": torch.tensor([ids]),
                        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
                        "pixel_values": torch.ones(1, 1, 3, 2, 2)}

        processor = Processor()
        batch = supervised_batch(processor, None)
        self.assertEqual(batch["labels"].tolist(), [[-100, -100, -100, 4, 5]])
        self.assertIn("pixel_values", batch)
        processor.mismatch = True
        with self.assertRaisesRegex(ValueError, "boundary"):
            supervised_batch(processor, None)

    def test_training_update_and_frozen_parameter(self):
        import torch

        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
                self.lora_B = torch.nn.Parameter(torch.tensor(0.0))

            def forward(self, target, **kwargs):
                from types import SimpleNamespace
                return SimpleNamespace(loss=(self.base + self.lora_B - target)**2)

        model = SmallModel()
        result = training_step(model, {"target": torch.tensor(2.0)}, 0.001)
        self.assertEqual(result["changed_parameter_tensors"], 1)
        self.assertTrue(result["frozen_parameter_sample_unchanged"])
        self.assertNotEqual(model.lora_B.item(), 0)
        self.assertEqual(model.base.item(), 1)

    def test_default_run_is_environment_only(self):
        process = subprocess.run([sys.executable, str(ROOT / "run_pilot.py")],
                                 capture_output=True, text=True, timeout=30,
                                 cwd=tempfile.gettempdir())
        self.assertIn('"scope": "Package presence', process.stdout)
        self.assertNotIn("Loading weights", process.stdout)
        self.assertIn(process.returncode, [0, 2])


if __name__ == "__main__":
    unittest.main()
