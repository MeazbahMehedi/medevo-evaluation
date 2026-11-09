import contextlib
import io
import json
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import Mock, patch

import test_audit_experiment as audit_fixtures
from check_auditor import check_one_case, grounded_request
from src.runtime import write_json
from src.training_collection import read_json, tree_hashes


class SingleAuditorCheckTests(unittest.TestCase):
    def setUp(self):
        fixture = audit_fixtures.AuditExperimentTests(methodName="test_only_the_system_instruction_changes")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.reply = {"action": "unresolved", "first_error_step": 0, "error_family": "unresolved",
                      "prefix_supported": False, "hint": "", "evidence": "Controlled transport fixture."}
        self.request = Mock(side_effect=self.transport)

    def transport(self, payload, config, folder, name):
        write_json(Path(folder) / "audit_request.json", payload)
        self.fixture.save_response(folder, self.reply)
        return self.reply

    def run_check(self, **options):
        f = self.fixture
        with patch("check_auditor.request_json", self.request), patch("src.model.load_primary") as load, \
                patch("src.repair_model.continue_reasoning") as continuation, \
                contextlib.redirect_stdout(io.StringIO()):
            result = check_one_case(f.project, f.data, f.collection, f.settings, f.config, **options)
        load.assert_not_called(); continuation.assert_not_called()
        return result

    def test_request_appends_exact_schema_and_preserves_every_other_field(self):
        original = read_json(self.fixture.collection / "cases/01/audit_request.json")
        original_bytes = json.dumps(original)
        trial = grounded_request(original)
        original_instruction = original["messages"][0]["content"]
        self.assertTrue(trial["messages"][0]["content"].startswith(original_instruction + "\n\n"))
        schema_text = trial["messages"][0]["content"].split("Output JSON schema (identical to response_format):\n", 1)[1]
        self.assertEqual(json.loads(schema_text), original["response_format"]["json_schema"]["schema"])
        trial["messages"][0]["content"] = original_instruction
        self.assertEqual(trial, original)
        self.assertEqual(json.dumps(original), original_bytes)

    def test_only_first_case_is_called_without_references_or_original_file_changes(self):
        before = tree_hashes(self.fixture.collection)
        result = self.run_check()
        selected = read_json(self.fixture.collection / "run.json")["selected_ids"]
        self.request.assert_called_once()
        self.assertEqual(result["selected_id"], selected[0])
        self.assertEqual(result["trial"]["outcome"], "auditor_unresolved")
        self.assertEqual(result["new_audit_calls_started"], 1)
        self.assertEqual(tree_hashes(self.fixture.collection), before)
        self.assertNotIn("MRI", json.dumps(self.request.call_args.args[0]))
        self.assertEqual(result["training_examples_exported"], 0)
        self.assertEqual(result["new_continuations"], 0)

    def test_invalid_repair_is_retained_without_repairing_fields_or_retrying(self):
        self.reply.update(action="repair", first_error_step=0, error_family="R", prefix_supported=True, hint="Recheck the image.")
        result = self.run_check()
        self.assertEqual(result["trial"]["outcome"], "invalid_audit")
        self.assertEqual(result["trial"]["raw_audit"]["first_error_step"], 0)
        self.request.assert_called_once()

    def test_hint_valid_repair_still_requires_review_and_no_continuation(self):
        self.reply.update(action="repair", first_error_step=1, error_family="R", prefix_supported=True, hint="Recheck the visible features.")
        result = self.run_check()
        self.assertEqual(result["trial"]["outcome"], "repair_requires_clinical_review")
        self.assertFalse(result["automatic_candidate_admission"])
        self.assertEqual(result["optimizer_steps"], 0)

    def test_transport_failure_stops_after_one_call_with_saved_failure_record(self):
        self.request.side_effect = RuntimeError("Controlled service failure")
        result = self.run_check()
        self.assertEqual(result["status"], "incomplete")
        self.request.assert_called_once()
        self.assertIn("Controlled service failure", result["request_error"])
        self.assertTrue((Path(result["output_folder"]) / "trial/request_error.json").exists())

    def test_validation_failure_prevents_any_network_call(self):
        path = self.fixture.collection / "cases/01/audit_request.json"
        value = read_json(path); value["temperature"] = .7; write_json(path, value)
        with self.assertRaises(ValueError):
            self.run_check()
        self.request.assert_not_called()

    def test_real_http_transport_sends_exactly_one_scripted_request(self):
        captured = []
        reply = self.reply
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        f = self.fixture
        config = {**f.config, "auditor_base_url": f"http://127.0.0.1:{server.server_port}/v1"}
        path = f.collection / "run.json"
        value = read_json(path); value["config"] = config; write_json(path, value)
        with contextlib.redirect_stdout(io.StringIO()):
            result = check_one_case(f.project, f.data, f.collection, f.settings, config)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(captured), 1)
        self.assertIn("Output JSON schema", captured[0]["messages"][0]["content"])

    def test_model_comparison_changes_only_model_and_preserves_both_baselines(self):
        baseline = Path(self.run_check()["output_folder"])
        baseline_hashes = tree_hashes(baseline)
        collection_hashes = tree_hashes(self.fixture.collection)
        original_payload = read_json(baseline / "trial/audit_request.json")
        self.request.reset_mock()
        result = self.run_check(replacement_model="qwen2.5vl:7b", baseline_check_dir=baseline)
        self.request.assert_called_once()
        trial_payload, effective_config = self.request.call_args.args[:2]
        self.assertEqual(trial_payload["model"], "qwen2.5vl:7b")
        self.assertEqual(effective_config, {**self.fixture.config, "auditor_model": "qwen2.5vl:7b"})
        self.assertEqual({**trial_payload, "model": original_payload["model"]}, original_payload)
        self.assertEqual(tree_hashes(baseline), baseline_hashes)
        self.assertEqual(tree_hashes(self.fixture.collection), collection_hashes)
        self.assertEqual(result["original_condition"], "schema-grounded 3B")
        self.assertEqual(result["trial_model"], "qwen2.5vl:7b")
        self.assertEqual(result["new_audit_calls_started"], 1)
        self.assertEqual(result["training_examples_exported"], 0)
        copied = Path(result["output_folder"]) / "baseline_3b/trial/audit_request.json"
        self.assertEqual(read_json(copied), original_payload)

    def test_replacement_requires_both_explicit_flags_and_approved_model(self):
        for options in ({"replacement_model": "qwen2.5vl:7b"}, {"baseline_check_dir": self.fixture.collection},
                        {"replacement_model": "unapproved:model", "baseline_check_dir": self.fixture.collection}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.run_check(**options)
        self.request.assert_not_called()

    def test_mismatched_grounded_baseline_prevents_any_new_call(self):
        baseline = Path(self.run_check()["output_folder"])
        self.request.reset_mock()
        path = baseline / "trial/audit_request.json"
        original = read_json(path)
        for field, value in (("temperature", .7), ("max_tokens", 128), ("model", "different:model")):
            write_json(path, {**original, field: value})
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "same-request"):
                self.run_check(replacement_model="qwen2.5vl:7b", baseline_check_dir=baseline)
        self.request.assert_not_called()

    def test_replacement_transport_failure_does_not_fall_back_to_3b(self):
        baseline = Path(self.run_check()["output_folder"])
        self.request.reset_mock()
        self.request.side_effect = RuntimeError("Controlled replacement model failure")
        result = self.run_check(replacement_model="qwen2.5vl:7b", baseline_check_dir=baseline)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["trial_model"], "qwen2.5vl:7b")
        self.request.assert_called_once()

    def test_real_http_model_comparison_sends_one_7b_request(self):
        captured = []
        reply = self.reply
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        f = self.fixture
        f.config = {**f.config, "auditor_base_url": f"http://127.0.0.1:{server.server_port}/v1"}
        path = f.collection / "run.json"
        value = read_json(path); value["config"] = f.config; write_json(path, value)
        baseline = Path(self.run_check()["output_folder"])
        with contextlib.redirect_stdout(io.StringIO()):
            result = check_one_case(f.project, f.data, f.collection, f.settings, f.config,
                                    replacement_model="qwen2.5vl:7b", baseline_check_dir=baseline)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0]["model"], "qwen2.5vl:7b")


if __name__ == "__main__":
    unittest.main()
