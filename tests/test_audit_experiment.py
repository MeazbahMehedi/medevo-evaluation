import contextlib
import io
import json
import os
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import unittest
from unittest.mock import Mock, patch

from PIL import Image

import test_training_collection as collection_fixtures
from src.audit_experiment import (SHORT_INSTRUCTION, assess_response, audit_worker, shorter_request,
                                  start_audit_experiment, validate_audit_inputs)
from src.repair_audit import audit_payload
from src.repair_model import reasoning_steps
from src.runtime import supervise, write_json
from src.training_collection import read_json, tree_hashes


class AuditExperimentTests(unittest.TestCase):
    def setUp(self):
        fixture = collection_fixtures.TrainingCollectionTests(
            methodName="test_selection_is_balanced_fixed_training_only_and_nine_new")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.collection = Path(fixture.start()[0]["output_folder"])
        self.project, self.data = fixture.project, fixture.data
        self.settings, self.config = fixture.settings, fixture.config
        self.audit = {"action": "skip", "first_error_step": 0, "error_family": "R", "prefix_supported": False,
                      "hint": "Unused fixture text.", "evidence": "Controlled transport fixture; no medical judgement."}
        rows = {row["id"]: row for row in fixture.records}
        for index, identity in enumerate(read_json(self.collection / "run.json")["selected_ids"], 1):
            folder = self.collection / "cases" / f"{index:02d}"
            if (folder / "audit_request.json").exists():
                continue
            row = rows[identity]
            initial = read_json(folder / "initial_reasoning.json")
            steps = read_json(folder / "reasoning_steps.json")
            with Image.open(self.data / row["image_path"]) as image:
                payload = audit_payload(image, row["question"], initial["raw_response"], steps,
                                        initial.get("entropy", {"status": "unavailable"}), self.config)
            write_json(folder / "audit_request.json", payload)
            self.save_response(folder, self.audit)
        self.request = Mock(side_effect=self.reply)

    def save_response(self, folder, value):
        write_json(Path(folder) / "audit_response.json", {"response_text": json.dumps({"choices": [
            {"message": {"content": json.dumps(value)}, "finish_reason": "stop"}]})})

    def reply(self, payload, config, folder, name):
        write_json(Path(folder) / "audit_request.json", payload)
        self.save_response(folder, self.audit)
        return self.audit

    def start(self, supervisor=None):
        def execute(command, settings, log):
            code = audit_worker(self.project, Path(command[-1]))
            return {"worker_exit_code": code, "limit_reached": None, "peak_process_tree_rss_bytes": 100}
        with patch("src.audit_experiment.request_json", self.request), \
                patch("src.audit_experiment.supervise", side_effect=supervisor or execute), \
                patch("src.model.load_primary") as model, patch("src.repair_model.generate_reasoning") as reasoning, \
                patch("src.repair_model.answer_readout") as readout, patch("src.repair_model.continue_reasoning") as continuation, \
                patch("src.model.training_probe") as train, contextlib.redirect_stdout(io.StringIO()):
            result = start_audit_experiment(self.project, self.data, self.collection, self.settings, self.config)
        model.assert_not_called(); reasoning.assert_not_called(); readout.assert_not_called()
        continuation.assert_not_called(); train.assert_not_called()
        return result

    def test_only_the_system_instruction_changes(self):
        payload = read_json(self.collection / "cases/01/audit_request.json")
        original = json.loads(json.dumps(payload))
        trial = shorter_request(payload)
        self.assertEqual(payload, original)
        self.assertLess(len(SHORT_INSTRUCTION), len(payload["messages"][0]["content"]))
        self.assertEqual(trial["messages"][0]["content"], SHORT_INSTRUCTION)
        trial["messages"][0]["content"] = payload["messages"][0]["content"]
        self.assertEqual(trial, payload)

    def test_full_comparison_makes_only_twelve_audit_calls_and_preserves_originals(self):
        before = tree_hashes(self.collection)
        result = self.start()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["new_audit_calls_started"], 12)
        self.assertEqual(self.request.call_count, 12)
        self.assertEqual(result["trial"]["saved_response_count"], 12)
        self.assertEqual(result["trial"]["schema_valid_count"], 12)
        self.assertEqual(result["trial"]["outcomes"], {"auditor_skip": 12})
        self.assertEqual(tree_hashes(self.collection), before)
        self.assertFalse(result["automatic_candidate_admission"])
        self.assertEqual(result["new_continuations"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["training_examples_exported"], 0)
        self.assertFalse(result["worker"]["ollama_peak_memory_measured"])
        for call in self.request.call_args_list:
            payload = call.args[0]
            self.assertNotIn("MRI", json.dumps(payload))
            self.assertEqual(payload["max_tokens"], self.config["auditor_max_tokens"])
            self.assertEqual(payload["temperature"], 0)
            self.assertEqual(payload["seed"], 7)
            visible = json.loads(payload["messages"][1]["content"][1]["text"])
            self.assertEqual(set(visible), {"question", "response", "steps", "entropy_proposal"})

    def test_invalid_repair_step_is_not_corrected_or_used(self):
        self.audit.update(action="repair", first_error_step=0, prefix_supported=True,
                          hint="Recheck the visible features.")
        result = self.start()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["trial"]["schema_valid_count"], 0)
        self.assertEqual(result["trial"]["invalid_audit_count"], 12)
        self.assertEqual(result["trial"]["repair_hint_pass_count"], 0)
        report = read_json(Path(result["output_folder"]) / "comparison.json")
        self.assertTrue(all(row["trial"]["raw_audit"]["first_error_step"] == 0 for row in report["cases"]))

    def test_reference_term_hints_remain_rejected(self):
        self.audit.update(action="repair", first_error_step=1, prefix_supported=True, hint="Recheck the MRI features.")
        result = self.start()
        self.assertEqual(result["trial"]["schema_valid_repair_count"], 12)
        self.assertEqual(result["trial"]["repair_hint_pass_count"], 0)
        self.assertEqual(result["trial"]["outcomes"], {"rejected_hint": 12})

    def test_structure_and_hint_passes_require_clinical_review_and_never_continue(self):
        self.audit.update(action="repair", first_error_step=1, prefix_supported=True, hint="Recheck the visible features.")
        result = self.start()
        self.assertEqual(result["trial"]["repair_hint_pass_count"], 12)
        self.assertEqual(result["trial"]["outcomes"], {"repair_requires_clinical_review": 12})
        self.assertEqual(result["trial"]["clinical_coherence"], "requires_review_of_all_raw_responses")
        self.assertEqual(result["new_continuations"], 0)

    def test_nonrepair_garbage_is_preserved_without_becoming_a_quality_pass(self):
        self.audit.update(hint="user user user repeated instructions", evidence="[")
        result = self.start()
        self.assertEqual(result["trial"]["schema_valid_count"], 12)
        self.assertEqual(result["trial"]["clinical_coherence"], "requires_review_of_all_raw_responses")
        report = read_json(Path(result["output_folder"]) / "comparison.json")
        self.assertTrue(all(not row["trial"]["hint_used_for_generation"] for row in report["cases"]))
        self.assertTrue(all(row["trial"]["hint"] == self.audit["hint"] for row in report["cases"]))

    def test_malformed_saved_json_is_invalid_without_retries(self):
        def invalid(payload, config, folder, name):
            write_json(Path(folder) / "audit_request.json", payload)
            write_json(Path(folder) / "audit_response.json", {"response_text": "not JSON"})
            raise ValueError("Controlled malformed reply")
        self.request.side_effect = invalid
        result = self.start()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["trial"]["invalid_audit_count"], 12)
        self.assertEqual(self.request.call_count, 12)

    def test_transport_failure_stops_with_partial_results_and_unset_full_rate(self):
        def flaky(*args):
            if self.request.call_count == 3:
                raise RuntimeError("Controlled service failure")
            return self.reply(*args)
        self.request.side_effect = flaky
        result = self.start()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["new_audit_calls_started"], 3)
        self.assertEqual(result["trial"]["saved_response_count"], 2)
        self.assertIsNone(result["trial"]["schema_valid_pct"])
        self.assertEqual(self.request.call_count, 3)

    def test_changed_auditor_config_or_saved_image_payload_fails_before_calls(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_audit_inputs(self.project, self.data, self.collection, self.settings,
                                  {**self.config, "auditor_max_tokens": 256})
        path = self.collection / "cases/01/audit_request.json"
        value = read_json(path)
        value["messages"][1]["content"][0]["image_url"]["url"] = "data:image/png;base64,changed"
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "frozen inputs"):
            validate_audit_inputs(self.project, self.data, self.collection, self.settings, self.config)
        self.request.assert_not_called()

    def test_changed_auditor_implementation_is_rejected(self):
        path = self.collection / "run.json"
        value = read_json(path); value["source_sha256"]["src/repair_audit.py"] = "0" * 64
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "implementation changed"):
            validate_audit_inputs(self.project, self.data, self.collection, self.settings, self.config)

    def test_worker_lock_request_integrity_and_repeat_guards_prevent_extra_calls(self):
        result = self.start()
        directory = Path(result["output_folder"])
        lock = directory / "run.lock"; lock.write_text("existing")
        with self.assertRaises(FileExistsError):
            audit_worker(self.project, directory)
        self.assertEqual(lock.read_text(), "existing"); lock.unlink()
        count = self.request.call_count
        with patch("src.audit_experiment.request_json", self.request):
            self.assertEqual(audit_worker(self.project, directory), 2)
        self.assertEqual(self.request.call_count, count)
        path = directory / "inputs.json"
        inputs = read_json(path); inputs[0]["request"]["seed"] = 8
        write_json(path, inputs)
        with patch("src.audit_experiment.request_json", self.request):
            self.assertEqual(audit_worker(self.project, directory), 2)
        self.assertEqual(self.request.call_count, count)

    def test_original_invalid_audit_is_assessed_without_rewriting_it(self):
        folder = self.fixture.root / "invalid_fixture"; folder.mkdir()
        value = {**self.audit, "action": "repair", "first_error_step": 0, "prefix_supported": True}
        self.save_response(folder, value)
        before = tree_hashes(folder)
        result = assess_response(folder / "audit_response.json", reasoning_steps("Controlled reasoning."), "MRI")
        self.assertEqual(result["outcome"], "invalid_audit")
        self.assertEqual(tree_hashes(folder), before)

    def test_cli_routes_only_to_the_auditor_experiment(self):
        import evaluate
        with patch("sys.argv", ["evaluate.py", "compare-audits", "--collection-dir", str(self.collection)]), \
                patch("src.audit_experiment.start_audit_experiment", return_value={"status": "complete"}) as start, \
                patch("src.evaluation.execute_run") as evaluation, \
                patch("src.readout_experiment.start_readout_experiment") as readout, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main(), 0)
        start.assert_called_once(); evaluation.assert_not_called(); readout.assert_not_called()

    def test_supervised_cli_calls_scripted_http_server_twelve_times(self):
        captured = []
        audit = self.audit
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                captured.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                body = json.dumps({"choices": [{"message": {"content": json.dumps(audit)}, "finish_reason": "stop"}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        self.config = {**self.config, "auditor_base_url": f"http://127.0.0.1:{server.server_port}/v1"}
        path = self.collection / "run.json"
        value = read_json(path); value["config"] = self.config; write_json(path, value)
        def child(command, settings, log):
            return supervise(command, {**settings, "timeout_seconds": 30}, log)
        with patch.dict(os.environ, {"AUDITOR_API_KEY": "controlled-audit-secret"}):
            result = self.start(child)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["trial"]["saved_response_count"], 12)
        self.assertEqual(len(captured), 12)
        self.assertTrue(all(p["messages"][0]["content"] == SHORT_INSTRUCTION for p in captured))
        for file in Path(result["output_folder"]).rglob("*.json"):
            self.assertNotIn("controlled-audit-secret", file.read_text())


if __name__ == "__main__":
    unittest.main()
