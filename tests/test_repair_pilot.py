import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.repair_audit import (audit_payload, local_endpoint, request_json, review_payload,
                              safe_hint, validate_medical_audit, validate_review)
from src.repair_model import entropy_proposal, generate_chat, reasoning_steps
from src.repair_pilot import (matched_readout, pilot_worker, read_pilot_config, run_case,
                              select_training_cases, summarize_pilot)
from src.runtime import write_json


ROOT = Path(__file__).resolve().parents[1]


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = read_pilot_config(ROOT / "configs/repair_pilot.json")
        self.image = Image.new("RGB", (32, 32), "white")
        self.row = {"id": "vqa_rad:train:fixture", "dataset": "vqa_rad", "role": "train",
                    "question": "Which modality is shown?", "answer": "MRI", "image_sha256": "opaque-image-hash"}
        self.initial = "The image is visible. This is CT."
        self.audit = {"action": "repair", "first_error_step": 2, "error_family": "V", "prefix_supported": True,
                      "hint": "Recheck the acquisition features before deciding.", "evidence": "The second step makes an unsupported inference."}
        def continuation(image, question, instruction, prefix, hint, limit):
            cue = "\nLet me recheck the evidence. " + hint + "\n"
            suffix = "I should reconsider the acquisition features."
            return {"preserved_prefix": prefix, "recovery_cue": cue, "assistant_prefill": prefix + cue,
                    "suffix": suffix, "complete_response": prefix + cue + suffix, "raw_response": suffix,
                    "assistant_prefill_verified": True, "input_prefix_unchanged": True,
                    "retained_prefix_unchanged": True, "hit_token_limit": False}
        self.callbacks = {
            "reasoning": Mock(return_value={"raw_response": self.initial, "hit_token_limit": False,
                                            "entropy": {"status": "unavailable", "reason": "unit fixture"}}),
            "readout": Mock(side_effect=[{"raw_response": "CT", "hit_token_limit": False},
                                         {"raw_response": "MRI", "hit_token_limit": False}]),
            "continuation": Mock(side_effect=continuation), "audit": Mock(return_value=self.audit),
            "review": Mock(return_value={"verdict": "supported", "hint_leakage": "none_detected", "reason": "Controlled review fixture."})}

    def case(self, name="case"):
        return run_case(self.row, self.image, "Original reasoning instruction.", self.config, self.root / name, self.callbacks)

    def test_sentence_spans_preserve_every_character(self):
        text = " First sentence.\nNext measurement is 1.5 cm.  Final fragment"
        steps = reasoning_steps(text)
        self.assertEqual("".join(step["text"] for step in steps), text)
        self.assertEqual(len(steps), 3)
        self.assertEqual(text[:steps[1]["start"]], " First sentence.")
        self.assertEqual(reasoning_steps("  "), [])

    def test_entropy_formula_and_constant_profiles(self):
        result = entropy_proposal("A. B. C.", [2, 5, 8], [1., 3., 2.])
        self.assertEqual(result["anchor"], 2)
        self.assertEqual(result["window"], [1, 2, 3])
        self.assertEqual([s["surge"] for s in result["steps"]], [0., 2., 0.])
        constant = entropy_proposal("A. B.", [2, 5], [2., 2.])
        self.assertEqual(constant["anchor"], 1)
        self.assertTrue(all(s["normalized_uncertainty"] == 0 for s in constant["steps"]))
        self.assertEqual(entropy_proposal("A.", None, [1])["status"], "unavailable")
        self.assertEqual(entropy_proposal("A.", [2], [float("nan")])["status"], "unavailable")

    def test_invalid_audits_cannot_select_arbitrary_character_boundaries(self):
        steps = reasoning_steps(self.initial)
        validate_medical_audit(self.audit, steps)
        variants = [{"first_error_step": True}, {"first_error_step": "2"}, {"first_error_step": 0},
                    {"first_error_step": 3}, {"prefix_supported": False}, {"error_family": "format_only"},
                    {"hint": ""}, {"evidence": ""}, {"prefix_end": 1}]
        for replacement in variants:
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                validate_medical_audit({**self.audit, **replacement}, steps)

    def test_nonrepair_auxiliary_fields_are_preserved_but_unused(self):
        for action in ("skip", "unresolved"):
            valid = {**self.audit, "action": action, "first_error_step": 0,
                     "prefix_supported": False, "hint": "Recheck the MRI features."}
            self.assertEqual(validate_medical_audit(valid, reasoning_steps(self.initial)), valid)
            self.callbacks["readout"].side_effect = [{"raw_response": "CT", "hit_token_limit": False}]
            self.callbacks["audit"].return_value = valid
            result = self.case(action)
            self.assertEqual(result["outcome"], "auditor_" + action)
            self.assertFalse(result["hint_used_for_generation"])
            self.assertEqual(json.loads((self.root / action / "audit.json").read_text()), valid)
        self.callbacks["continuation"].assert_not_called()

    def test_hint_guard_rejects_direct_labels_numbers_and_solutions(self):
        self.assertTrue(safe_hint("Recheck the acquisition features.", "MRI")[0])
        for hint, reference in (("Check the MRI features.", "MRI"), ("Count the two structures.", "2"),
                                ("Count the 2 structures.", "2"), ("The answer is MRI.", "MRI"),
                                ("Check the visible features.\nThen answer.", "CT")):
            self.assertFalse(safe_hint(hint, reference)[0])

    def test_training_selection_is_fixed_and_excludes_feedback_and_test(self):
        records = [{"id": f"{dataset}:{role}:{index}", "dataset": dataset, "role": role}
                   for dataset in ("vqa_rad", "slake") for role in ("train", "feedback", "test") for index in range(4)]
        selected = select_training_cases(records, 7)
        self.assertEqual(selected, select_training_cases(list(reversed(records)), 7))
        self.assertEqual([r["dataset"] for r in selected], ["vqa_rad", "vqa_rad", "slake"])
        self.assertTrue(all(r["role"] == "train" for r in selected))
        with self.assertRaises(ValueError):
            select_training_cases([r for r in records if r["role"] != "train"], 7)

    def test_readout_verifier_does_not_guess_semantic_or_numeric_equivalence(self):
        self.assertTrue(matched_readout({"raw_response": " MRI. ", "hit_token_limit": False}, "MRI"))
        self.assertFalse(matched_readout({"raw_response": "CT scan", "hit_token_limit": False}, "CT"))
        self.assertFalse(matched_readout({"raw_response": "two", "hit_token_limit": False}, "2"))
        self.assertFalse(matched_readout({"raw_response": "MRI", "hit_token_limit": True}, "MRI"))

    def test_gold_blind_payload_contains_only_observable_inputs(self):
        payload = audit_payload(self.image, self.row["question"], self.initial, reasoning_steps(self.initial), {}, self.config)
        visible = json.loads(payload["messages"][1]["content"][1]["text"])
        self.assertEqual(set(visible), {"question", "response", "steps", "entropy_proposal"})
        self.assertNotIn("MRI", json.dumps(payload))
        self.assertNotIn(self.row["id"], json.dumps(payload))
        self.assertTrue(payload["messages"][1]["content"][0]["image_url"]["url"].startswith("data:image/png;base64,"))
        review = review_payload(self.image, self.row["question"], "Candidate.", "model-produced answer", self.config)
        self.assertEqual(set(json.loads(review["messages"][1]["content"][1]["text"])),
                         {"question", "candidate_response", "model_answer_readout"})

    def test_successful_path_is_preliminary_and_never_exports_training_data(self):
        result = self.case()
        self.assertEqual(result["outcome"], "preliminary_candidate")
        self.assertEqual(result["prefix_end"], len("The image is visible."))
        self.assertTrue(result["retained_prefix_unchanged"])
        self.assertEqual(result["training_examples_exported"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(result["learnability_gate"]["status"], "not_implemented_in_this_pilot")
        self.assertEqual(result["training_admission"], "not_exported_pending_full_admission_checks")
        args = self.callbacks["continuation"].call_args.args
        self.assertEqual(args[3], "The image is visible.")
        self.assertEqual(args[4], self.audit["hint"])
        self.assertNotIn("MRI", json.dumps(args[1:]))
        self.assertEqual(self.callbacks["review"].call_count, 1)

    def test_first_step_error_records_full_regeneration(self):
        self.callbacks["audit"].return_value = {**self.audit, "first_error_step": 1}
        result = self.case()
        self.assertEqual(result["prefix_end"], 0)
        self.assertEqual(result["repair_mode"], "full_regeneration")

    def test_initial_match_skips_audit_but_is_not_admitted_for_training(self):
        self.callbacks["readout"].side_effect = [{"raw_response": "MRI", "hit_token_limit": False}]
        result = self.case()
        self.assertEqual(result["outcome"], "initial_answer_match")
        self.callbacks["audit"].assert_not_called()
        self.callbacks["continuation"].assert_not_called()
        self.assertEqual(result["training_examples_exported"], 0)

    def test_skip_unresolved_and_invalid_audits_stop_continuation(self):
        for index, action in enumerate(("skip", "unresolved", "invalid")):
            self.callbacks["readout"].side_effect = [{"raw_response": "CT", "hit_token_limit": False}]
            self.callbacks["audit"].return_value = ({**self.audit, "action": action, "first_error_step": 0, "hint": ""}
                                                    if action != "invalid" else {"prefix_end": 1})
            result = self.case(str(index))
            self.assertIn(result["outcome"], ("auditor_skip", "auditor_unresolved", "rejected_invalid_audit"))
        self.callbacks["continuation"].assert_not_called()

    def test_reference_bearing_hint_is_rejected_before_primary_continuation(self):
        self.callbacks["audit"].return_value = {**self.audit, "hint": "Recheck the MRI features."}
        self.assertEqual(self.case()["outcome"], "rejected_hint")
        self.callbacks["continuation"].assert_not_called()

    def test_corrupted_prefix_truncation_and_unchanged_suffix_are_rejected(self):
        original = self.callbacks["continuation"].side_effect
        for index, change in enumerate(({"assistant_prefill_verified": False}, {"hit_token_limit": True},
                                         {"suffix": " This is CT.", "raw_response": " This is CT."})):
            self.callbacks["readout"].side_effect = [{"raw_response": "CT", "hit_token_limit": False}]
            def altered(*args, replacement=change):
                value = original(*args); value.update(replacement)
                value["complete_response"] = value["assistant_prefill"] + value["suffix"]
                return value
            self.callbacks["continuation"].side_effect = altered
            result = self.case(str(index))
            self.assertIn(result["outcome"], ("rejected_prefix_integrity", "rejected_empty_or_truncated_suffix", "rejected_unchanged_erroneous_suffix"))
        self.callbacks["review"].assert_not_called()

    def test_wrong_candidate_answer_stops_before_review(self):
        self.callbacks["readout"].side_effect = [{"raw_response": "CT", "hit_token_limit": False},
                                               {"raw_response": "CT", "hit_token_limit": False}]
        self.assertEqual(self.case()["outcome"], "rejected_candidate_answer_mismatch")
        self.callbacks["review"].assert_not_called()

    def test_unsupported_candidate_review_prevents_preliminary_acceptance(self):
        self.callbacks["review"].return_value = {"verdict": "unsupported", "hint_leakage": "suspected", "reason": "Controlled fixture."}
        self.assertEqual(self.case()["outcome"], "rejected_candidate_integrity")
        with self.assertRaises(ValueError):
            validate_review({"verdict": "supported"})

    def test_service_failure_is_recorded_without_losing_case_evidence(self):
        self.callbacks["audit"].side_effect = RuntimeError("controlled service failure")
        result = self.case()
        self.assertEqual(result["outcome"], "infrastructure_error")
        self.assertTrue((self.root / "case" / "initial_reasoning.json").exists())
        self.assertEqual(json.loads((self.root / "case" / "case.json").read_text())["outcome"], "infrastructure_error")

    def test_config_and_endpoint_reject_remote_or_expanded_scope(self):
        for endpoint in ("http://example.com/v1", "https://example.com/v1", "http://user:pw@127.0.0.1/v1", "http://localhost/v1?key=x"):
            with self.assertRaises(ValueError):
                local_endpoint(endpoint)
        path = self.root / "config.json"
        for change in ({"selection": {"vqa_rad": 3, "slake": 1}}, {"rollout_max_new_tokens": True},
                       {"auditor_api_key": "unexpected"}):
            write_json(path, {**self.config, **change})
            with self.assertRaises(ValueError):
                read_pilot_config(path)

    def test_worker_lock_is_not_removed_by_a_second_worker(self):
        lock = self.root / "run.lock"; lock.write_text("existing")
        with self.assertRaises(FileExistsError):
            pilot_worker(ROOT, self.root, self.root)
        self.assertEqual(lock.read_text(), "existing")

    def test_pilot_summary_never_confuses_preliminary_candidates_with_training(self):
        for index, outcome in enumerate(("initial_answer_match", "auditor_unresolved", "preliminary_candidate"), 1):
            directory = self.root / "cases" / f"{index:02d}"; directory.mkdir(parents=True)
            write_json(directory / "case.json", {"id": str(index), "outcome": outcome, "audit_calls": int(index > 1),
                       "review_calls": int(index == 3), "initial_answer_match": index == 1,
                       "retained_prefix_unchanged": index == 3})
        result = summarize_pilot(self.root, {"selected_ids": ["1", "2", "3"]})
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["preliminary_candidate_count"], 1)
        self.assertEqual(result["training_examples_exported"], 0)
        self.assertFalse(result["adapters_created"])
        failed = summarize_pilot(self.root, {"selected_ids": ["1", "2", "3"]}, {"worker_exit_code": 2, "limit_reached": "time_limit"})
        self.assertEqual(failed["status"], "incomplete")

    def test_local_http_contract_keeps_credentials_out_of_evidence(self):
        audit = self.audit
        captured = {}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass
            def do_POST(self):
                captured["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                captured["authorization"] = self.headers.get("Authorization")
                body = json.dumps({"choices": [{"message": {"content": json.dumps(audit)}}]}).encode()
                self.send_response(200); self.end_headers(); self.wfile.write(body)
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        self.addCleanup(server.server_close); self.addCleanup(server.shutdown)
        config = {**self.config, "auditor_base_url": f"http://127.0.0.1:{server.server_port}/v1"}
        payload = audit_payload(self.image, self.row["question"], self.initial, reasoning_steps(self.initial), {}, config)
        with patch.dict(os.environ, {"AUDITOR_API_KEY": "controlled-secret", "HTTP_PROXY": "http://127.0.0.1:1"}):
            result = request_json(payload, config, self.root, "audit")
        self.assertEqual(result, audit)
        self.assertEqual(captured["body"], payload)
        self.assertEqual(captured["authorization"], "Bearer controlled-secret")
        for path in self.root.glob("*.json"):
            self.assertNotIn("controlled-secret", path.read_text())

    def test_model_rejects_a_changed_prefill_before_generation(self):
        processor = Mock(); processor.apply_chat_template.return_value = "changed"
        model = Mock()
        with self.assertRaisesRegex(ValueError, "changed the assistant prefill"):
            generate_chat(model, processor, [], 4, prefill="original")
        model.generate.assert_not_called()

    def test_repair_cli_is_separate_from_evaluation_and_adapter_training(self):
        import evaluate
        with patch("sys.argv", ["evaluate.py", "repair-pilot"]), \
                patch("src.repair_pilot.start_repair_pilot", return_value={"status": "complete"}) as start, \
                patch("src.evaluation.execute_run") as evaluate_run, \
                patch("src.model.training_probe") as train, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main(), 0)
        start.assert_called_once(); evaluate_run.assert_not_called(); train.assert_not_called()


if __name__ == "__main__":
    unittest.main()
