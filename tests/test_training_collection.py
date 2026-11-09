import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.answer_diagnostic import PLAIN_SCORING_ID
from src.benchmarks import decoded_sha256, file_sha256, load_bundle, write_jsonl
from src.evaluation import source_fingerprints
from src.repair_audit import audit_payload
from src.repair_model import READOUT_INSTRUCTION, reasoning_steps
from src.repair_pilot import read_pilot_config, select_training_cases
from src.runtime import read_settings, write_json
from src.training_collection import (LEGACY_SOURCES, classify_readout, collect_cases, collection_worker,
                                     read_json, select_collection_cases, start_collection,
                                     tree_hashes, validate_previous)


ROOT = Path(__file__).resolve().parents[1]


class TrainingCollectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.project = self.root / "project"
        shutil.copytree(ROOT / "src", self.project / "src", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy2(ROOT / "evaluate.py", self.project / "evaluate.py")
        self.settings = read_settings(ROOT / "settings.json")
        self.config = read_pilot_config(ROOT / "configs/repair_pilot.json")
        self.data = self.root / "data"
        (self.data / "images").mkdir(parents=True)
        self.records = []
        for dataset in ("vqa_rad", "slake"):
            for role, count in (("train", 8), ("feedback", 1), ("test", 1)):
                for index in range(count):
                    number = len(self.records)
                    image = Image.new("RGB", (16, 16), (number * 11, 30, 50))
                    path = self.data / "images" / f"{number}.png"
                    image.save(path)
                    self.records.append({"id": f"{dataset}:{role}:{index}", "dataset": dataset, "role": role,
                        "source_split": role, "question": "Which modality is shown?", "answer": "MRI",
                        "answer_type": "open", "image_path": f"images/{number}.png",
                        "image_sha256": decoded_sha256(image), "image_file_sha256": file_sha256(path)})
        for role in ("train", "feedback", "test"):
            write_jsonl(self.data / (role + ".jsonl"), [r for r in self.records if r["role"] == role])
        write_jsonl(self.data / "exclusions.jsonl", [])
        self.metadata = {"sources": {}, "protocol": {"prompt": "Original reasoning instruction."},
            "counts": {role: sum(r["role"] == role for r in self.records) for role in ("train", "feedback", "test")},
            "manifest_sha256": {role: file_sha256(self.data / (role + ".jsonl")) for role in ("train", "feedback", "test")},
            "exclusions_sha256": file_sha256(self.data / "exclusions.jsonl")}
        write_json(self.data / "dataset.json", self.metadata)
        self.pilot = self.root / "previous"
        self.pilot.mkdir()
        self.old_rows = select_training_cases(self.records, 7)
        hashes = source_fingerprints(self.project)
        hashes.update(LEGACY_SOURCES)
        write_json(self.pilot / "run.json", {"pilot_id": "training-repair-pilot-v1", "role": "train",
            "settings": self.settings, "config": self.config, "selected_ids": [r["id"] for r in self.old_rows],
            "dataset_metadata_sha256": file_sha256(self.data / "dataset.json"),
            "source_sha256": {name.replace("/", "\\"): digest for name, digest in hashes.items()},
            "reasoning_instruction": self.metadata["protocol"]["prompt"],
            "readout_instruction": READOUT_INSTRUCTION, "verifier_id": PLAIN_SCORING_ID})
        for index, row in enumerate(self.old_rows, 1):
            folder = self.pilot / "cases" / f"{index:02d}"
            folder.mkdir(parents=True)
            initial = {"raw_response": "Visible scan. This is CT.", "hit_token_limit": False}
            steps = reasoning_steps(initial["raw_response"])
            request = {"question": row["question"], "image_sha256": row["image_sha256"],
                       "reasoning_instruction": self.metadata["protocol"]["prompt"]}
            write_json(folder / "initial_request.json", request)
            write_json(folder / "initial_reasoning.json", initial)
            write_json(folder / "reasoning_steps.json", steps)
            write_json(folder / "initial_readout_request.json", {**request,
                       "assistant_reasoning": initial["raw_response"], "readout_instruction": READOUT_INSTRUCTION})
            write_json(folder / "initial_readout.json", {"raw_response": row["question"] if index == 1 else "CT", "hit_token_limit": False})
            write_json(folder / "verification.json", {"reference": row["answer"], "scoring_id": PLAIN_SCORING_ID,
                       "initial_answer_match": False, "candidate_answer_match": None})
            with Image.open(self.data / row["image_path"]) as image:
                payload = audit_payload(image, row["question"], initial["raw_response"], steps, {"status": "unavailable"}, self.config)
            write_json(folder / "audit_request.json", payload)
            audit = {"action": "unresolved" if index == 1 else "skip", "first_error_step": 0,
                     "error_family": "R", "prefix_supported": False, "hint": "Unused fixture text.",
                     "evidence": "Controlled test only; not a medical assessment."}
            envelope = {"choices": [{"message": {"content": json.dumps(audit)}, "finish_reason": "stop"}]}
            write_json(folder / "audit_response.json", {"response_text": json.dumps(envelope)})
            write_json(folder / "case.json", {"id": row["id"], "dataset": row["dataset"], "role": "train",
                       "outcome": "rejected_invalid_audit", "audit_calls": 1, "review_calls": 0,
                       "optimizer_steps": 0, "training_examples_exported": 0})
        self.audit = {"action": "skip", "first_error_step": 0, "error_family": "R", "prefix_supported": False,
                      "hint": "Unused hint.", "evidence": "Controlled fixture."}
        self.reasoning = Mock(return_value={"raw_response": "The visible scan is CT.", "hit_token_limit": False})
        self.readout = Mock(return_value={"raw_response": "CT", "hit_token_limit": False})
        self.request = Mock(return_value=self.audit)
        self.continuation = Mock()

    def validate(self):
        return validate_previous(self.project, self.data, self.pilot, self.settings, self.config, self.metadata, self.records)

    def start(self, supervisor=None):
        def execute(command, settings, log):
            directory = Path(command[-1])
            code = collection_worker(self.project, self.data, directory)
            return {"worker_exit_code": code, "limit_reached": None, "peak_process_tree_rss_bytes": 100,
                    "elapsed_seconds": .1}
        model = Mock()
        model.parameters.return_value = [SimpleNamespace(numel=lambda: 10, element_size=lambda: 4)]
        with patch("src.model.load_primary", return_value=(model, Mock())) as load, \
                patch("src.repair_model.generate_reasoning", self.reasoning), \
                patch("src.repair_model.answer_readout", self.readout), \
                patch("src.repair_model.continue_reasoning", self.continuation), \
                patch("src.repair_audit.request_json", self.request), \
                patch("src.training_collection.supervise", side_effect=supervisor or execute), \
                patch("psutil.virtual_memory", return_value=SimpleNamespace(available=32 * 1024**3, total=32 * 1024**3)), \
                contextlib.redirect_stdout(io.StringIO()):
            result = start_collection(self.project, self.data, self.pilot, self.settings, self.config)
        return result, load

    def test_selection_is_balanced_fixed_training_only_and_nine_new(self):
        selected = select_collection_cases(self.records, 7)
        self.assertEqual(selected, select_collection_cases(list(reversed(self.records)), 7))
        self.assertEqual([r["dataset"] for r in selected].count("slake"), 6)
        self.assertTrue(all(r["role"] == "train" for r in selected))
        self.assertEqual(len({r["id"] for r in selected} - {r["id"] for r in self.old_rows}), 9)
        changed = [{**r, "answer": "Different reference", "question": "Different question"} for r in self.records]
        self.assertEqual([r["id"] for r in selected], [r["id"] for r in select_collection_cases(changed, 7)])
        with self.assertRaises(ValueError):
            select_collection_cases([r for r in self.records if r["dataset"] != "slake"], 7)

    def test_echoes_are_diagnostic_without_relaxing_lexical_matching(self):
        def label(answer, reference="yes", truncated=False):
            return classify_readout({"raw_response": answer, "hit_token_limit": truncated}, "Is there a finding?", reference)
        self.assertEqual(label(" Is there a finding? "), "question_echo")
        self.assertEqual(label("Yes."), "answer_match")
        self.assertEqual(label("The answer is yes."), "answer_mismatch")
        self.assertEqual(label("T2-weighted MRI scan", "mr flair"), "answer_mismatch")
        self.assertEqual(label("yes", truncated=True), "truncated_readout")
        self.assertEqual(label("  "), "empty_readout")
        self.assertEqual(classify_readout(None, "Question?", "yes"), "not_recorded")

    def test_reuse_validates_legacy_sources_and_preserves_raw_nonrepair_decisions(self):
        before = tree_hashes(self.pilot)
        _, hashes, reused = self.validate()
        self.assertEqual(hashes, before)
        self.assertEqual(tree_hashes(self.pilot), before)
        self.assertEqual([r["outcome"] for r in reused.values()], ["auditor_unresolved", "auditor_skip", "auditor_skip"])
        self.assertTrue(all(r["original_outcome"] == "rejected_invalid_audit" for r in reused.values()))

    def test_changed_controls_fail_before_any_generation(self):
        for key, value in (("settings", {**self.settings, "seed": 8}),
                           ("config", {**self.config, "readout_max_new_tokens": 16}),
                           ("reasoning_instruction", "Changed prompt"), ("verifier_id", "changed")):
            path = self.pilot / "run.json"
            original = read_json(path)
            write_json(path, {**original, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate()
            write_json(path, original)
        self.reasoning.assert_not_called()

    def test_unknown_source_change_and_auditor_prompt_change_are_rejected(self):
        path = self.pilot / "run.json"
        original = read_json(path)
        value = read_json(path)
        value["source_sha256"]["src\\repair_model.py"] = "0" * 64
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "implementation differs"):
            self.validate()
        write_json(path, original)
        path = self.pilot / "cases/01/audit_request.json"
        value = read_json(path); value["messages"][0]["content"] = "Changed audit instructions"
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "auditor request differs"):
            self.validate()

    def test_corrupted_case_role_reference_or_span_cannot_be_reused(self):
        for name, change in (("case.json", {"role": "test"}), ("verification.json", {"reference": "CT"}),
                              ("initial_request.json", {"image_sha256": "changed"})):
            path = self.pilot / "cases/01" / name
            original = read_json(path); write_json(path, {**original, **change})
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.validate()
            write_json(path, original)

    def test_archived_repair_cannot_be_treated_as_a_reused_success(self):
        path = self.pilot / "cases/01/audit_response.json"
        audit = {**self.audit, "action": "repair", "first_error_step": 1,
                 "prefix_supported": True, "hint": "Recheck the visible features."}
        write_json(path, {"response_text": json.dumps({"choices": [{"message": {"content": json.dumps(audit)}}]})})
        with self.assertRaisesRegex(ValueError, "only the three recorded non-repair"):
            self.validate()

    def test_full_collection_reuses_three_and_calls_model_only_nine_times(self):
        before = tree_hashes(self.pilot)
        result, load = self.start()
        self.assertEqual(result["status"], "complete")
        self.assertEqual((result["recorded_count"], result["reused_count"], result["new_case_count"]), (12, 3, 9))
        self.assertEqual(result["initial_readout_categories"], {"question_echo": 1, "answer_mismatch": 11})
        self.assertEqual(result["audit_calls_new"], 9)
        self.assertEqual(result["historical_audit_calls_reused"], 3)
        self.assertEqual(result["potential_candidate_count"], 0)
        self.assertEqual(result["next_action"], "reassess_model_and_auditor_before_training")
        self.assertEqual(result["training_examples_exported"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertFalse(result["adapters_created"])
        self.assertEqual(self.reasoning.call_count, 9)
        self.assertEqual(self.readout.call_count, 9)
        self.assertEqual(self.request.call_count, 9)
        self.continuation.assert_not_called(); load.assert_called_once()
        self.assertEqual(tree_hashes(self.pilot), before)
        directory = Path(result["output_folder"])
        self.assertEqual(tree_hashes(directory / "previous_pilot"), before)
        for record in result["cases"]:
            if record["reused"]:
                self.assertEqual(record["original_outcome"], "rejected_invalid_audit")
        for call in self.reasoning.call_args_list:
            self.assertEqual(call.args[-1], self.config["rollout_max_new_tokens"])
            self.assertNotIn("MRI", json.dumps(call.args[3:]))
        for call in self.request.call_args_list:
            self.assertNotIn('"reference"', json.dumps(call.args[0]))
        self.assertFalse((directory / "run.lock").exists())

    def test_correct_rollouts_are_indexed_but_remain_unadmitted(self):
        self.readout.return_value = {"raw_response": "MRI", "hit_token_limit": False}
        result, _ = self.start()
        self.assertEqual(result["potential_candidate_count"], 9)
        self.request.assert_not_called(); self.continuation.assert_not_called()
        bank = read_json(Path(result["output_folder"]) / "candidate_bank.json")
        self.assertEqual(bank["training_targets_exported"], 0)
        self.assertTrue(all(c["admission_status"] == "pending_full_admission_checks" for c in bank["candidates"]))

    def test_repair_attempts_are_retained_even_when_hint_is_rejected(self):
        self.request.return_value = {**self.audit, "action": "repair", "first_error_step": 1,
                                    "prefix_supported": True, "hint": "Recheck the MRI features."}
        result, _ = self.start()
        self.assertEqual(result["repair_decision_count"], 9)
        self.assertEqual(result["potential_candidate_count"], 0)
        attempts = read_json(Path(result["output_folder"]) / "repair_attempts.json")["attempts"]
        self.assertTrue(all(a["outcome"] == "rejected_hint" and not a["continuation_performed"] for a in attempts))
        self.continuation.assert_not_called()

    def test_valid_repair_paths_are_retained_without_training_admission(self):
        self.request.side_effect = lambda payload, *args: ({"verdict": "supported", "hint_leakage": "none_detected",
            "reason": "Controlled review fixture."} if payload["response_format"]["json_schema"]["name"] == "medical_repair_review" else
            {**self.audit, "action": "repair", "first_error_step": 1, "prefix_supported": True,
             "hint": "Recheck the acquisition features."})
        self.readout.side_effect = [{"raw_response": answer, "hit_token_limit": False} for _ in range(9) for answer in ("CT", "MRI")]
        def continuation(model, processor, image, question, instruction, prefix, hint, limit):
            cue = "\nLet me recheck the evidence. " + hint + "\n"
            suffix = "Reconsidered the visible acquisition features."
            return {"preserved_prefix": prefix, "recovery_cue": cue, "assistant_prefill": prefix + cue,
                    "complete_response": prefix + cue + suffix, "suffix": suffix, "hit_token_limit": False,
                    "assistant_prefill_verified": True, "input_prefix_unchanged": True, "retained_prefix_unchanged": True}
        self.continuation.side_effect = continuation
        result, _ = self.start()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["potential_candidate_count"], 9)
        self.assertEqual(result["repair_decision_count"], 9)
        self.assertEqual(result["review_calls_new"], 9)
        self.assertEqual(result["optimizer_steps"], 0)
        bank = read_json(Path(result["output_folder"]) / "candidate_bank.json")
        self.assertTrue(all(c["kind"] == "repaired_rollout" for c in bank["candidates"]))
        for call in self.continuation.call_args_list:
            self.assertNotIn("MRI", json.dumps(call.args[3:]))

    def test_infrastructure_failure_stops_without_replacement_questions(self):
        self.reasoning.side_effect = RuntimeError("Controlled inference failure")
        result, _ = self.start()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["new_case_count"], 1)
        self.assertEqual(result["reused_count"], 3)
        self.assertEqual(result["recorded_count"], 4)
        self.assertEqual(result["next_action"], "finish_or_investigate_incomplete_collection")
        self.assertEqual(self.reasoning.call_count, 1)

    def test_time_limit_retains_reused_evidence_and_reports_incomplete(self):
        def timed_out(*args):
            return {"worker_exit_code": 2, "limit_reached": "time_limit", "peak_process_tree_rss_bytes": 0}
        result, load = self.start(timed_out)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["reused_count"], 3)
        self.assertEqual(result["new_case_count"], 0)
        load.assert_not_called()

    def test_worker_checks_copied_evidence_and_preserves_existing_lock(self):
        result, _ = self.start()
        directory = Path(result["output_folder"])
        lock = directory / "run.lock"; lock.write_text("another worker")
        with self.assertRaises(FileExistsError):
            collection_worker(self.project, self.data, directory)
        self.assertEqual(lock.read_text(), "another worker"); lock.unlink()
        (directory / "previous_pilot/run.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed after creation"), patch("src.model.load_primary") as load:
            collection_worker(self.project, self.data, directory)
        load.assert_not_called()
        self.assertFalse(lock.exists())

    def test_collection_cli_cannot_start_adapter_training_or_evaluation(self):
        import evaluate
        with patch("sys.argv", ["evaluate.py", "collect-training", "--previous-pilot", str(self.pilot)]), \
                patch("src.training_collection.start_collection", return_value={"status": "complete"}) as start, \
                patch("src.evaluation.execute_run") as evaluation, patch("src.model.training_probe") as train, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main(), 0)
        start.assert_called_once(); evaluation.assert_not_called(); train.assert_not_called()

    def test_supervised_cli_worker_completes_with_scripted_model_only(self):
        from src.runtime import supervise
        harness = self.root / "scripted_worker.py"
        harness.write_text('''import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(sys.argv.pop(1)).parent))
import evaluate
model = Mock()
model.parameters.return_value = [SimpleNamespace(numel=lambda: 10, element_size=lambda: 4)]
with patch("src.model.load_primary", return_value=(model, Mock())), \\
     patch("src.repair_model.generate_reasoning", return_value={"raw_response": "Controlled reasoning fixture.", "hit_token_limit": False}), \\
     patch("src.repair_model.answer_readout", return_value={"raw_response": "MRI", "hit_token_limit": False}), \\
     patch("src.repair_audit.request_json", side_effect=RuntimeError("Unexpected auditor call in scripted fixture")):
    raise SystemExit(evaluate.main())
''', encoding="utf-8")
        def child(command, settings, log):
            return supervise([command[0], str(harness), *command[1:]], {**settings, "timeout_seconds": 30}, log)
        result, _ = self.start(child)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["new_case_count"], 9)
        self.assertEqual(result["potential_candidate_count"], 9)
        self.assertEqual(result["worker"]["worker_exit_code"], 0)
        self.assertEqual(result["training_examples_exported"], 0)
        console = (Path(result["output_folder"]) / "console.log").read_text()
        self.assertEqual(console.count("collecting new training evidence"), 9)
        self.assertEqual(console.count("reused saved non-repair decision"), 3)


if __name__ == "__main__":
    unittest.main()
