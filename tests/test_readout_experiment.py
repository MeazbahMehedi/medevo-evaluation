import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import test_training_collection as collection_fixtures
from src.readout_experiment import (INSTRUCTION, TOKEN_LIMIT, readout_request, readout_worker,
                                    start_readout_experiment, validate_collection)
from src.runtime import supervise, write_json
from src.training_collection import read_json, tree_hashes


class ReadoutExperimentTests(unittest.TestCase):
    def setUp(self):
        fixture = collection_fixtures.TrainingCollectionTests(
            methodName="test_selection_is_balanced_fixed_training_only_and_nine_new")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        self.collection = Path(fixture.start()[0]["output_folder"])
        self.project, self.data = fixture.project, fixture.data
        self.settings = fixture.settings
        self.generate = Mock(return_value={"raw_response": "MRI", "generated_tokens": 2,
                                          "hit_token_limit": False, "generation_seconds": .01})

    def start(self, supervisor=None):
        def execute(command, settings, log):
            code = readout_worker(self.project, Path(command[-1]))
            return {"worker_exit_code": code, "limit_reached": None, "peak_process_tree_rss_bytes": 100}
        model = Mock()
        model.parameters.return_value = [SimpleNamespace(numel=lambda: 10, element_size=lambda: 4)]
        with patch("src.model.load_primary", return_value=(model, Mock())) as load, \
                patch("src.readout_experiment.generate_chat", self.generate), \
                patch("src.repair_model.generate_reasoning") as reasoning, \
                patch("src.repair_audit.request_json") as auditor, \
                patch("src.model.training_probe") as training, \
                patch("src.readout_experiment.supervise", side_effect=supervisor or execute), \
                patch("psutil.virtual_memory", return_value=SimpleNamespace(available=32 * 1024**3, total=32 * 1024**3)), \
                contextlib.redirect_stdout(io.StringIO()):
            result = start_readout_experiment(self.project, self.data, self.collection, self.settings)
        reasoning.assert_not_called(); auditor.assert_not_called(); training.assert_not_called()
        return result, load

    def test_request_is_one_text_message_with_only_question_and_saved_reasoning(self):
        request = readout_request("What is visible?", "The response says aorta.")
        self.assertEqual(request["max_new_tokens"], 32)
        self.assertEqual(len(request["messages"]), 1)
        self.assertEqual(request["messages"][0]["role"], "user")
        content = request["messages"][0]["content"]
        self.assertEqual(len(content), 1)
        self.assertEqual(set(content[0]), {"type", "text"})
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("The response says aorta.", content[0]["text"])
        self.assertIn(INSTRUCTION, content[0]["text"])
        self.assertIn("do not solve", content[0]["text"].lower())
        with self.assertRaises(ValueError):
            readout_request("Question?", " ")

    def test_completed_comparison_keeps_originals_and_runs_exactly_twelve_readouts(self):
        before = tree_hashes(self.collection)
        result, load = self.start()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["new_readout_calls_recorded"], 12)
        self.assertEqual(self.generate.call_count, 12)
        load.assert_called_once()
        self.assertEqual(tree_hashes(self.collection), before)
        self.assertEqual(result["metrics"]["original_match_count"], 0)
        self.assertEqual(result["metrics"]["trial_match_count"], 12)
        self.assertEqual(result["by_dataset"]["slake"]["selected_count"], 6)
        self.assertEqual(result["by_dataset"]["vqa_rad"]["selected_count"], 6)
        for call in self.generate.call_args_list:
            messages, cap = call.args[2:]
            self.assertEqual(cap, TOKEN_LIMIT)
            self.assertEqual(len(messages), 1)
            self.assertEqual(messages[0]["content"][0]["type"], "text")
            # MRI occurs only in these fixtures' verifier files, never in their saved reasoning.
            self.assertNotIn("MRI", messages[0]["content"][0]["text"])

    def test_reference_match_alone_does_not_admit_or_certify_a_candidate(self):
        result, _ = self.start()
        self.assertEqual(result["metrics"]["trial_whole_answer_em_pct"], 100)
        self.assertFalse(result["automatic_candidate_admission"])
        self.assertEqual(result["training_examples_exported"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertFalse(result["adapters_created"])
        report = read_json(Path(result["output_folder"]) / "comparison.json")
        for row in report["cases"]:
            self.assertEqual(row["readout_fidelity"], "requires_review_against_saved_reasoning")
            self.assertEqual(row["training_admission"], "not_assessed")

    def test_truncated_readouts_cannot_be_counted_as_matches(self):
        self.generate.return_value.update(hit_token_limit=True, generated_tokens=32)
        result, _ = self.start()
        self.assertEqual(result["metrics"]["trial_match_count"], 0)
        report = read_json(Path(result["output_folder"]) / "comparison.json")
        self.assertTrue(all(r["trial_category"] == "truncated_readout" for r in report["cases"]))

    def test_failure_preserves_partial_results_without_full_accuracy_or_retry(self):
        good = self.generate.return_value
        self.generate.side_effect = [good, good, RuntimeError("Controlled readout failure")]
        result, _ = self.start()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["new_readout_calls_recorded"], 2)
        self.assertIsNone(result["metrics"]["trial_whole_answer_em_pct"])
        self.assertEqual(self.generate.call_count, 3)
        self.assertIn("Controlled readout failure", result["error"])

    def test_timeout_before_worker_start_preserves_original_evidence(self):
        def timeout(*args):
            return {"worker_exit_code": 2, "limit_reached": "time_limit", "peak_process_tree_rss_bytes": 0}
        result, load = self.start(timeout)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["new_readout_calls_recorded"], 0)
        self.assertIsNone(result["metrics"]["trial_whole_answer_em_pct"])
        load.assert_not_called()

    def test_changed_settings_budget_source_or_incomplete_collection_are_rejected(self):
        changes = (("run.json", "settings", {**self.settings, "cpu_threads": 8}),
                   ("run.json", "config", {**self.fixture.config, "readout_max_new_tokens": 16}),
                   ("summary.json", "status", "incomplete"))
        for name, key, value in changes:
            path = self.collection / name
            before = read_json(path)
            write_json(path, {**before, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_collection(self.project, self.data, self.collection, self.settings)
            write_json(path, before)
        path = self.collection / "run.json"
        value = read_json(path); value["source_sha256"]["src/model.py"] = "0" * 64
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "implementation changed"):
            validate_collection(self.project, self.data, self.collection, self.settings)

    def test_foreign_question_reference_or_role_is_rejected_before_loading_model(self):
        changes = (("initial_request.json", "question", "Different question"),
                   ("verification.json", "reference", "Different reference"), ("case.json", "role", "test"))
        for name, key, value in changes:
            path = self.collection / "cases/01" / name
            before = read_json(path); write_json(path, {**before, key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_collection(self.project, self.data, self.collection, self.settings)
            write_json(path, before)

    def test_locked_original_collection_cannot_be_used(self):
        (self.collection / "run.lock").write_text("active")
        with self.assertRaisesRegex(ValueError, "worker has stopped"):
            validate_collection(self.project, self.data, self.collection, self.settings)

    def test_worker_preserves_existing_lock_and_rejects_changed_requests(self):
        result, _ = self.start()
        directory = Path(result["output_folder"])
        lock = directory / "run.lock"; lock.write_text("existing")
        with self.assertRaises(FileExistsError):
            readout_worker(self.project, directory)
        self.assertEqual(lock.read_text(), "existing"); lock.unlink()
        path = directory / "inputs.json"
        inputs = read_json(path)
        inputs[0]["request"]["messages"][0]["content"][0]["text"] += " altered"
        write_json(path, inputs)
        with patch("src.model.load_primary") as load:
            self.assertEqual(readout_worker(self.project, directory), 2)
        load.assert_not_called()
        self.assertFalse(lock.exists())

    def test_completed_worker_cannot_repeat_the_twelve_calls(self):
        result, _ = self.start()
        with patch("src.model.load_primary") as load:
            self.assertEqual(readout_worker(self.project, Path(result["output_folder"])), 2)
        load.assert_not_called()

    def test_cli_routes_only_to_the_readout_experiment(self):
        import evaluate
        with patch("sys.argv", ["evaluate.py", "compare-readouts", "--collection-dir", str(self.collection)]), \
                patch("src.readout_experiment.start_readout_experiment", return_value={"status": "complete"}) as start, \
                patch("src.training_collection.start_collection") as collection, \
                patch("src.evaluation.execute_run") as evaluation, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main(), 0)
        start.assert_called_once(); collection.assert_not_called(); evaluation.assert_not_called()

    def test_supervised_cli_worker_with_scripted_model_makes_twelve_calls(self):
        harness = self.fixture.root / "scripted_readout_worker.py"
        harness.write_text('''import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
sys.path.insert(0, str(Path(sys.argv.pop(1)).parent))
import evaluate
model = Mock()
model.parameters.return_value = [SimpleNamespace(numel=lambda: 10, element_size=lambda: 4)]
with patch("src.model.load_primary", return_value=(model, Mock())), \\
     patch("src.readout_experiment.generate_chat", return_value={"raw_response": "MRI", "generated_tokens": 2, "hit_token_limit": False}), \\
     patch("src.repair_audit.request_json", side_effect=RuntimeError("Unexpected auditor call")):
    raise SystemExit(evaluate.main())
''', encoding="utf-8")
        def child(command, settings, log):
            return supervise([command[0], str(harness), *command[1:]], {**settings, "timeout_seconds": 30}, log)
        result, _ = self.start(child)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["new_readout_calls_recorded"], 12)
        self.assertEqual(result["worker"]["worker_exit_code"], 0)
        console = (Path(result["output_folder"]) / "console.log").read_text()
        self.assertEqual(console.count("text-only readout recorded"), 12)


if __name__ == "__main__":
    unittest.main()
