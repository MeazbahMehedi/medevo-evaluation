import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from src.benchmarks import file_sha256
from src.repair_audit import validate_medical_audit
from src.repair_model import reasoning_steps
from src.repair_review import inspect_repair
from src.runtime import write_json


ROOT = Path(__file__).resolve().parents[1]
REPORTED_AUDITS = json.loads((ROOT / "tests/fixtures/nonrepair_audits.json").read_text())


class RepairReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        (self.root / "evaluate.py").write_text("pass\n")
        self.pilot = self.root / "outputs" / "original_pilot"
        self.pilot.mkdir(parents=True)
        ids = ["vqa_rad:train:000201", "vqa_rad:train:001467", "slake:train:4022"]
        write_json(self.pilot / "run.json", {"pilot_id": "training-repair-pilot-v1", "role": "train", "selected_ids": ids})
        self.text = "A controlled first sentence. Another controlled sentence."
        for index, (identity, audit) in enumerate(zip(ids, REPORTED_AUDITS), 1):
            folder = self.pilot / "cases" / f"{index:02d}"; folder.mkdir(parents=True)
            write_json(folder / "case.json", {"id": identity, "role": "train", "outcome": "rejected_invalid_audit"})
            write_json(folder / "initial_reasoning.json", {"raw_response": self.text})
            write_json(folder / "reasoning_steps.json", reasoning_steps(self.text))
            self.save_response(folder, audit)

    def save_response(self, folder, audit):
        envelope = {"choices": [{"message": {"content": json.dumps(audit)}, "finish_reason": "stop"}]}
        write_json(folder / "audit_response.json", {"response_text": json.dumps(envelope), "roundtrip_seconds": .1})

    def test_reported_nonrepair_actions_stay_unchanged_with_zero_continuations(self):
        first = self.pilot / "cases/01"
        write_json(first / "initial_request.json", {"question": "Controlled question?"})
        write_json(first / "initial_readout.json", {"raw_response": "Controlled response"})
        write_json(first / "verification.json", {"reference": "Controlled reference"})
        before = {str(path): file_sha256(path) for path in self.pilot.rglob("*") if path.is_file()}
        with patch("src.model.load_primary") as load, patch("src.repair_model.continue_reasoning") as continue_model:
            result = inspect_repair(self.root, self.pilot)
        self.assertEqual(result["recorded_outcomes"], {"rejected_invalid_audit": 3})
        self.assertEqual(result["review_outcomes"], {"auditor_unresolved": 1, "auditor_skip": 2})
        self.assertEqual(result["nonrepair_decision_count"], 3)
        self.assertEqual(result["schema_valid_repair_decision_count"], 0)
        self.assertEqual(result["new_continuations_performed"], 0)
        self.assertEqual(result["case_details"][0]["initial_answer_readout"], "Controlled response")
        self.assertEqual(result["case_details"][0]["reference"], "Controlled reference")
        load.assert_not_called(); continue_model.assert_not_called()
        for name, digest in before.items():
            self.assertEqual(file_sha256(name), digest)
        report = json.loads((Path(result["review_folder"]) / "audit_review.json").read_text())
        self.assertEqual([row["raw_audit"] for row in report["cases"]], REPORTED_AUDITS)
        self.assertTrue(all(not row["hint_used_for_generation"] for row in report["cases"]))

    def test_real_repair_validation_is_not_relaxed(self):
        steps = reasoning_steps(self.text)
        for audit in REPORTED_AUDITS:
            with self.assertRaises(ValueError):
                validate_medical_audit({**audit, "action": "repair"}, steps)
        with self.assertRaises(ValueError):
            validate_medical_audit({**REPORTED_AUDITS[1], "first_error_step": True}, steps)
        with self.assertRaises(ValueError):
            validate_medical_audit({**REPORTED_AUDITS[1], "first_error_step": 999}, steps)

    def test_valid_repair_is_only_flagged_for_remaining_checks(self):
        audit = {**REPORTED_AUDITS[1], "action": "repair", "first_error_step": 1, "prefix_supported": True,
                 "hint": "Recheck the visible features."}
        self.save_response(self.pilot / "cases/01", audit)
        result = inspect_repair(self.root, self.pilot)
        self.assertEqual(result["schema_valid_repair_decision_count"], 1)
        self.assertEqual(result["new_continuations_performed"], 0)
        self.assertEqual(result["training_examples_exported"], 0)

    def test_malformed_saved_response_is_reported_without_becoming_a_skip(self):
        folder = self.pilot / "cases/02"
        write_json(folder / "audit_response.json", {"response_text": "not JSON"})
        result = inspect_repair(self.root, self.pilot)
        self.assertEqual(result["invalid_saved_audit_count"], 1)
        self.assertEqual(result["review_outcomes"]["invalid_saved_audit"], 1)

    def test_lock_and_mismatched_saved_spans_are_rejected(self):
        lock = self.pilot / "run.lock"; lock.write_text("active")
        with self.assertRaises(ValueError):
            inspect_repair(self.root, self.pilot)
        self.assertTrue(lock.exists()); lock.unlink()
        write_json(self.pilot / "cases/01/reasoning_steps.json", [])
        with self.assertRaisesRegex(ValueError, "steps disagree"):
            inspect_repair(self.root, self.pilot)

    def test_foreign_case_identity_cannot_be_reviewed_as_the_selected_case(self):
        write_json(self.pilot / "cases/01/case.json", {"id": "different", "role": "train", "outcome": "rejected_invalid_audit"})
        with self.assertRaisesRegex(ValueError, "identity or role"):
            inspect_repair(self.root, self.pilot)

    def test_cli_only_reads_existing_pilot(self):
        import evaluate
        with patch("sys.argv", ["evaluate.py", "inspect-repair", "--run-dir", str(self.pilot)]), \
                patch("src.repair_review.inspect_repair", return_value={"status": "complete"}) as review, \
                patch("src.repair_pilot.start_repair_pilot") as start, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main(), 0)
        review.assert_called_once(); start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
