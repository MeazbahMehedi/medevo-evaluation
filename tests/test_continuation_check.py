import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock

from continue_case import ROOT, approved_proposal, run_attempt, start_check
from src.training_collection import read_json


class ContinuationCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        assets = ROOT / "configs/continuation_case"
        self.visible = read_json(assets / "saved_input.json")
        self.audit = read_json(assets / "approved_audit.json")["legacy_audit"]
        self.prefix = self.visible["response"][:self.visible["steps"][1]["start"]]
        self.request = {"question": self.visible["question"], "reasoning_instruction": "original instruction",
                        "image_sha256": "recorded-image", "preserved_prefix": self.prefix,
                        "minimal_hint": self.audit["hint"]}
        self.reference = "zqxv_controlled_reference"
        cue = "\nLet me recheck the evidence. " + self.audit["hint"] + "\n"
        self.candidate = {"preserved_prefix": self.prefix, "recovery_cue": cue,
                          "assistant_prefill": self.prefix + cue, "suffix": "A new bounded continuation.",
                          "complete_response": self.prefix + cue + "A new bounded continuation.",
                          "assistant_prefill_verified": True, "input_prefix_unchanged": True,
                          "retained_prefix_unchanged": True, "hit_token_limit": False}
        self.continuation = Mock(return_value=self.candidate)
        self.readout = Mock(return_value={"raw_response": "unresolved", "hit_token_limit": False})
        self.callbacks = {"continuation": self.continuation, "readout": self.readout}

    def attempt(self):
        return run_attempt(self.directory, self.request, self.reference, "Original wrong suffix.", self.callbacks)

    def test_one_attempt_preserves_inputs_and_never_sends_reference_to_callbacks(self):
        before = json.dumps(self.request)
        result = self.attempt()
        self.continuation.assert_called_once_with(self.prefix, self.audit["hint"], 128)
        self.readout.assert_called_once_with(self.candidate["complete_response"], 32)
        self.assertNotIn(self.reference, repr(self.continuation.call_args) + repr(self.readout.call_args))
        self.assertEqual(json.dumps(self.request), before)
        self.assertEqual(result["outcome"], "candidate_answer_mismatch")
        self.assertEqual(result["training_examples_admitted"], 0)
        self.assertEqual(read_json(self.directory / "candidate_reasoning.json"), self.candidate)

    def test_matching_answer_is_still_pending_review_without_training(self):
        self.readout.return_value = {"raw_response": self.reference, "hit_token_limit": False}
        result = self.attempt()
        self.assertEqual(result["outcome"], "answer_match_pending_review")
        self.assertEqual(result["training_examples_exported"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertFalse(result["automatic_candidate_admission"])

    def test_changed_prefix_stops_before_readout_and_keeps_bad_output(self):
        self.candidate["complete_response"] = "Changed first sentence."
        result = self.attempt()
        self.assertEqual(result["outcome"], "rejected_prefix_integrity")
        self.readout.assert_not_called()
        self.assertEqual(read_json(self.directory / "candidate_reasoning.json"), self.candidate)

    def test_truncation_stops_without_a_longer_retry(self):
        self.candidate["hit_token_limit"] = True
        self.assertEqual(self.attempt()["outcome"], "rejected_empty_or_truncated_suffix")
        self.continuation.assert_called_once()
        self.readout.assert_not_called()

    def test_model_failure_is_recorded_without_retry_or_readout(self):
        self.continuation.side_effect = RuntimeError("controlled generation failure")
        result = self.attempt()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["continuation_calls_started"], 1)
        self.continuation.assert_called_once()
        self.readout.assert_not_called()
        self.assertIn("controlled generation failure", read_json(self.directory / "summary.json")["error"])

    def test_attempt_and_parent_output_cannot_silently_be_repeated(self):
        self.attempt()
        with self.assertRaises(FileExistsError):
            self.attempt()
        self.continuation.assert_called_once()
        (self.directory / "outputs/continuation_case_v1").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "already exists"):
            start_check(self.directory, self.directory, self.directory, {}, {})

    def test_sealed_proposal_rejects_edited_hint_and_changed_reasoning(self):
        source = ROOT / "configs/continuation_case"
        assets = self.directory / "assets"
        shutil.copytree(source, assets)
        initial = {"raw_response": self.visible["response"]}
        approved_proposal(assets, initial, self.visible["steps"], self.reference)
        with self.assertRaisesRegex(ValueError, "reasoning or step"):
            approved_proposal(assets, {"raw_response": "Different reasoning."}, self.visible["steps"], self.reference)
        path = assets / "approved_audit.json"
        original = read_json(path)
        original["legacy_audit"]["hint"] = "Check a different feature."
        path.write_text(json.dumps(original))
        with self.assertRaisesRegex(ValueError, "sealed audit"):
            approved_proposal(assets, initial, self.visible["steps"], self.reference)

    def test_unchanged_wrong_suffix_is_not_a_repair(self):
        self.candidate["suffix"] = "Original wrong suffix."
        self.candidate["complete_response"] = self.candidate["assistant_prefill"] + self.candidate["suffix"]
        self.assertEqual(self.attempt()["outcome"], "rejected_unchanged_erroneous_suffix")
        self.readout.assert_not_called()


if __name__ == "__main__":
    unittest.main()
