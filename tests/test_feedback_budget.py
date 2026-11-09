import contextlib
import io
import json
from pathlib import Path
import unittest

from PIL import Image

import compare_feedback_budget as budget
import evaluate_feedback as original
from feedback_scoring import score_formatted, unwrap_formatting
from src.benchmarks import decoded_sha256, file_sha256, write_jsonl
from src.runtime import write_json
from src.training_collection import read_json, tree_hashes
import test_feedback_baseline as fixtures


class FormattingScorerTests(unittest.TestCase):
    def test_nested_wrappers_only_and_historical_score_preserved(self):
        text = r"\boxed{\textbf{\text{yes}}}"
        self.assertTrue(score_formatted(text, "yes")["strict_correct"])
        self.assertFalse(original.score_answer(text, "yes")["strict_correct"])
        self.assertTrue(score_formatted(r"\boxed{\text{}}", "yes")["parse_failed"])
        self.assertFalse(score_formatted(r"\boxed{\mathrm{yes}}", "yes")["strict_correct"])
        self.assertEqual(unwrap_formatting(r"\text{unclosed"), r"\text{unclosed")
        self.assertFalse(score_formatted(r"\text{yes}", "yes")["strict_correct"])


class FeedbackBudgetTests(unittest.TestCase):
    def setUp(self):
        fixture = fixtures.FeedbackBaselineTests(methodName="test_twenty_reference_free_requests_fixed_selection_and_preserved_sources")
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.f = fixture.f
        for dataset, target, red in (("vqa_rad", 79, 45), ("slake", 40, 85)):
            for i in range(target - 10):
                image = Image.new("RGB", (8, 8), (red, i, 210))
                path = self.f.data / "images" / f"expanded_{dataset}_{i}.png"
                image.save(path)
                self.f.rows.append({"id": f"{dataset}:feedback:expanded{i}", "dataset": dataset, "role": "feedback",
                                    "question": "Which organ is shown?", "answer": "kidney", "answer_type": "open",
                                    "image_path": "images/" + path.name, "image_sha256": decoded_sha256(image),
                                    "image_file_sha256": file_sha256(path)})
        write_jsonl(self.f.data / "feedback.jsonl", [r for r in self.f.rows if r["role"] == "feedback"])
        metadata = read_json(self.f.data / "dataset.json")
        metadata["counts"]["feedback"] = 119
        metadata["manifest_sha256"]["feedback"] = file_sha256(self.f.data / "feedback.jsonl")
        write_json(self.f.data / "dataset.json", metadata)
        baseline_calls = [0]
        def baseline_transport(route, payload=None, timeout=20):
            result = fixture.transport(route, payload, timeout)
            if route == "/api/chat":
                baseline_calls[0] += 1
                if baseline_calls[0] <= 4:
                    result.update(done_reason="length", eval_count=512)
                    result["message"]["content"] = "Incomplete output"
            return result
        with contextlib.redirect_stdout(io.StringIO()):
            original.run_feedback(self.f.root, self.f.data, baseline_transport, available_ram=lambda: 16)
        self.baseline = self.f.root / "outputs/qwen35_feedback_20_v1"
        self.plan_dir = self.f.root / "outputs" / budget.PLAN_FOLDER
        self.calls = []
        self.fail_at = None
        self.unboxed = 0
        self.fixture = fixture

    def transport(self, route, payload=None, timeout=20):
        self.calls.append((route, payload, timeout))
        if route == "/api/chat":
            n = sum(c[0] == route for c in self.calls)
            if n == self.fail_at: raise TimeoutError("Controlled timeout")
            return {"model": original.MODEL, "done": True, "done_reason": "stop", "eval_count": 600,
                    "message": {"content": "kidney" if n <= self.unboxed else r"\boxed{\text{kidney}}"}}
        return self.fixture.transport(route, payload, timeout)

    def inspect(self):
        return budget.inspect_selection(self.f.root, self.f.data, self.baseline)

    def run_batch(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return budget.run_comparison(self.f.root, self.plan_dir, self.transport, available_ram=lambda: 16)

    def test_offline_audit_selects_79_plus_40_and_reuses_plan_without_inference(self):
        before = tree_hashes(self.f.data), tree_hashes(self.baseline)
        result = self.inspect()
        self.assertEqual(result["counts"], {"vqa_rad": 79, "slake": 40})
        self.assertEqual(result["new_generation_calls"], 0)
        self.assertEqual(result["test_role_overlap"], 0)
        self.assertEqual(result["training_role_overlap"], 0)
        self.assertEqual(result, self.inspect())
        self.assertFalse(self.calls)
        self.assertEqual(before, (tree_hashes(self.f.data), tree_hashes(self.baseline)))

    def test_full_119_run_dual_scoring_paired_outputs_and_only_cap_changes(self):
        self.inspect()
        before = tree_hashes(self.baseline), tree_hashes(self.f.data)
        result = self.run_batch()
        chats = [c for c in self.calls if c[0] == "/api/chat"]
        self.assertEqual(len(chats), 119)
        for _, payload, timeout in chats:
            expected = {**original.OPTIONS, "num_predict": 1024}
            self.assertEqual(payload["options"], expected)
            self.assertFalse(payload["think"])
            self.assertEqual(payload["messages"][0]["content"], "Which organ is shown?\n" + original.PROMPT)
            self.assertNotIn("kidney", json.dumps(payload))
            self.assertLessEqual(timeout, 300)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["decision"]["outcome"], "adopt_1024")
        self.assertEqual(result["scorers"][budget.FORMATTED_SCORER]["overall"]["strict_em_accuracy_pct"], 100)
        self.assertEqual(result["scorers"][original.SCORING_ID]["overall"]["strict_em_accuracy_pct"], 0)
        self.assertEqual(result["scorers"][original.SCORING_ID]["overall"]["token_limit_count"], 0)
        self.assertEqual(result["earlier_sixteen_reproduction"]["available_count"], 16)
        self.assertEqual(result["earlier_sixteen_reproduction"]["raw_text_identical_count"], 0)
        self.assertEqual(result["test_generation_requests"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(before, (tree_hashes(self.baseline), tree_hashes(self.f.data)))
        with self.assertRaisesRegex(ValueError, "attempt already exists"):
            self.run_batch()

    def test_twelve_missing_boxes_fails_predeclared_threshold(self):
        self.unboxed = 12
        self.inspect()
        result = self.run_batch()
        self.assertEqual(result["decision"]["missing_boxes"], 12)
        self.assertEqual(result["decision"]["outcome"], "retain_512")

    def test_eleven_missing_boxes_passes_predeclared_threshold(self):
        self.unboxed = 11
        self.inspect()
        self.assertEqual(self.run_batch()["decision"]["outcome"], "adopt_1024")

    def test_partial_run_preserves_evidence_has_no_decision_or_retry(self):
        self.inspect()
        self.fail_at = 3
        result = self.run_batch()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["generation_requests_started"], 3)
        self.assertEqual(result["decision"]["outcome"], "undecided_incomplete")
        self.assertIsNone(result["scorers"][budget.FORMATTED_SCORER]["overall"]["strict_em_accuracy_pct"])
        self.assertTrue((self.f.root / "outputs" / budget.RUN_FOLDER / "cases/001/response.json").exists())

    def test_tampered_plan_blocks_before_service_calls(self):
        self.inspect()
        plan = read_json(self.plan_dir / "plan.json")
        plan["selected_ids"][0] = "vqa_rad:test:0"
        write_json(self.plan_dir / "plan.json", plan)
        with self.assertRaisesRegex(ValueError, "Plan or selection changed"):
            self.run_batch()
        self.assertFalse(self.calls)

    def test_changed_baseline_request_blocks_offline_audit(self):
        path = self.baseline / "cases/01/request.json"
        value = read_json(path)
        value["think"] = True
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "previous request differs"):
            self.inspect()


if __name__ == "__main__":
    unittest.main()
