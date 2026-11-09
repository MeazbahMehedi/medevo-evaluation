import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

import evaluate_feedback as feedback
from src.benchmarks import decoded_sha256, file_sha256, load_bundle, write_jsonl
from src.runtime import write_json
from src.training_collection import read_json, tree_hashes
import test_prompt_experiment as fixtures


class FeedbackBaselineTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.PromptExperimentTests(methodName="test_pairing_controls_snapshots_and_original_files_are_preserved")
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        for dataset_index, dataset in enumerate(feedback.DATASETS):
            for index in range(5):
                image = Image.new("RGB", (8, 8), (dataset_index * 100, index * 20, 190))
                path = f.data / "images" / f"extra_{dataset}_{index}.png"
                image.save(path)
                f.rows.append({"id": f"{dataset}:feedback:extra{index}", "dataset": dataset, "role": "feedback",
                               "question": "Which organ is shown?", "answer": "kidney", "answer_type": "open",
                               "image_path": "images/" + path.name, "image_sha256": decoded_sha256(image),
                               "image_file_sha256": file_sha256(path)})
        write_jsonl(f.data / "feedback.jsonl", [r for r in f.rows if r["role"] == "feedback"])
        metadata = read_json(f.data / "dataset.json")
        metadata["counts"]["feedback"] = 20
        metadata["manifest_sha256"]["feedback"] = file_sha256(f.data / "feedback.jsonl")
        write_json(f.data / "dataset.json", metadata)
        self.calls = []
        self.failure_at = None
        self.thinking = {"values": [False, True], "default": True}
        self.digest = "a" * 64

    def transport(self, route, payload=None, timeout=20):
        self.calls.append((route, payload, timeout))
        if route == "/api/version": return {"version": "controlled"}
        if route == "/api/tags": return {"models": [{"name": feedback.MODEL, "digest": self.digest}]}
        if route == "/api/show": return {"capabilities": ["vision", "thinking"], "thinking": self.thinking}
        if route == "/api/ps": return {"models": []}
        if route == "/api/chat":
            count = sum(c[0] == route for c in self.calls)
            if count == self.failure_at: raise TimeoutError("Controlled service failure")
            return {"model": feedback.MODEL, "done": True, "done_reason": "stop", "eval_count": 5,
                    "message": {"content": r"\boxed{kidney}"}, "total_duration": 1000000000}
        raise AssertionError(route)

    def run_batch(self, transport=None, ram=lambda: 16):
        with contextlib.redirect_stdout(io.StringIO()):
            return feedback.run_feedback(self.f.root, self.f.data, transport or self.transport, available_ram=ram)

    def test_twenty_reference_free_requests_fixed_selection_and_preserved_sources(self):
        before = tree_hashes(self.f.data), {d: tree_hashes(p) for d, p in self.f.baselines.items()}
        result = self.run_batch()
        chats = [c for c in self.calls if c[0] == "/api/chat"]
        self.assertEqual(len(chats), 20)
        for _, payload, timeout in chats:
            self.assertFalse(payload["think"])
            self.assertEqual(payload["options"], feedback.OPTIONS)
            self.assertEqual(payload["messages"][0]["content"], "Which organ is shown?\n" + feedback.PROMPT)
            self.assertNotIn("kidney", json.dumps(payload))
            self.assertLessEqual(timeout, 300)
        self.assertEqual(result["overall"]["strict_em_accuracy_pct"], 100)
        self.assertEqual(result["overall"]["boxed_answer_rate_pct"], 100)
        self.assertEqual(result["overall"]["generated_tokens"]["mean"], 5)
        self.assertEqual(result["test_generation_requests"], 0)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertEqual(before, (tree_hashes(self.f.data), {d: tree_hashes(p) for d, p in self.f.baselines.items()}))
        run = read_json(self.f.root / "outputs/qwen35_feedback_20_v1/run.json")
        self.assertEqual(len(set(run["selected_ids"])), 20)
        self.assertTrue(all(":feedback:" in identity for identity in run["selected_ids"]))
        with self.assertRaisesRegex(ValueError, "attempt already exists"):
            self.run_batch()
        self.assertEqual(sum(c[0] == "/api/chat" for c in self.calls), 20)

    def test_timeout_preserves_partial_records_without_retry_or_complete_accuracy(self):
        self.failure_at = 3
        result = self.run_batch()
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["generation_requests_started"], 3)
        self.assertEqual(result["overall"]["generated_count"], 2)
        self.assertIsNone(result["overall"]["strict_em_accuracy_pct"])
        self.assertIsNone(result["overall"]["boxed_answer_rate_pct"])
        self.assertTrue((self.f.root / "outputs/qwen35_feedback_20_v1/cases/01/response.json").exists())

    def test_unsupported_thinking_control_or_low_ram_blocks_before_generation(self):
        self.thinking = {"values": [True]}
        with self.assertRaisesRegex(ValueError, "think=false"):
            self.run_batch()
        self.assertFalse(any(c[0] == "/api/chat" for c in self.calls))
        with self.assertRaisesRegex(RuntimeError, "10 GiB"):
            self.run_batch(ram=lambda: 5)
        self.assertFalse((self.f.root / "outputs/qwen35_feedback_20_v1").exists())

    def test_thinking_leak_stops_and_retains_raw_reply(self):
        def transport(route, payload=None, timeout=20):
            reply = self.transport(route, payload, timeout)
            if route == "/api/chat": reply["message"]["thinking"] = "Unexpected reasoning field"
            return reply
        result = self.run_batch(transport)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(result["generation_requests_started"], 1)
        raw = read_json(self.f.root / "outputs/qwen35_feedback_20_v1/cases/01/response.json")
        self.assertIn("thinking", raw["message"])

    def test_empty_unboxed_and_truncated_answers_are_retained_without_retries(self):
        def transport(route, payload=None, timeout=20):
            reply = self.transport(route, payload, timeout)
            if route == "/api/chat":
                n = sum(c[0] == route for c in self.calls)
                if n == 1: reply["message"]["content"] = ""
                if n == 2: reply["message"]["content"] = "kidney"
                if n == 3: reply.update(done_reason="length", eval_count=512)
            return reply
        result = self.run_batch(transport)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["overall"]["parse_failure_count"], 2)
        self.assertEqual(result["overall"]["token_limit_count"], 1)
        self.assertEqual(result["overall"]["strict_em_accuracy_pct"], 90)
        self.assertEqual(result["generation_requests_started"], 20)

    def test_selection_does_not_expand_into_train_or_test_when_feedback_is_short(self):
        metadata, records = load_bundle(self.f.data)
        records = [r for r in records if r["id"] != "vqa_rad:feedback:extra0"]
        with self.assertRaisesRegex(ValueError, "ten eligible feedback"):
            feedback.select_feedback(metadata, records)

    def test_model_digest_change_invalidates_complete_claim(self):
        tags = 0
        def transport(route, payload=None, timeout=20):
            nonlocal tags
            if route == "/api/tags":
                tags += 1
                if tags > 1: self.digest = "b" * 64
            return self.transport(route, payload, timeout)
        result = self.run_batch(transport)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["overall"]["strict_em_accuracy_pct"])
        self.assertIn("identity changed", result["error"])

    def test_deadline_exceeded_by_last_response_cannot_claim_complete(self):
        clock = [0.0]
        def transport(route, payload=None, timeout=20):
            result = self.transport(route, payload, timeout)
            if route == "/api/chat" and sum(c[0] == route for c in self.calls) == 20:
                clock[0] = feedback.TOTAL_SECONDS + 1
            return result
        with patch.object(feedback.time, "perf_counter", side_effect=lambda: clock[0]):
            result = self.run_batch(transport)
        self.assertEqual(result["generation_requests_started"], 20)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["overall"]["strict_em_accuracy_pct"])
        self.assertIn("budget exceeded", result["error"])


if __name__ == "__main__":
    unittest.main()
