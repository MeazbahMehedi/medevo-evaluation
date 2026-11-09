import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check_primary import CONCISE_INSTRUCTION, MODEL, SELECTED_ID, prepare_case, run_check
from src.runtime import write_json
from src.training_collection import read_json, tree_hashes


class PrimaryCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.collection = self.root / "collection"
        self.collection.mkdir()
        write_json(self.collection / "verification.json", {"reference": "secret_reference"})
        write_json(self.collection / "initial_request.json", {"question": "Question"})
        self.payload = {"model": MODEL, "messages": [{"role": "user", "content": "Question\nOriginal instruction", "images": ["aW1hZ2U="]}],
                        "stream": False, "keep_alive": "2m",
                        "options": {"temperature": 0, "seed": 7, "num_predict": 128, "num_ctx": 4096, "num_thread": 4, "num_gpu": 0}}
        self.calls = []
        self.reply = {"model": MODEL, "done": True, "done_reason": "stop", "message": {"content": "\\boxed{unknown}"},
                      "eval_count": 5, "total_duration": 1000000000, "load_duration": 100000000, "eval_duration": 500000000}

    def transport(self, route, payload=None, timeout=20):
        self.calls.append((route, payload, timeout))
        if route == "/api/version": return {"version": "controlled"}
        if route == "/api/tags": return {"models": [{"name": MODEL, "digest": "fixture-digest"}]}
        if route == "/api/show": return {"capabilities": ["completion", "vision"]}
        if route == "/api/ps": return {"models": [{"name": MODEL, "digest": "fixture-digest", "size": 6 * 1024**3, "size_vram": 0}]}
        if route == "/api/chat": return self.reply
        raise AssertionError("Unexpected endpoint")

    def run_probe(self, transport=None, **options):
        with patch("check_primary.prepare_case", return_value=(self.collection, self.payload)), contextlib.redirect_stdout(io.StringIO()):
            return run_check(self.root, self.collection, transport=transport or self.transport, available_ram=lambda: 16, **options)

    def test_one_image_generation_excludes_reference_and_keeps_sources_unchanged(self):
        before = tree_hashes(self.collection)
        result = self.run_probe()
        chats = [c for c in self.calls if c[0] == "/api/chat"]
        self.assertEqual(len(chats), 1)
        self.assertNotIn("secret_reference", str(chats[0][1]))
        self.assertEqual(chats[0][2], 600)
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["model_inference_completed"])
        self.assertTrue(result["cpu_only_reported"])
        self.assertIsNone(result["peak_ollama_ram_gib"])
        self.assertEqual(result["training_examples_admitted"], 0)
        self.assertEqual(tree_hashes(self.collection), before)

    def test_truncation_is_preserved_without_a_longer_generation_retry(self):
        self.reply.update(done_reason="length", eval_count=128)
        result = self.run_probe()
        self.assertEqual(result["outcome"], "truncated_output")
        self.assertTrue(result["model_inference_completed"])
        self.assertFalse(result["finished_answer_available"])
        self.assertEqual(len([c for c in self.calls if c[0] == "/api/chat"]), 1)
        self.assertEqual(read_json(self.root / "outputs/larger_primary_v1/response.json"), self.reply)

    def test_generation_failure_is_not_retried_or_reported_as_a_fit(self):
        def failing(route, payload=None, timeout=20):
            if route == "/api/chat":
                self.calls.append((route, payload, timeout))
                raise TimeoutError("controlled timeout")
            return self.transport(route, payload, timeout)
        result = self.run_probe(failing)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("controlled timeout", result["error"])
        self.assertEqual(len([c for c in self.calls if c[0] == "/api/chat"]), 1)

    def test_repeat_and_low_memory_stop_before_model_generation(self):
        with patch("check_primary.prepare_case", return_value=(self.collection, self.payload)):
            with self.assertRaisesRegex(RuntimeError, "8 GiB"):
                run_check(self.root, self.collection, transport=self.transport, available_ram=lambda: 4)
        self.assertFalse(self.calls)
        self.run_probe()
        count = len(self.calls)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_probe()
        self.assertEqual(len(self.calls), count)

    def test_nontraining_source_is_rejected_before_reading_images(self):
        write_json(self.collection / "run.json", {"collection_id": "training-collection-12-v1", "role": "test", "selected_ids": [SELECTED_ID] * 12})
        write_json(self.collection / "summary.json", {"status": "complete"})
        with self.assertRaisesRegex(ValueError, "training collection"):
            prepare_case(self.collection)

    def truncated_baseline(self):
        self.reply.update(done_reason="length", eval_count=128)
        baseline = Path(self.run_probe()["output_folder"])
        self.calls.clear()
        self.reply = {**self.reply, "done_reason": "stop", "eval_count": 250,
                      "message": {"content": "A completed controlled answer."}}
        return baseline

    def test_extended_run_changes_only_budget_and_preserves_baseline(self):
        baseline = self.truncated_baseline()
        before = tree_hashes(baseline)
        original = read_json(baseline / "request.json")
        result = self.run_probe(max_new_tokens=512, baseline_dir=baseline)
        chats = [c for c in self.calls if c[0] == "/api/chat"]
        self.assertEqual(len(chats), 1)
        trial = chats[0][1]
        self.assertEqual(trial["options"]["num_predict"], 512)
        trial_without_change = {**trial, "options": {**trial["options"], "num_predict": 128}}
        self.assertEqual(trial_without_change, original)
        self.assertEqual(tree_hashes(baseline), before)
        self.assertTrue(result["baseline_preserved"])
        self.assertEqual(result["training_examples_admitted"], 0)
        copied = self.root / "outputs/larger_primary_512_v1/baseline_128/response.json"
        self.assertEqual(read_json(copied), read_json(baseline / "response.json"))

    def test_changed_baseline_prompt_is_rejected_before_any_request(self):
        baseline = self.truncated_baseline()
        request = read_json(baseline / "request.json")
        request["messages"][0]["content"] += "Changed prompt."
        write_json(baseline / "request.json", request)
        with self.assertRaisesRegex(ValueError, "same-case, same-request"):
            self.run_probe(max_new_tokens=512, baseline_dir=baseline)
        self.assertFalse(self.calls)

    def test_extended_run_rejects_changed_model_or_server_before_generation(self):
        baseline = self.truncated_baseline()
        for changed in ("/api/tags", "/api/version"):
            def changed_transport(route, payload=None, timeout=20):
                if route == changed:
                    return {"version": "different"} if changed == "/api/version" else {"models": [{"name": MODEL, "digest": "different"}]}
                return self.transport(route, payload, timeout)
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, "digest or Ollama version"):
                self.run_probe(transport=changed_transport, max_new_tokens=512, baseline_dir=baseline)
            self.assertFalse([c for c in self.calls if c[0] == "/api/chat"])

    def test_extension_requires_explicit_baseline_and_approved_budget(self):
        for options in ({"max_new_tokens": 512}, {"max_new_tokens": 1024}, {"baseline_dir": self.collection}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "512-token experiment"):
                self.run_probe(**options)
        self.assertFalse(self.calls)

    def test_extended_truncation_is_recorded_without_more_attempts(self):
        baseline = self.truncated_baseline()
        self.reply.update(done_reason="length", eval_count=512)
        result = self.run_probe(max_new_tokens=512, baseline_dir=baseline)
        self.assertEqual(result["outcome"], "truncated_output")
        self.assertEqual(result["generated_tokens"], 512)
        self.assertTrue(result["baseline_preserved"])
        self.assertEqual(len([c for c in self.calls if c[0] == "/api/chat"]), 1)

    def truncated_extended_baseline(self):
        original = self.truncated_baseline()
        self.reply.update(done_reason="length", eval_count=512)
        extended = Path(self.run_probe(max_new_tokens=512, baseline_dir=original)["output_folder"])
        self.calls.clear()
        self.reply = {**self.reply, "done_reason": "stop", "eval_count": 60,
                      "message": {"content": "1. Controlled observation.\\n\\boxed{unknown}"}}
        return original, extended

    def test_concise_prompt_changes_only_instruction_and_preserves_both_runs(self):
        original, extended = self.truncated_extended_baseline()
        hashes = tree_hashes(original), tree_hashes(extended)
        baseline_request = read_json(extended / "request.json")
        result = self.run_probe(max_new_tokens=512, baseline_dir=extended, concise_prompt=True)
        chats = [c for c in self.calls if c[0] == "/api/chat"]
        self.assertEqual(len(chats), 1)
        trial = chats[0][1]
        self.assertEqual(trial["messages"][0]["content"], "Question\n" + CONCISE_INSTRUCTION)
        restored = {**trial, "messages": [{**trial["messages"][0], "content": baseline_request["messages"][0]["content"]}]}
        self.assertEqual(restored, baseline_request)
        self.assertEqual((tree_hashes(original), tree_hashes(extended)), hashes)
        self.assertEqual(result["changed_request_field"], "messages[0].content")
        self.assertEqual(result["training_examples_admitted"], 0)
        self.assertTrue(result["baseline_preserved"])
        self.assertNotIn("secret_reference", str(trial))

    def test_concise_prompt_rejects_128_token_baseline_before_generation(self):
        original = self.truncated_baseline()
        with self.assertRaisesRegex(ValueError, "512-token baseline"):
            self.run_probe(max_new_tokens=512, baseline_dir=original, concise_prompt=True)
        self.assertFalse(self.calls)

    def test_concise_prompt_rejects_changed_baseline_options_before_generation(self):
        _, extended = self.truncated_extended_baseline()
        payload = read_json(extended / "request.json")
        payload["options"]["temperature"] = .7
        write_json(extended / "request.json", payload)
        with self.assertRaisesRegex(ValueError, "same-case, same-request"):
            self.run_probe(max_new_tokens=512, baseline_dir=extended, concise_prompt=True)
        self.assertFalse(self.calls)

    def test_concise_prompt_truncation_is_not_retried(self):
        _, extended = self.truncated_extended_baseline()
        self.reply.update(done_reason="length", eval_count=512)
        result = self.run_probe(max_new_tokens=512, baseline_dir=extended, concise_prompt=True)
        self.assertEqual(result["outcome"], "truncated_output")
        self.assertEqual(len([c for c in self.calls if c[0] == "/api/chat"]), 1)
        self.assertTrue(result["baseline_preserved"])

    def test_concise_prompt_rejects_changed_server_identity_before_generation(self):
        _, extended = self.truncated_extended_baseline()
        def changed_version(route, payload=None, timeout=20):
            return {"version": "different"} if route == "/api/version" else self.transport(route, payload, timeout)
        with self.assertRaisesRegex(ValueError, "digest or Ollama version"):
            self.run_probe(transport=changed_version, max_new_tokens=512, baseline_dir=extended, concise_prompt=True)
        self.assertFalse([c for c in self.calls if c[0] == "/api/chat"])


if __name__ == "__main__":
    unittest.main()
