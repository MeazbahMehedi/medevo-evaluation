import base64
import contextlib
import hashlib
import io
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image

from check_primary import CONCISE_CHECK_ID, CONCISE_INSTRUCTION, MODEL, ROOT, SELECTED_ID
from evaluate_primary import OPTIONS, frozen_profile, run_evaluation
from src.runtime import write_json
from src.training_collection import read_json, tree_hashes
import test_prompt_experiment as fixtures


class PrimaryFeedbackTests(unittest.TestCase):
    def setUp(self):
        f = fixtures.PromptExperimentTests(methodName="test_pairing_controls_snapshots_and_original_files_are_preserved")
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.f = f
        self.profile = f.root / "profile"
        self.profile.mkdir()
        image = Image.new("RGB", (8, 8), "red")
        buffer = io.BytesIO(); image.save(buffer, format="PNG")
        raw = buffer.getvalue()
        check = patch("evaluate_primary.AUDIT_IMAGE_SHA256", hashlib.sha256(raw).hexdigest())
        check.start(); self.addCleanup(check.stop)
        question = read_json(ROOT / "configs/continuation_case/saved_input.json")["question"]
        self.request = {"model": MODEL, "messages": [{"role": "user", "content": question + "\n" + CONCISE_INSTRUCTION,
                        "images": [base64.b64encode(raw).decode()]}], "stream": False, "keep_alive": "2m", "options": OPTIONS}
        run = {"check_id": CONCISE_CHECK_ID, "selected_id": SELECTED_ID, "role": "train", "max_new_tokens": 512,
               "prompt_variant": "concise-three-step-v1", "model_digest": "fixture-digest", "ollama_version": "controlled"}
        summary = {**run, "status": "complete", "generation_requests_started": 1, "raw_response": "A completed controlled answer."}
        reply = {"model": MODEL, "done": True, "done_reason": "stop", "message": {"content": summary["raw_response"]}}
        for name, value in {"run.json": run, "summary.json": summary, "request.json": self.request,
                            "response.json": reply, "model_details.json": {"capabilities": ["vision"]}}.items():
            write_json(self.profile / name, value)
        self.calls = []
        self.failure_at = None

    def transport(self, route, payload=None, timeout=20):
        self.calls.append((route, payload, timeout))
        if route == "/api/version": return {"version": "controlled"}
        if route == "/api/tags": return {"models": [{"name": MODEL, "digest": "fixture-digest"}]}
        if route == "/api/ps": return {"models": [{"name": MODEL, "digest": "fixture-digest", "size_vram": 0}]}
        if route == "/api/chat":
            count = len([c for c in self.calls if c[0] == route])
            if self.failure_at == count:
                raise TimeoutError("Controlled service timeout")
            return {"model": MODEL, "done": True, "done_reason": "stop", "message": {"content": "\\boxed{kidney}"}, "eval_count": 5}
        raise AssertionError("Unexpected endpoint")

    def run_batch(self, transport=None):
        f = self.f
        with contextlib.redirect_stdout(io.StringIO()):
            return run_evaluation(f.root, f.data, self.profile, f.baselines, transport or self.transport, available_ram=lambda: 16)

    def test_ten_fixed_feedback_requests_exclude_references_and_preserve_all_sources(self):
        before = tree_hashes(self.profile), tree_hashes(self.f.data), {d: tree_hashes(p) for d, p in self.f.baselines.items()}
        report = self.run_batch()
        chats = [c for c in self.calls if c[0] == "/api/chat"]
        self.assertEqual(len(chats), 10)
        for _, payload, _ in chats:
            self.assertEqual(payload["options"], OPTIONS)
            self.assertEqual(payload["messages"][0]["content"], "Which organ is shown?\n" + CONCISE_INSTRUCTION)
            self.assertNotIn("kidney", str(payload))
            self.assertEqual(set(payload["messages"][0]), {"role", "content", "images"})
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["overall"]["strict_em_accuracy_pct"], 100)
        for dataset in ("vqa_rad", "slake"):
            self.assertEqual(report["datasets"][dataset]["summary"]["selected_count"], 5)
        self.assertEqual((tree_hashes(self.profile), tree_hashes(self.f.data), {d: tree_hashes(p) for d, p in self.f.baselines.items()}), before)
        self.assertEqual(report["training_examples_admitted"], 0)
        self.assertIsNone(report["peak_ollama_ram_gib"])

    def test_transport_failure_stops_without_retry_or_full_accuracy_claim(self):
        self.failure_at = 3
        report = self.run_batch()
        self.assertEqual(len([c for c in self.calls if c[0] == "/api/chat"]), 3)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["overall"]["generated_count"], 2)
        self.assertEqual(report["overall"]["error_count"], 1)
        self.assertEqual(report["overall"]["missing_count"], 7)
        self.assertIsNone(report["overall"]["strict_em_accuracy_pct"])
        self.assertIn("Controlled service timeout", report["error"])

    def test_foreign_or_reordered_selection_is_rejected_before_network(self):
        path = self.f.baselines["vqa_rad"] / "run.json"
        value = read_json(path)
        value["signature"]["selected_ids"].reverse()
        write_json(path, value)
        with self.assertRaisesRegex(ValueError, "fixed five-question selection"):
            self.run_batch()
        self.assertFalse(self.calls)

    def test_test_role_baseline_is_rejected_before_network(self):
        path = self.f.baselines["slake"] / "run.json"
        value = read_json(path); value["signature"]["role"] = "test"; write_json(path, value)
        with self.assertRaisesRegex(ValueError, "feedback roles"):
            self.run_batch()
        self.assertFalse(self.calls)

    def test_changed_profile_controls_are_rejected_before_network(self):
        request = read_json(self.profile / "request.json")
        request["options"]["num_predict"] = 1024
        write_json(self.profile / "request.json", request)
        with self.assertRaisesRegex(ValueError, "generation controls"):
            self.run_batch()
        self.assertFalse(self.calls)

    def test_changed_model_digest_prevents_generation(self):
        def changed(route, payload=None, timeout=20):
            if route == "/api/tags": return {"models": [{"name": MODEL, "digest": "different"}]}
            return self.transport(route, payload, timeout)
        with self.assertRaisesRegex(ValueError, "model digest or server version"):
            self.run_batch(changed)
        self.assertFalse([c for c in self.calls if c[0] == "/api/chat"])

    def test_missing_boxes_remain_wrong_without_a_plain_answer_fallback(self):
        def unboxed(route, payload=None, timeout=20):
            reply = self.transport(route, payload, timeout)
            if route == "/api/chat": reply["message"]["content"] = "kidney"
            return reply
        report = self.run_batch(unboxed)
        self.assertEqual(report["overall"]["parse_failure_count"], 10)
        self.assertEqual(report["overall"]["strict_em_accuracy_pct"], 0)

    def test_completed_output_cannot_be_overwritten_by_a_repeat(self):
        self.run_batch()
        count = len(self.calls)
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.run_batch()
        self.assertEqual(len(self.calls), count)

    def test_postrun_identity_failure_masks_accuracy_without_discarding_predictions(self):
        def mismatch(route, payload=None, timeout=20):
            reply = self.transport(route, payload, timeout)
            if route == "/api/ps": reply["models"][0]["digest"] = "different"
            return reply
        report = self.run_batch(mismatch)
        self.assertEqual(report["status"], "incomplete")
        self.assertEqual(report["overall"]["generated_count"], 10)
        self.assertIsNone(report["overall"]["strict_em_accuracy_pct"])
        for value in report["datasets"].values():
            self.assertIsNone(value["summary"]["strict_em_accuracy_pct"])
        self.assertTrue((Path(report["output_folder"]) / "cases/10/response.json").exists())


if __name__ == "__main__":
    unittest.main()
