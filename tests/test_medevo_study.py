"""Checks for the frozen-protocol study code using stand-in models (no Ollama, no Transformers)."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import compile_results
import freeze_protocol
import run_test_evaluation as rte
from medevo_study import collection, core, ollama_backend


def _benchmark_data_available():
    try:
        core.load_records()
        return True
    except Exception:
        return False


DATA_AVAILABLE = _benchmark_data_available()
REQUIRES_DATA = unittest.skipUnless(DATA_AVAILABLE, "requires prepared benchmark data in data/benchmarks")


def boxed(answer):
    return ("The image is an axial abdominal CT with contrast. The structure in question lies in the upper "
            f"abdomen and its density matches solid organ tissue. Therefore the answer is \\boxed{{{answer}}}")


@REQUIRES_DATA
class SelectionTests(unittest.TestCase):
    def setUp(self):
        _, self.records = core.load_records()

    def test_stratified_sample_is_proportional_and_deterministic(self):
        first = core.stratified_sample(self.records, "test", "vqa_rad", 200, 7, "x")
        again = core.stratified_sample(self.records, "test", "vqa_rad", 200, 7, "x")
        self.assertEqual([r["id"] for r in first], [r["id"] for r in again])
        self.assertEqual(len(first), 200)
        self.assertTrue(all(r["role"] == "test" for r in first))
        closed = sum(r["answer_type"] == "closed" for r in first)
        pool = [r for r in self.records if r["role"] == "test" and r["dataset"] == "vqa_rad"]
        expected = 200 * sum(r["answer_type"] == "closed" for r in pool) / len(pool)
        self.assertLessEqual(abs(closed - expected), 1)

    def test_freeze_protocol_uses_feedback_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "feedback").mkdir()
            core.write_json(tmp / "feedback/summary.json", {"decision": {"outcome": "adopt_1024", "missing_boxes": 5}})
            (tmp / "qwen35_feedback_119_plan_v1").mkdir()
            core.write_json(tmp / "qwen35_feedback_119_plan_v1/plan.json", {"model_digest": "abc", "ollama_version": {"v": 1}})
            with mock.patch.object(freeze_protocol, "FEEDBACK_RUN", tmp / "feedback"), \
                    mock.patch.object(freeze_protocol, "PROTOCOL_DIR", tmp / "frozen"), \
                    mock.patch.object(freeze_protocol, "OUTPUTS", tmp):
                result = freeze_protocol.freeze(200, None, None)
                self.assertEqual(result["output_token_budget"], 1024)
                protocol = core.read_json(tmp / "frozen/protocol.json")
                self.assertEqual(len(protocol["test_ids"]), 400)
                self.assertEqual(len(set(protocol["test_ids"])), 400)
                self.assertEqual(protocol["baseline_model"]["options"]["num_predict"], 1024)
                with self.assertRaises(ValueError):
                    freeze_protocol.freeze(200, None, None)

    def test_freeze_refuses_without_decision(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(freeze_protocol, "FEEDBACK_RUN", Path(tmp) / "missing"):
                with self.assertRaises(ValueError):
                    freeze_protocol.decide_budget(None, None)
                self.assertEqual(freeze_protocol.decide_budget(512, "documented reason")[0], 512)


class ScoringTests(unittest.TestCase):
    def test_primary_metric_unwraps_text(self):
        self.assertTrue(core.primary_correct(r"so \boxed{\text{Nodule}}", "nodule"))
        self.assertFalse(core.primary_correct("no box here: nodule", "nodule"))

    def test_binary_mapping_is_secondary_only(self):
        self.assertFalse(core.primary_correct(r"Thus, yes. \boxed{1}", "Yes"))
        self.assertTrue(core.score_binary_mapped(r"Thus, yes. \boxed{1}", "Yes")["normalized_correct"])
        self.assertTrue(core.score_binary_mapped(r"\boxed{0}", "no")["normalized_correct"])
        self.assertFalse(core.score_binary_mapped(r"\boxed{1}", "no")["normalized_correct"])
        self.assertFalse(core.score_binary_mapped(r"\boxed{1}", "2")["normalized_correct"])
        self.assertTrue(core.score_binary_mapped(r"\boxed{2}", "2")["normalized_correct"])

    def test_bootstrap_and_mcnemar(self):
        ci = core.bootstrap_ci([1] * 50 + [0] * 50)
        self.assertLess(ci[0], 50)
        self.assertGreater(ci[1], 50)
        self.assertEqual(core.mcnemar_exact(0, 0), 1.0)
        self.assertLess(core.mcnemar_exact(0, 12), 0.01)
        paired = core.paired_bootstrap({"a": 0, "b": 1, "c": 0}, {"a": 1, "b": 1, "c": 1})
        self.assertAlmostEqual(paired["difference_points"], 200 / 3)

    def test_hint_leak_guard(self):
        self.assertTrue(core.hint_leaks_answer("Look at the suprasellar cistern", "suprasellar cistern"))
        self.assertTrue(core.hint_leaks_answer("The answer is yes.", "yes"))
        self.assertFalse(core.hint_leaks_answer("Check for no-contrast artifacts on both sides", "yes"))
        self.assertFalse(core.hint_leaks_answer("Compare intensity across the midline", "right"))

    def test_audit_validation(self):
        steps = core.reasoning_steps("First step. Second step. Third step.")
        good = {"action": "repair", "first_error_step": 2, "error_type": "V", "evidence": "x", "hint": "Recheck it."}
        self.assertEqual(ollama_backend.validate_audit(good, steps)[0], "repair")
        self.assertEqual(ollama_backend.validate_audit({**good, "first_error_step": 0}, steps)[0], "invalid")
        self.assertEqual(ollama_backend.validate_audit({**good, "action": "skip"}, steps)[0], "skip")
        self.assertEqual(ollama_backend.validate_audit(None, steps)[0], "invalid")

    def test_repetition_detection(self):
        looping = " ".join(["the lesion is bright on this sequence"] * 20)
        self.assertIn("repetitive", core.integrity_problems(looping, False))
        self.assertEqual(core.integrity_problems(boxed("yes"), False), [])


class FakePrimary:
    def __init__(self, rollouts, continuations, nll=(0.6, 0.5)):
        self.rollouts, self.continuations, self.nll_values = list(rollouts), list(continuations), list(nll)
        self.calls = []

    def generate(self, record, prefill, sample, seed):
        self.calls.append(prefill)
        text = self.continuations.pop(0) if prefill else self.rollouts.pop(0)
        return {"raw_response": text, "hit_token_limit": False, "generated_tokens": 50,
                "generation_seconds": 1.0, "generated_token_ids": [1, 2, 3]}

    def nll(self, record, prefill, ids):
        return self.nll_values[0] if "recheck" not in prefill else self.nll_values[1]


class FakeAuditor:
    def __init__(self, value):
        self.value = value

    def audit(self, record, text, steps):
        decision, reason = ollama_backend.validate_audit(self.value, steps)
        return decision, reason, self.value, 2.0


RECORD = {"id": "slake:train:1", "dataset": "slake", "answer_type": "open", "question": "Which organ?", "answer": "Liver"}
REPAIR = {"action": "repair", "first_error_step": 2, "error_type": "V", "evidence": "misread", "hint": "Re-examine the upper left region."}


class CollectionTests(unittest.TestCase):
    def test_correct_rollout_is_admitted_without_audit(self):
        primary = FakePrimary([boxed("Spleen"), boxed("Liver"), boxed("Liver"), boxed("Kidney")], [])
        case = collection.process_question(RECORD, primary, FakeAuditor(REPAIR))
        self.assertEqual(case["outcome"], "correct_rollout_admitted")
        self.assertEqual(len(case["admitted"]), 1)
        self.assertNotIn("audit", case)

    def test_failure_is_repaired_from_prefix_and_hint_removed_from_target(self):
        failed = "The image is an abdominal CT. The organ on the left is the spleen. So \\boxed{Spleen}"
        primary = FakePrimary([failed] * 4, [boxed("Liver")])
        case = collection.process_question(RECORD, primary, FakeAuditor(REPAIR))
        self.assertEqual(case["outcome"], "repair_admitted")
        target = case["admitted"][0]["target_text"]
        self.assertTrue(target.startswith("The image is an abdominal CT."))
        self.assertNotIn("recheck", target.lower())
        self.assertIn("Re-examine", primary.calls[-1])

    def test_leaking_hint_and_unlearnable_repair_are_rejected(self):
        failed = "The image is an abdominal CT. The organ is the spleen. So \\boxed{Spleen}"
        leak = {**REPAIR, "hint": "Look again, it is the liver."}
        case = collection.process_question(RECORD, FakePrimary([failed] * 4, []), FakeAuditor(leak))
        self.assertEqual(case["outcome"], "hint_rejected_leak")
        primary = FakePrimary([failed] * 4, [boxed("Liver")] * 3, nll=(3.0, 0.5))
        case = collection.process_question(RECORD, primary, FakeAuditor(REPAIR))
        self.assertEqual(case["outcome"], "repair_attempts_rejected")
        self.assertIn("not_learnable_nll", case["repair"]["attempts"][0]["problems"])

    def test_skip_audit_keeps_no_example(self):
        failed = "The image is an abdominal CT. The organ is the spleen. So \\boxed{Spleen}"
        skip = {"action": "skip", "first_error_step": 0, "error_type": "none", "evidence": "", "hint": ""}
        case = collection.process_question(RECORD, FakePrimary([failed] * 4, []), FakeAuditor(skip))
        self.assertEqual(case["outcome"], "audit_skip")
        self.assertEqual(case["admitted"], [])


@REQUIRES_DATA
class EvaluationTests(unittest.TestCase):
    def test_resumable_evaluation_and_compilation(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _, records = core.load_records()
            ids = [r["id"] for r in core.stratified_sample(records, "test", "vqa_rad", 3, 7, "t")]
            ids += [r["id"] for r in core.stratified_sample(records, "test", "slake", 3, 7, "t")]
            by_id = {r["id"]: r for r in records}
            protocol_dir = tmp / "frozen"
            protocol_dir.mkdir()
            core.write_json(protocol_dir / "protocol.json", {
                "dataset_file_sha256": core.data_hashes(), "test_ids": ids, "output_token_budget": 1024,
                "prompt": core.PAPER_PROMPT, "test_selection": {"per_dataset": 3, "seed": 7, "counts": {}},
                "primary_metric": {"definition": "x"}, "uncertainty": "x"})
            calls = []

            def fake_backend(system, protocol, adapter_dir):
                def run(record):
                    calls.append(record["id"])
                    if len(calls) == 2 and system == "primary":
                        raise KeyboardInterrupt
                    answer = record["answer"] if system == "adapted" or len(calls) % 2 else "wrong"
                    return {"id": record["id"], "status": "generated", "raw_response": boxed(answer),
                            "generation_seconds": 1.0, "generated_tokens": 40, "hit_token_limit": False}
                return run, {"fake": system}

            with mock.patch.object(rte, "OUTPUTS", tmp), mock.patch.object(rte, "PROTOCOL_DIR", protocol_dir), \
                    mock.patch.object(rte, "make_backend", fake_backend):
                partial = rte.evaluate("primary", 0, None, 0)
                self.assertEqual(partial["status"], "incomplete")
                complete = rte.evaluate("primary", 0, None, 0)
                self.assertEqual(complete["status"], "complete")
                adapted = rte.evaluate("adapted", 0, None, 0)
                overall = adapted["metrics"][adapted["primary_metric"]]["overall"]
                self.assertEqual(overall["accuracy_pct"], 100.0)
                self.assertEqual(len(core.read_jsonl(tmp / "test_eval_qwen35_2b_base_v1/predictions.jsonl")), 6)
            with mock.patch.object(compile_results, "OUTPUTS", tmp), \
                    mock.patch.object(compile_results, "PROTOCOL_DIR", protocol_dir), \
                    mock.patch.object(compile_results, "OUT", tmp / "results"):
                result = compile_results.compile_all()
                self.assertIn("adapted_vs_primary", result["paired"])
                text = (tmp / "results/technical_report_draft.md").read_text(encoding="utf-8")
                self.assertIn("Effect of adaptation", text)
                self.assertNotIn("{", text.split("## 1.")[0])


if __name__ == "__main__":
    unittest.main()


class FakePrimaryV2:
    """Losses chosen so the paper's learnability gates pass unless overridden."""

    def __init__(self, continuations, with_input=0.5, without_input=1.0, wrong=0.4):
        self.continuations = list(continuations)
        self.losses = {"with": with_input, "without": without_input, "wrong": wrong}
        self.prefills = []

    def generate(self, record, prefill, sample, seed):
        self.prefills.append(prefill)
        return {"raw_response": self.continuations.pop(0), "hit_token_limit": False, "generated_tokens": 40,
                "generation_seconds": 1.0, "generated_token_ids": [7, 8, 9]}

    def ids(self, text):
        return [1, 2]

    def nll(self, record, prefill, ids, include_input):
        if ids == [1, 2]:
            return self.losses["wrong"]
        return self.losses["with"] if include_input else self.losses["without"]


class FakeAuditorV2:
    def __init__(self, value):
        self.value = value
        self.info = {}

    def audit(self, record, steps):
        decision, reason = ollama_backend.validate_audit_v2(self.value, steps)
        return decision, reason, self.value, 1.0


REPAIR_V2_AUDIT = {"action": "repair", "first_error_step": 2, "error_type": "V",
                   "key_error": "Misread the organ.", "hint": "Re-examine the upper left region."}
FAILED = "The image is an abdominal CT. The organ on the left is the spleen. So \\boxed{Spleen}"


class RepairV2Tests(unittest.TestCase):
    def test_admitted_target_keeps_prefix_and_cue(self):
        primary = FakePrimaryV2([boxed("Liver")])
        result = collection.repair_failed_rollout(RECORD, FAILED, primary, FakeAuditorV2(REPAIR_V2_AUDIT))
        self.assertEqual(result["outcome"], "repair_admitted")
        target = result["admitted"][0]["target_text"]
        self.assertTrue(target.startswith("The image is an abdominal CT.\nLet me recheck the evidence. "
                                          "Re-examine the upper left region.\n"))
        self.assertTrue(target.endswith("\\boxed{Liver}"))
        attempt = result["repair"]["attempts"][0]
        self.assertAlmostEqual(attempt["repair_ratio"], 0.5)
        self.assertAlmostEqual(attempt["repair_gap"], 0.1)

    def test_learnability_gates_follow_the_paper(self):
        ratio_fail = FakePrimaryV2([boxed("Liver")] * 5, with_input=0.95, without_input=1.0, wrong=0.9)
        result = collection.repair_failed_rollout(RECORD, FAILED, ratio_fail, FakeAuditorV2(REPAIR_V2_AUDIT))
        self.assertEqual(result["outcome"], "repair_attempts_rejected")
        self.assertEqual(len(result["repair"]["attempts"]), 5)
        self.assertIn("repair_ratio_out_of_range", result["repair"]["attempts"][0]["problems"])
        gap_fail = FakePrimaryV2([boxed("Liver")] * 5, with_input=0.5, without_input=1.0, wrong=0.6)
        result = collection.repair_failed_rollout(RECORD, FAILED, gap_fail, FakeAuditorV2(REPAIR_V2_AUDIT))
        self.assertIn("repair_gap_out_of_range", result["repair"]["attempts"][0]["problems"])

    def test_format_only_skip_and_invalid_audits(self):
        skip = {"action": "skip", "first_error_step": 0, "error_type": "none", "key_error": "", "hint": ""}
        result = collection.repair_failed_rollout(RECORD, FAILED, FakePrimaryV2([]), FakeAuditorV2(skip))
        self.assertEqual(result["outcome"], "audit_skip")
        bad = {**REPAIR_V2_AUDIT, "first_error_step": 9}
        result = collection.repair_failed_rollout(RECORD, FAILED, FakePrimaryV2([]), FakeAuditorV2(bad))
        self.assertEqual(result["outcome"], "audit_invalid")

    def test_audit_v2_prompt_states_failure_and_hides_reference(self):
        self.assertIn("FAILED", ollama_backend.AUDIT_INSTRUCTION_V2)
        self.assertIn("skip' only when", ollama_backend.AUDIT_INSTRUCTION_V2)
        self.assertNotIn("unresolved", ollama_backend.AUDIT_SCHEMA_V2["properties"]["action"]["enum"])


@REQUIRES_DATA
class ReauditRunTests(unittest.TestCase):
    def test_reaudit_reuses_v1_failures_and_merges_bank(self):
        import reaudit_failures
        _, records = core.load_records()
        train = [r for r in records if r["role"] == "train"][:3]
        cases = [
            {"id": train[0]["id"], "dataset": train[0]["dataset"], "answer_type": train[0]["answer_type"],
             "outcome": "correct_rollout_admitted", "rollouts": [],
             "admitted": [{"kind": "correct_rollout", "rollout": 0, "target_text": boxed(train[0]["answer"])}]},
            {"id": train[1]["id"], "dataset": train[1]["dataset"], "answer_type": train[1]["answer_type"],
             "outcome": "audit_skip", "audit": {"rollout": 0}, "rollouts": [{"raw_response": FAILED}], "admitted": []},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "v1").mkdir()
            core.write_jsonl(tmp / "v1/cases.jsonl", cases)
            protocol_dir = tmp / "frozen"
            protocol_dir.mkdir()
            core.write_json(protocol_dir / "protocol.json", {"dataset_file_sha256": core.data_hashes(),
                                                             "output_token_budget": 1024, "prompt": core.PAPER_PROMPT,
                                                             "baseline_model": {"model_digest": "x"}})
            answer = train[1]["answer"]
            primary = FakePrimaryV2([boxed(answer)])
            with mock.patch.object(reaudit_failures, "SOURCE", tmp / "v1"), \
                    mock.patch.object(reaudit_failures, "FOLDER", tmp / "v2"), \
                    mock.patch.object(reaudit_failures, "PROTOCOL_DIR", protocol_dir), \
                    mock.patch.object(reaudit_failures, "PrimaryModel", lambda protocol: primary), \
                    mock.patch.object(reaudit_failures, "Auditor",
                                      lambda protocol, think: FakeAuditorV2(REPAIR_V2_AUDIT)):
                primary.info = {}
                summary = reaudit_failures.run(0, False)
            self.assertEqual(summary["targets"], 1)
            self.assertEqual(summary["admitted_repairs"], 1)
            self.assertEqual(summary["bank_size"], 2)
            kinds = sorted(item["kind"] for item in core.read_jsonl(tmp / "v2/training_bank.jsonl"))
            self.assertEqual(kinds, ["correct_rollout", "repair"])
