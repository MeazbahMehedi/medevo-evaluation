import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.answer_diagnostic import diagnose_answers, diagnostic_summary, normalize_plain, score_predictions
from src.benchmarks import decoded_sha256, file_sha256, write_jsonl
from src.evaluation import create_run, read_protocol, score_run
from src.runtime import write_json


ROOT = Path(__file__).resolve().parents[1]


def prediction(identity, response):
    return {"id": identity, "status": "generated", "raw_response": response,
            "generated_tokens": 3, "generation_seconds": .5, "hit_token_limit": False}


class PlainAnswerTests(unittest.TestCase):
    def test_fixed_normalization(self):
        self.assertEqual(normalize_plain("  Posterior \n brain. "), "posterior brain")
        self.assertEqual(normalize_plain("YES!?"), "yes")
        self.assertEqual(normalize_plain("1.5 cm."), "1.5 cm")

    def test_complete_response_has_no_extraction_or_semantic_mapping(self):
        cases = [("CT", "ct", True), ("X-ray.", "X-Ray", True), ("Yes.", "yes", True),
                 ("CT scan", "CT", False), ("The image is a CT scan.", "CT", False),
                 ("two", "2", False), ("no mass", "mass", False),
                 ("1.5 cm", "15 cm", False), (r"\boxed{CT}", "CT", False),
                 ("Answer: CT", "CT", False), ("not yes", "yes", False), ("", "yes", False)]
        for response, reference, expected in cases:
            with self.subTest(response=response):
                selected = [{"id": "a", "dataset": "slake", "question": "Q", "answer_type": "open", "answer": reference}]
                row = score_predictions(selected, [prediction("a", response)])[0]
                self.assertEqual(row["plain_whole_answer_correct"], expected)

    def test_boxed_and_plain_methods_remain_independent(self):
        selected = [{"id": identity, "dataset": "slake", "question": "Q", "answer_type": "open", "answer": "MRI"}
                    for identity in ("boxed", "plain")]
        rows = score_predictions(selected, [prediction("boxed", r"\boxed{MRI}"), prediction("plain", "MRI")])
        self.assertTrue(rows[0]["strict_correct"])
        self.assertFalse(rows[0]["plain_whole_answer_correct"])
        self.assertFalse(rows[1]["strict_correct"])
        self.assertTrue(rows[1]["plain_whole_answer_correct"])

    def test_missing_or_failed_generation_cannot_produce_full_accuracy(self):
        selected = [{"id": identity, "dataset": "slake", "question": "Q", "answer_type": "open", "answer": "MRI"}
                    for identity in ("a", "b", "c")]
        rows = score_predictions(selected, [prediction("a", "MRI"), {"id": "b", "status": "generation_error"}])
        result = diagnostic_summary(rows)
        self.assertEqual(result["status"], "incomplete")
        self.assertIsNone(result["plain_whole_answer_em_accuracy_pct"])
        self.assertIsNone(result["boxed_strict_em_accuracy_pct"])
        self.assertEqual(result["plain_generated_only_em_accuracy_pct"], 100)
        self.assertEqual(result["missing_count"], 1)
        self.assertEqual(result["error_count"], 1)

    def test_reference_is_not_taken_from_prediction_metadata(self):
        selected = [{"id": "a", "dataset": "slake", "question": "Q", "answer_type": "open", "answer": "MRI"}]
        row = score_predictions(selected, [{**prediction("a", "CT"), "reference": "CT", "answer_type": "closed"}])[0]
        self.assertEqual(row["reference"], "MRI")
        self.assertEqual(row["answer_type"], "open")
        self.assertFalse(row["plain_whole_answer_correct"])


class SavedDiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        for name in ("model.py", "scoring.py"):
            shutil.copyfile(ROOT / "src" / name, self.root / "src" / name)
        shutil.copyfile(ROOT / "evaluate.py", self.root / "evaluate.py")
        self.protocol = read_protocol(ROOT / "configs/benchmark_protocol.json")
        self.settings = json.loads((ROOT / "settings.json").read_text())
        self.data = self.root / "data"
        (self.data / "images").mkdir(parents=True)
        rows = []
        for dataset_index, dataset in enumerate(("vqa_rad", "slake")):
            for index in range(5):
                image = Image.new("RGB", (8, 8), (dataset_index * 90, index * 40, 40))
                path = self.data / "images" / f"{dataset}_{index}.png"; image.save(path)
                rows.append({"id": f"{dataset}:{index}", "dataset": dataset, "role": "feedback", "question": "Which modality?",
                    "answer_type": "open", "answer": "CT", "image_path": "images/" + path.name,
                    "image_sha256": decoded_sha256(image), "image_file_sha256": file_sha256(path)})
        for role in ("train", "feedback", "test"):
            write_jsonl(self.data / (role + ".jsonl"), rows if role == "feedback" else [])
        write_jsonl(self.data / "exclusions.jsonl", [])
        write_json(self.data / "dataset.json", {"protocol": self.protocol, "sources": {},
            "counts": {"train": 0, "feedback": 10, "test": 0},
            "manifest_sha256": {role: file_sha256(self.data / (role + ".jsonl")) for role in ("train", "feedback", "test")},
            "exclusions_sha256": file_sha256(self.data / "exclusions.jsonl")})
        self.comparison = self.root / "outputs" / "comparison"
        self.comparison.mkdir(parents=True)
        report = {"status": "complete", "role": "feedback", "experiment_id": "concise-boxed-v1", "settings": self.settings,
            "original_prompt": self.protocol["prompt"], "trial_prompt": "Answer briefly in a box.",
            "dataset_metadata_sha256": file_sha256(self.data / "dataset.json"), "datasets": {}, "baseline_evidence": {}}
        self.trials = {}
        for dataset in ("vqa_rad", "slake"):
            summaries, paths = {}, {}
            for branch in ("original", "trial"):
                directory = create_run(self.root, self.data, dataset, "feedback", 5, self.settings, self.protocol)
                record = json.loads((directory / "run.json").read_text())
                if branch == "trial":
                    record["signature"]["protocol"].update(prompt=report["trial_prompt"],
                        protocol_id=self.protocol["protocol_id"] + "+" + report["experiment_id"])
                record["signature"]["code_sha256"] = {k.replace("/", "\\"): v for k, v in record["signature"]["code_sha256"].items()}
                write_json(directory / "run.json", record)
                write_jsonl(directory / "predictions.jsonl", [prediction(identity, "The image is CT." if branch == "original" else "CT.")
                            for identity in record["signature"]["selected_ids"]])
                metrics = score_run(directory, self.data)
                write_json(directory / "resources.json", {"worker_exit_code": 0, "limit_reached": None})
                paths[branch], summaries[branch] = directory, metrics["summary"]
            snapshot = self.comparison / "baseline" / dataset
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(paths["original"], snapshot)
            report["baseline_evidence"][dataset] = {"file_sha256": {name: file_sha256(snapshot / name)
                for name in ("run.json", "predictions.jsonl", "metrics.json", "scored.jsonl", "resources.json")}}
            report["datasets"][dataset] = {"trial_run": str(paths["trial"]),
                "selected_ids": json.loads((paths["trial"] / "run.json").read_text())["signature"]["selected_ids"],
                "baseline": {"summary": summaries["original"]}, "trial": {"summary": summaries["trial"]}}
            self.trials[dataset] = paths["trial"]
        self.report_path = self.comparison / "comparison.json"
        write_json(self.report_path, report)

    def test_read_only_end_to_end_report_never_calls_model(self):
        before = {str(p): file_sha256(p) for p in self.root.rglob("*") if p.is_file()}
        with patch("src.model.load_primary", side_effect=AssertionError("No inference permitted")) as load, \
                patch("src.model.generate_response", side_effect=AssertionError("No inference permitted")) as generate:
            result = diagnose_answers(self.root, self.data, self.comparison)
        load.assert_not_called(); generate.assert_not_called()
        self.assertEqual(result["status"], "complete")
        self.assertFalse(result["new_inference_performed"])
        for dataset in ("vqa_rad", "slake"):
            self.assertEqual(result["datasets"][dataset]["trial"]["plain_whole_answer_em_accuracy_pct"], 100)
            self.assertEqual(result["datasets"][dataset]["trial"]["boxed_strict_em_accuracy_pct"], 0)
            self.assertEqual(result["datasets"][dataset]["original"]["plain_whole_answer_em_accuracy_pct"], 0)
        for name, digest in before.items():
            self.assertEqual(file_sha256(name), digest)
        self.assertTrue((Path(result["diagnostic_folder"]) / "diagnostic.md").exists())
        self.assertEqual(len(list(Path(result["diagnostic_folder"]).glob("*.jsonl"))), 4)

    def test_rejects_incomplete_comparison_and_changed_dataset_identity(self):
        record = json.loads(self.report_path.read_text())
        for key, value in (("status", "incomplete"), ("dataset_metadata_sha256", "0" * 64)):
            with self.subTest(key=key):
                write_json(self.report_path, {**record, key: value})
                with self.assertRaises(ValueError):
                    diagnose_answers(self.root, self.data, self.comparison)

    def test_rejects_test_role_changed_prompt_and_foreign_ids(self):
        path = self.trials["slake"] / "run.json"
        original = json.loads(path.read_text())
        changes = [("role", "test"), ("selected_ids", ["foreign"] * 5)]
        changed = json.loads(json.dumps(original)); changed["signature"]["protocol"]["prompt"] = "different"
        variants = [changed]
        for key, value in changes:
            record = json.loads(json.dumps(original)); record["signature"][key] = value; variants.append(record)
        for record in variants:
            write_json(path, record)
            with self.assertRaises(ValueError):
                diagnose_answers(self.root, self.data, self.comparison)

    def test_rejects_incomplete_saved_predictions(self):
        path = self.trials["vqa_rad"] / "predictions.jsonl"
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "saved run is incomplete"):
            diagnose_answers(self.root, self.data, self.comparison)

    def test_rejects_changed_original_evidence_and_scorer(self):
        path = self.comparison / "baseline" / "vqa_rad" / "predictions.jsonl"
        original = path.read_bytes(); path.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "Original baseline evidence differs"):
            diagnose_answers(self.root, self.data, self.comparison)
        path.write_bytes(original)
        (self.root / "src/scoring.py").write_text("# changed\n")
        with self.assertRaisesRegex(ValueError, "boxed scorer implementation changed"):
            diagnose_answers(self.root, self.data, self.comparison)

    def test_cli_routes_to_scoring_without_generation_or_data_preparation(self):
        import evaluate
        with patch("sys.argv", ["evaluate.py", "diagnose-answers", "--comparison-dir", str(self.comparison)]), \
                patch("src.answer_diagnostic.diagnose_answers", return_value={"status": "complete"}) as diagnose, \
                patch("src.evaluation.execute_run") as execute, patch("src.benchmarks.prepare_bundle") as prepare, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(evaluate.main(), 0)
        diagnose.assert_called_once(); execute.assert_not_called(); prepare.assert_not_called()


if __name__ == "__main__":
    unittest.main()
