import io
import json
import contextlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

from PIL import Image

from src.benchmarks import (assign_roles, decoded_sha256, export_source_rows, file_sha256,
                            image_path, load_bundle, prepare_bundle, validate_records, write_jsonl)
from src.evaluation import (create_run, evaluation_worker, indexed_predictions,
                            prediction_journal, read_protocol, score_run, select_records)
from src.runtime import write_json
from src.scoring import boxed_answer, normalize_answer, score_answer, summarize, token_f1


ROOT = Path(__file__).resolve().parents[1]


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.protocol = read_protocol(ROOT / "configs" / "benchmark_protocol.json")

    def bundle(self):
        data = self.root / "data"
        (data / "images").mkdir(parents=True)
        records = []
        for index, role in enumerate(("train", "feedback", "test")):
            image = Image.new("RGB", (8, 8), (index * 70, 10, 20))
            path = data / "images" / f"{index}.png"
            image.save(path)
            row = {"id": role, "role": role, "dataset": "vqa_rad", "source_split": role,
                   "question": "Which organ is shown?", "answer": "kidney",
                   "answer_type": "open", "image_path": "images/" + path.name,
                   "image_sha256": decoded_sha256(image), "image_file_sha256": file_sha256(path)}
            records.append(row)
            write_jsonl(data / (role + ".jsonl"), [row])
        write_jsonl(data / "exclusions.jsonl", [])
        write_json(data / "dataset.json", {"protocol": self.protocol, "sources": {},
                   "counts": {role: 1 for role in ("train", "feedback", "test")},
                   "manifest_sha256": {role: file_sha256(data / (role + ".jsonl"))
                                       for role in ("train", "feedback", "test")},
                   "exclusions_sha256": file_sha256(data / "exclusions.jsonl")})
        return data, records

    def run_directory(self, data):
        project = self.root / "project"
        (project / "src").mkdir(parents=True)
        (project / "src" / "example.py").write_text("pass\n")
        (project / "evaluate.py").write_text("pass\n")
        settings = {"model_id": "test-only", "revision": "a" * 40}
        directory = create_run(project, data, "vqa_rad", "feedback", 1, settings, self.protocol)
        return project, directory, settings

    def test_last_box_and_malformed_box(self):
        self.assertEqual(boxed_answer(r"Earlier \boxed{no}; final \boxed{yes}"), "yes")
        self.assertEqual(boxed_answer(r"\boxed{left {kidney}}"), "left {kidney}")
        for response in ("yes", r"\boxed{}", r"\boxed{yes} then \boxed{no"):
            self.assertIsNone(boxed_answer(response))

    def test_normalization_preserves_medical_distinctions(self):
        self.assertEqual(normalize_answer("  LEFT   kidney. "), "left kidney")
        self.assertNotEqual(normalize_answer("no mass"), normalize_answer("mass"))
        self.assertNotEqual(normalize_answer("1.5 cm"), normalize_answer("15 cm"))
        self.assertFalse(score_answer(r"\boxed{kidney.}", "kidney")["strict_correct"])
        self.assertTrue(score_answer(r"\boxed{kidney.}", "kidney")["normalized_correct"])

    def test_f1_counts_repeated_tokens(self):
        self.assertAlmostEqual(token_f1("left left kidney", "left kidney"), .8)
        self.assertEqual(token_f1("no", "yes"), 0)

    def test_format_failure_is_wrong_answer_not_missing_coverage(self):
        result = summarize([{"status": "generated", **score_answer("kidney", "kidney")}])
        self.assertEqual(result["coverage_pct"], 100)
        self.assertEqual(result["strict_em_accuracy_pct"], 0)
        self.assertEqual(result["parse_failure_count"], 1)

    def test_incomplete_run_has_no_primary_accuracy(self):
        rows = [{"status": "generated", **score_answer(r"\boxed{kidney}", "kidney")},
                {"status": "generation_error"}, {"status": "missing"}]
        result = summarize(rows)
        self.assertIsNone(result["strict_em_accuracy_pct"])
        self.assertAlmostEqual(result["coverage_pct"], 100 / 3)
        self.assertEqual(result["generated_only_metrics"]["strict_em_accuracy_pct"], 100)

    def test_selection_is_reproducible_across_input_order(self):
        _, rows = self.bundle()
        self.assertEqual(select_records(rows, "vqa_rad", "feedback", 7, 1),
                         select_records(list(reversed(rows)), "vqa_rad", "feedback", 7, 1))
        with self.assertRaises(ValueError):
            select_records(rows, "vqa_rad", "feedback", 7, 0)

    def test_default_cli_has_no_data_or_model_operation(self):
        import evaluate
        output = io.StringIO()
        with patch("sys.argv", ["evaluate.py"]), contextlib.redirect_stdout(output), \
                patch("src.benchmarks.prepare_bundle") as prepare:
            self.assertEqual(evaluate.main(), 0)
            prepare.assert_not_called()
        self.assertIn("Prepare medical VQA", output.getvalue())
        self.assertNotIn("==SUPPRESS==", output.getvalue())

    def test_final_test_requires_explicit_flag_before_loading_data(self):
        import evaluate
        output = io.StringIO()
        with patch("sys.argv", ["evaluate.py", "run", "--benchmark", "vqa_rad",
                                "--role", "test", "--full"]), \
                contextlib.redirect_stderr(output), patch("src.evaluation.create_run") as create:
            self.assertEqual(evaluate.main(), 2)
            create.assert_not_called()
        self.assertIn("reserved for the frozen final protocol", output.getvalue())

    def test_paths_reject_escape_on_windows_and_posix(self):
        for value in ("../outside.png", "/tmp/image.png", "C:/image.png", "images\\x.png"):
            with self.assertRaises(ValueError):
                image_path(self.root, value)

    def test_exact_test_images_are_excluded_from_adaptation(self):
        rows = [{"id": "a", "dataset": "vqa_rad", "source_split": "train",
                 "image_sha256": "same", "question": "Q", "answer": "A"},
                {"id": "b", "dataset": "vqa_rad", "source_split": "test",
                 "image_sha256": "same", "question": "Q2", "answer": "B"}]
        kept, excluded = assign_roles(rows, 7, .1)
        self.assertEqual([r["id"] for r in kept], ["b"])
        self.assertEqual(excluded[0]["reason"], "test_image_overlap")

    def test_validation_images_never_enter_training(self):
        rows = [{"id": str(index), "dataset": "slake", "source_split": split,
                 "image_sha256": "same", "question": f"Q{index}", "answer": "A"}
                for index, split in enumerate(("train", "validation"))]
        kept, _ = assign_roles(rows, 7, .1)
        self.assertEqual({r["role"] for r in kept}, {"feedback"})

    def test_conflicting_adaptation_references_fail(self):
        rows = [{"id": str(index), "dataset": "slake", "source_split": "train",
                 "image_sha256": "same", "question": "Q", "answer": answer}
                for index, answer in enumerate(("A", "B"))]
        with self.assertRaises(ValueError):
            assign_roles(rows, 7, .1)

    def test_changed_image_or_manifest_is_detected(self):
        data, _ = self.bundle()
        load_bundle(data, verify_images=True)
        (data / "images" / "0.png").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Image file changed"):
            load_bundle(data, verify_images=True)
        (data / "train.jsonl").write_text("{}\n")
        with self.assertRaisesRegex(ValueError, "Manifest changed"):
            load_bundle(data)

    def test_same_image_across_roles_is_detected(self):
        data, rows = self.bundle()
        rows[1]["image_sha256"] = rows[0]["image_sha256"]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_records(data, rows, verify_images=False)

    def test_resume_rejects_changed_code_and_protocol(self):
        data, _ = self.bundle()
        project, directory, settings = self.run_directory(data)
        self.assertEqual(create_run(project, data, "vqa_rad", "feedback", 1,
                                   settings, self.protocol, directory), directory)
        (project / "evaluate.py").write_text("print('changed')\n")
        with self.assertRaisesRegex(ValueError, "Resume rejected"):
            create_run(project, data, "vqa_rad", "feedback", 1, settings, self.protocol, directory)
        with self.assertRaisesRegex(ValueError, "protocol differs"):
            create_run(project, data, "vqa_rad", "feedback", 1, settings,
                       {**self.protocol, "max_new_tokens": 64})

    def test_journal_recovery_preserves_original(self):
        path = self.root / "predictions.jsonl"
        original = b'{"id":"first"}\n{"id":'
        path.write_bytes(original)
        with self.assertRaises(ValueError):
            prediction_journal(path)
        self.assertEqual(prediction_journal(path, recover=True), [{"id": "first"}])
        self.assertEqual(next(self.root.glob("*.interrupted-*")).read_bytes(), original)
        path.write_text('{invalid}\n{"id":"last"}\n')
        with self.assertRaises(ValueError):
            prediction_journal(path, recover=True)

    def test_duplicate_and_foreign_predictions_are_rejected(self):
        prediction = {"id": "a", "status": "generation_error"}
        for rows in ([prediction, prediction], [{**prediction, "id": "other"}]):
            with self.assertRaises(ValueError):
                indexed_predictions(rows, ["a"])

    def test_worker_uses_question_only_and_resumes_without_generation(self):
        data, _ = self.bundle()
        _, directory, _ = self.run_directory(data)
        generator = Mock(return_value={"raw_response": r"\boxed{kidney}", "generated_tokens": 6,
                                       "generation_seconds": .5, "hit_token_limit": False})
        model = Mock()
        model.parameters.return_value = []
        loader = Mock(return_value=(model, object()))
        self.assertEqual(evaluation_worker(directory, data, loader, generator), 0)
        prompt = generator.call_args.args[3]
        self.assertEqual(prompt, "Which organ is shown?\n" + self.protocol["prompt"])
        self.assertNotIn("kidney", prompt)
        self.assertNotIn("open", prompt)
        self.assertEqual(evaluation_worker(directory, data, loader, generator), 0)
        self.assertEqual(generator.call_count, 1)
        metrics = score_run(directory, data)
        self.assertEqual(metrics["role"], "feedback")
        self.assertEqual(metrics["scope"], "subset")
        self.assertEqual(metrics["summary"]["strict_em_accuracy_pct"], 100)
        self.assertFalse((directory / "run.lock").exists())

    def test_generation_error_is_recorded_without_false_accuracy(self):
        data, _ = self.bundle()
        _, directory, _ = self.run_directory(data)
        model = Mock()
        model.parameters.return_value = []
        self.assertEqual(evaluation_worker(directory, data, lambda _: (model, None),
                                          Mock(side_effect=RuntimeError("fixture failure"))), 2)
        metrics = score_run(directory, data)["summary"]
        self.assertEqual(metrics["error_count"], 1)
        self.assertIsNone(metrics["strict_em_accuracy_pct"])

    def test_unstarted_run_scores_as_incomplete(self):
        data, _ = self.bundle()
        _, directory, _ = self.run_directory(data)
        metrics = score_run(directory, data)["summary"]
        self.assertEqual(metrics["missing_count"], 1)
        self.assertIsNone(metrics["strict_em_accuracy_pct"])

    def test_data_preparation_preserves_existing_destination(self):
        destination = self.root / "existing"
        destination.mkdir()
        marker = destination / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            prepare_bundle(destination, {}, self.protocol)
        self.assertEqual(marker.read_text(), "keep")

    def test_source_export_filters_language_and_records_missing_training_answer(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        buffer = io.BytesIO()
        Image.new("RGB", (8, 8), "red").save(buffer, format="PNG")
        paths = {}
        for split in ("train", "test"):
            path = self.root / (split + ".parquet")
            pq.write_table(pa.Table.from_pylist([{"image": {"bytes": buffer.getvalue(), "path": None},
                                                "question": "Q?", "answer": "yes"}]), path)
            paths[("vqa_rad", split)] = path
        for split in ("train", "validation", "test"):
            base = {"qid": 1, "img_name": "xmlab1/source.jpg", "question": "Q?",
                    "answer": "A", "answer_type": "OPEN", "q_lang": "en"}
            rows = [base, {**base, "qid": 2, "q_lang": "zh"}]
            if split == "train":
                rows.append({**base, "qid": 3, "answer": ""})
            path = self.root / (split + ".json")
            path.write_text(json.dumps(rows))
            paths[("slake", split)] = path
        archive_path = self.root / "imgs.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("imgs/xmlab1/source.jpg", buffer.getvalue())
        paths[("slake", "images")] = archive_path
        destination = self.root / "export"
        destination.mkdir()
        records, exclusions = export_source_rows(paths, destination)
        self.assertEqual(len(records), 5)
        self.assertEqual(sum(r["reason"] == "non_english" for r in exclusions), 3)
        self.assertEqual(sum(r["reason"] == "missing_question_or_reference" for r in exclusions), 1)
        test_path = paths[("slake", "test")]
        test_path.write_text(json.dumps([{**base, "answer": ""}]))
        other = self.root / "other"
        other.mkdir()
        with self.assertRaisesRegex(ValueError, "Test reference is missing"):
            export_source_rows(paths, other)


if __name__ == "__main__":
    unittest.main()
