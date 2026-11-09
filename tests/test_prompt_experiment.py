import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from src.benchmarks import decoded_sha256, file_sha256, write_jsonl
from src.evaluation import create_run, evaluation_worker, read_protocol, score_run
from src.prompt_experiment import BASELINE_FILES, compare_prompts
from src.runtime import write_json


ROOT = Path(__file__).resolve().parents[1]


class PromptExperimentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "configs").mkdir()
        (self.root / "src").mkdir()
        shutil.copyfile(ROOT / "configs/answer_format.json", self.root / "configs/answer_format.json")
        shutil.copyfile(ROOT / "evaluate.py", self.root / "evaluate.py")
        for name in ("model.py", "scoring.py", "evaluation.py", "prompt_experiment.py"):
            shutil.copyfile(ROOT / "src" / name, self.root / "src" / name)
        self.settings = json.loads((ROOT / "settings.json").read_text())
        self.protocol = read_protocol(ROOT / "configs/benchmark_protocol.json")
        memory = patch("psutil.virtual_memory", return_value=SimpleNamespace(available=64 * 1024**3, total=64 * 1024**3))
        memory.start()
        self.addCleanup(memory.stop)
        self.data = self.root / "data"
        (self.data / "images").mkdir(parents=True)
        self.rows = []
        for dataset_index, dataset in enumerate(("vqa_rad", "slake")):
            for index, role in enumerate(("train", "test", "feedback", "feedback", "feedback", "feedback", "feedback")):
                image = Image.new("RGB", (8, 8), (dataset_index * 100, index * 25, 30))
                path = self.data / "images" / f"{dataset}_{index}.png"
                image.save(path)
                self.rows.append({"id": f"{dataset}:{role}:{index}", "dataset": dataset, "role": role,
                    "question": "Which organ is shown?", "answer": "kidney", "answer_type": "open",
                    "image_path": "images/" + path.name, "image_sha256": decoded_sha256(image),
                    "image_file_sha256": file_sha256(path)})
        for role in ("train", "feedback", "test"):
            write_jsonl(self.data / (role + ".jsonl"), [r for r in self.rows if r["role"] == role])
        write_jsonl(self.data / "exclusions.jsonl", [])
        write_json(self.data / "dataset.json", {"protocol": self.protocol, "sources": {},
            "counts": {"train": 2, "feedback": 10, "test": 2},
            "manifest_sha256": {role: file_sha256(self.data / (role + ".jsonl")) for role in ("train", "feedback", "test")},
            "exclusions_sha256": file_sha256(self.data / "exclusions.jsonl")})
        self.baselines = {}
        for dataset in ("vqa_rad", "slake"):
            directory = create_run(self.root, self.data, dataset, "feedback", 5, self.settings, self.protocol)
            run = json.loads((directory / "run.json").read_text())
            write_jsonl(directory / "predictions.jsonl", [{"id": identity, "status": "generated",
                "raw_response": "kidney", "generated_tokens": 1, "generation_seconds": 1.0,
                "hit_token_limit": False} for identity in run["signature"]["selected_ids"]])
            score_run(directory, self.data)
            write_json(directory / "resources.json", {"worker_exit_code": 0, "limit_reached": None,
                                                       "elapsed_seconds": 6, "memory_measurement_plausible": False})
            self.baselines[dataset] = directory

    def run_experiment(self, runner):
        return compare_prompts(self.root, self.data, self.baselines, self.settings, self.protocol, runner)

    def successful_runner(self, captured):
        def run(root, directory, data, settings, protocol):
            captured.append((directory, settings, protocol))
            model = Mock()
            model.parameters.return_value = []
            generate = Mock(return_value={"raw_response": r"\boxed{kidney}", "generated_tokens": 5,
                                         "generation_seconds": .5, "hit_token_limit": False})
            self.assertEqual(evaluation_worker(directory, data, lambda _: (model, None), generate), 0)
            for call in generate.call_args_list:
                self.assertEqual(call.args[3], "Which organ is shown?\n" + protocol["prompt"])
                self.assertNotIn("kidney", call.args[3])
            result = json.loads((directory / "metrics.json").read_text())
            result["worker"] = {"worker_exit_code": 0, "limit_reached": None, "elapsed_seconds": 4,
                                "memory_measurement_plausible": False}
            return result
        return run

    def test_pairing_controls_snapshots_and_original_files_are_preserved(self):
        before = {dataset: {name: file_sha256(directory / name) for name in BASELINE_FILES}
                  for dataset, directory in self.baselines.items()}
        data_before = file_sha256(self.data / "dataset.json")
        captured = []
        result = self.run_experiment(self.successful_runner(captured))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(captured), 2)
        for dataset, directory in self.baselines.items():
            self.assertEqual(before[dataset], {name: file_sha256(directory / name) for name in BASELINE_FILES})
            trial = result["datasets"][dataset]
            self.assertEqual(trial["original_format_success_pct"], 0)
            self.assertEqual(trial["trial_format_success_pct"], 100)
            self.assertEqual(trial["change_percentage_points"]["strict_em_accuracy"], 100)
            trial_run = json.loads((Path(result["trial_folders"][dataset]) / "run.json").read_text())
            original = json.loads((directory / "run.json").read_text())
            self.assertEqual(trial_run["signature"]["selected_ids"], original["signature"]["selected_ids"])
            self.assertEqual(trial_run["signature"]["role"], "feedback")
            self.assertEqual(trial_run["signature"]["settings"], self.settings)
            for name, expected in before[dataset].items():
                self.assertEqual(file_sha256(Path(result["comparison_folder"]) / "baseline" / dataset / name), expected)
        for _, settings, protocol in captured:
            self.assertEqual(settings, self.settings)
            self.assertEqual({k: v for k, v in protocol.items() if k not in ("prompt", "protocol_id")},
                             {k: v for k, v in self.protocol.items() if k not in ("prompt", "protocol_id")})
            self.assertNotEqual(protocol["prompt"], self.protocol["prompt"])
        self.assertEqual(file_sha256(self.data / "dataset.json"), data_before)
        self.assertTrue((Path(result["comparison_folder"]) / "comparison.md").is_file())

    def test_preflight_rejects_changed_controls_before_generation(self):
        path = self.baselines["vqa_rad"] / "run.json"
        original = json.loads(path.read_text())
        variants = []
        for key, value in (("role", "test"), ("limit", 4), ("dataset_metadata_sha256", "0" * 64)):
            record = json.loads(json.dumps(original)); record["signature"][key] = value; variants.append(record)
        record = json.loads(json.dumps(original)); record["signature"]["settings"]["image_edge"] = 1024; variants.append(record)
        record = json.loads(json.dumps(original)); record["signature"]["protocol"]["max_new_tokens"] = 64; variants.append(record)
        record = json.loads(json.dumps(original)); record["signature"]["code_sha256"]["src/model.py"] = "0" * 64; variants.append(record)
        record = json.loads(json.dumps(original)); record["signature"]["packages"]["torch"] = "different"; variants.append(record)
        record = json.loads(json.dumps(original)); record["signature"]["selected_ids"][0] = "vqa_rad:train:0"; variants.append(record)
        for index, record in enumerate(variants):
            with self.subTest(variant=index):
                write_json(path, record)
                runner = Mock()
                with self.assertRaises(ValueError):
                    self.run_experiment(runner)
                runner.assert_not_called()
        write_json(path, original)
        self.assertFalse(list((self.root / "outputs").glob("*_prompt_comparison_*")))

    def test_windows_source_paths_are_accepted_and_scorer_changes_are_rejected(self):
        for directory in self.baselines.values():
            path = directory / "run.json"
            record = json.loads(path.read_text())
            record["signature"]["code_sha256"] = {k.replace("/", "\\"): v for k, v in record["signature"]["code_sha256"].items()}
            write_json(path, record)
        self.assertEqual(self.run_experiment(self.successful_runner([]))["status"], "complete")
        path = self.baselines["vqa_rad"] / "run.json"
        record = json.loads(path.read_text()); record["signature"]["code_sha256"]["src\\scoring.py"] = "0" * 64
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "Scoring implementation changed"):
            score_run(path.parent, self.data)
        with self.assertRaisesRegex(ValueError, "model implementation or scorer"):
            self.run_experiment(Mock())

    def test_incomplete_baseline_is_rejected(self):
        path = self.baselines["vqa_rad"] / "predictions.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        write_jsonl(path, rows[:-1])
        runner = Mock()
        with self.assertRaisesRegex(ValueError, "all five generations"):
            self.run_experiment(runner)
        runner.assert_not_called()

    def test_metrics_disagreeing_with_raw_baseline_are_rejected(self):
        path = self.baselines["slake"] / "metrics.json"
        record = json.loads(path.read_text()); record["summary"]["strict_em_accuracy_pct"] = 100
        write_json(path, record)
        with self.assertRaisesRegex(ValueError, "metrics disagree"):
            self.run_experiment(Mock())

    def test_incomplete_trial_is_reported_and_second_trial_is_not_started(self):
        def interrupted(root, directory, data, settings, protocol):
            result = score_run(directory, data)
            result["worker"] = {"worker_exit_code": 2, "limit_reached": "time_limit"}
            return result
        runner = Mock(side_effect=interrupted)
        result = self.run_experiment(runner)
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(runner.call_count, 1)
        self.assertIsNone(result["datasets"]["vqa_rad"]["trial_strict_em_accuracy_pct"])
        self.assertIsNone(result["datasets"]["vqa_rad"]["change_percentage_points"]["strict_em_accuracy"])
        self.assertIn("Incomplete", result["error"])

    def test_baseline_mutation_during_experiment_invalidates_comparison(self):
        successful = self.successful_runner([])
        def mutating(*args):
            result = successful(*args)
            path = self.baselines["vqa_rad"] / "resources.json"
            path.write_text(path.read_text() + " ")
            return result
        result = self.run_experiment(mutating)
        self.assertEqual(result["status"], "incomplete")
        self.assertIn("comparison is invalid", result["error"])


if __name__ == "__main__":
    unittest.main()
