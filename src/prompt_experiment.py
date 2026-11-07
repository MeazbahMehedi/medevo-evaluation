from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import uuid

from .benchmarks import file_sha256, load_bundle
from .evaluation import (create_run, execute_run, indexed_predictions, prediction_journal,
                         recorded_source_hashes, select_records, source_fingerprints)
from .runtime import environment_snapshot, safe_error, write_json
from .scoring import score_answer, summarize


BASELINE_FILES = ("run.json", "predictions.jsonl", "metrics.json", "scored.jsonl", "resources.json")


def baseline_evidence(root, data_dir, directory, dataset, settings, protocol, records, environment):
    directory = Path(directory).resolve()
    if (directory / "run.lock").exists():
        raise ValueError("A baseline is locked; wait until its worker has stopped")
    hashes = {name: file_sha256(directory / name) for name in BASELINE_FILES}
    signature = json.loads((directory / "run.json").read_text(encoding="utf-8"))["signature"]
    if signature["benchmark"] != dataset or signature["role"] != "feedback":
        raise ValueError("Prompt comparison accepts feedback baselines only, from the specified dataset")
    if signature["limit"] != 5 or signature["scope"] != "subset" or len(signature["selected_ids"]) != 5:
        raise ValueError("This diagnostic requires exactly five selected feedback questions per dataset")
    if signature["settings"] != settings or signature["protocol"] != protocol:
        raise ValueError("Baseline model settings or protocol differ from the current configuration")
    if signature["dataset_metadata_sha256"] != file_sha256(Path(data_dir) / "dataset.json"):
        raise ValueError("Baseline and current dataset identities differ")
    current_sources = source_fingerprints(root)
    old_sources = recorded_source_hashes(signature)
    for name in ("src/model.py", "src/scoring.py"):
        if old_sources.get(name) != current_sources.get(name) or name not in current_sources:
            raise ValueError("The model implementation or scorer differs from the original baseline")
    if signature["packages"] != environment["packages"] or signature["python"] != environment["python"]:
        raise ValueError("Baseline and current interpreter or direct package versions differ")
    selected = select_records(records, dataset, "feedback", protocol["seed"], 5)
    ids = [r["id"] for r in selected]
    if ids != signature["selected_ids"]:
        raise ValueError("Baseline IDs differ from the fixed feedback selection")
    predictions = indexed_predictions(prediction_journal(directory / "predictions.jsonl"), ids)
    if len(predictions) != 5 or any(p["status"] != "generated" for p in predictions.values()):
        raise ValueError("Finish the baseline first; all five generations must be present")
    summary = summarize([{**predictions[r["id"]],
                          **score_answer(predictions[r["id"]]["raw_response"], r["answer"])} for r in selected])
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if metrics["summary"] != summary:
        raise ValueError("Stored baseline metrics disagree with the raw predictions")
    resources = json.loads((directory / "resources.json").read_text(encoding="utf-8"))
    if resources["worker_exit_code"] != 0 or resources["limit_reached"] is not None:
        raise ValueError("The baseline worker did not finish cleanly")
    return {"directory": str(directory), "file_sha256": hashes, "signature": signature,
            "summary": summary, "resources": resources}


def format_success(summary):
    if summary["status"] != "complete":
        return None
    return 100 * (summary["generated_count"] - summary["parse_failure_count"]) / summary["selected_count"]


def dataset_comparison(baseline, trial_dir, trial_result):
    original, trial = baseline["summary"], trial_result["summary"]
    metric_names = ("strict_em_accuracy_pct", "normalized_em_accuracy_pct", "mean_token_f1_pct")
    baseline_format, trial_format = format_success(original), format_success(trial)
    return {"baseline_run": baseline["directory"], "trial_run": str(trial_dir),
            "selected_ids": baseline["signature"]["selected_ids"],
            "baseline": {"summary": original, "format_success_pct": baseline_format,
                         "resources": baseline["resources"]},
            "trial": {"summary": trial, "format_success_pct": trial_format,
                      "resources": trial_result["worker"]},
            "change_percentage_points": {
                **{name.removesuffix("_pct"): trial[name] - original[name]
                   if trial[name] is not None and original[name] is not None else None for name in metric_names},
                "format_success": trial_format - baseline_format if trial_format is not None else None}}


def write_comparison(directory, report):
    write_json(directory / "comparison.json", report)
    lines = ["# Answer-format experiment", "", "Feedback development only; not final test evaluation.", "",
             "The original prompt and concise prompt use the same selected questions, model, image settings, token limit and scorer.", "",
             "| Dataset | Questions | Original format % | Trial format % | Original strict EM % | Trial strict EM % |",
             "|---|---:|---:|---:|---:|---:|"]
    def cell(value):
        return "incomplete" if value is None else f"{value:.1f}"
    for dataset, result in report["datasets"].items():
        old, new = result["baseline"], result["trial"]
        lines.append(f"| {dataset} | {len(result['selected_ids'])} | {cell(old['format_success_pct'])} | "
                     f"{cell(new['format_success_pct'])} | {cell(old['summary']['strict_em_accuracy_pct'])} | "
                     f"{cell(new['summary']['strict_em_accuracy_pct'])} |")
    lines.extend(["", f"Status: {report['status']}.", "",
                  "Original prompt:", "", "```text", report["original_prompt"], "```", "",
                  "Trial prompt:", "", "```text", report["trial_prompt"], "```", "",
                  "Raw predictions and per-question scores remain in the linked run folders. Original baseline files are copied under baseline/.", "",
                  "Ten reused feedback questions do not estimate benchmark accuracy. Format compliance and answer correctness are separate outcomes.",
                  "Any reported timing difference is an observation across separate attempts, not a controlled hardware speedup.",
                  "The concise prompt differs from the paper and is not adopted for final evaluation by this command."])
    if report.get("error"):
        lines.extend(["", "Error: " + report["error"]])
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_prompts(root, data_dir, baseline_dirs, settings, protocol, runner=None):
    root, data_dir = Path(root).resolve(), Path(data_dir).resolve()
    metadata, records = load_bundle(data_dir, verify_images=True)
    if metadata["protocol"] != protocol:
        raise ValueError("Data preparation protocol differs from the baseline protocol")
    profile_path = root / "configs" / "answer_format.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if set(profile) != {"experiment_id", "prompt"} or not isinstance(profile["experiment_id"], str) or not profile["experiment_id"].strip():
        raise ValueError("Invalid answer-format profile")
    if not isinstance(profile["prompt"], str) or not profile["prompt"].strip() or profile["prompt"] == protocol["prompt"]:
        raise ValueError("The trial prompt must be a nonempty alternative to the original prompt")
    import psutil
    if psutil.virtual_memory().available < settings["minimum_available_gib"] * 1024**3:
        raise RuntimeError("Available RAM is below the configured minimum")
    environment = environment_snapshot(root)
    baselines = {dataset: baseline_evidence(root, data_dir, baseline_dirs[dataset], dataset,
                   settings, protocol, records, environment) for dataset in ("vqa_rad", "slake")}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "outputs" / f"{stamp}_prompt_comparison_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    report = {"experiment_id": profile["experiment_id"], "status": "running", "role": "feedback",
              "scope": "five fixed questions per dataset", "comparison_folder": str(directory),
              "original_prompt": protocol["prompt"], "trial_prompt": profile["prompt"],
              "dataset_metadata_sha256": file_sha256(data_dir / "dataset.json"),
              "profile_sha256": file_sha256(profile_path), "settings": settings,
              "source_sha256": source_fingerprints(root), "baseline_evidence": baselines, "datasets": {}}
    print(f"Comparison folder: {directory}", flush=True)
    try:
        for dataset, evidence in baselines.items():
            snapshot = directory / "baseline" / dataset
            snapshot.mkdir(parents=True)
            for name, expected in evidence["file_sha256"].items():
                shutil.copyfile(Path(evidence["directory"]) / name, snapshot / name)
                if file_sha256(snapshot / name) != expected:
                    raise ValueError("A baseline changed while its evidence was copied")
        write_comparison(directory, report)
        for dataset, evidence in baselines.items():
            trial_dir = create_run(root, data_dir, dataset, "feedback", 5, settings, protocol)
            trial_run = json.loads((trial_dir / "run.json").read_text(encoding="utf-8"))
            if trial_run["signature"]["selected_ids"] != evidence["signature"]["selected_ids"]:
                raise ValueError("Trial and original selections differ")
            trial_protocol = {**protocol, "protocol_id": protocol["protocol_id"] + "+" + profile["experiment_id"],
                              "prompt": profile["prompt"]}
            trial_run["signature"]["protocol"] = trial_protocol
            trial_run["signature"]["prompt_experiment"] = {
                "experiment_id": profile["experiment_id"], "comparison_folder": str(directory),
                "original_protocol": protocol, "baseline_run": evidence["directory"],
                "baseline_file_sha256": evidence["file_sha256"], "profile_sha256": report["profile_sha256"]}
            trial_run["purpose"] = "answer-format diagnostic on fixed feedback questions"
            write_json(trial_dir / "run.json", trial_run)
            report.setdefault("trial_folders", {})[dataset] = str(trial_dir)
            write_comparison(directory, report)
            print(f"Trial folder ({dataset}): {trial_dir}", flush=True)
            result = (runner or execute_run)(root, trial_dir, data_dir, settings, trial_protocol)
            report["datasets"][dataset] = dataset_comparison(evidence, trial_dir, result)
            write_comparison(directory, report)
            if result["worker"]["worker_exit_code"] != 0 or result["worker"]["limit_reached"] or result["summary"]["status"] != "complete":
                raise RuntimeError(f"Incomplete {dataset} trial; retain its partial results before proceeding")
        for evidence in baselines.values():
            for name, expected in evidence["file_sha256"].items():
                if file_sha256(Path(evidence["directory"]) / name) != expected:
                    raise ValueError("Original baseline changed during the experiment; comparison is invalid")
        report.update(status="complete", original_baseline_files_unchanged=True)
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="incomplete", error=safe_error(error))
    write_comparison(directory, report)
    return {"status": report["status"], "comparison_folder": str(directory),
            "trial_folders": report.get("trial_folders", {}),
            "datasets": {dataset: {
                "question_count": len(value["selected_ids"]),
                "original_format_success_pct": value["baseline"]["format_success_pct"],
                "trial_format_success_pct": value["trial"]["format_success_pct"],
                "original_strict_em_accuracy_pct": value["baseline"]["summary"]["strict_em_accuracy_pct"],
                "trial_strict_em_accuracy_pct": value["trial"]["summary"]["strict_em_accuracy_pct"],
                "change_percentage_points": value["change_percentage_points"],
                "trial_generation_seconds": value["trial"]["summary"]["generation_seconds"]}
                for dataset, value in report["datasets"].items()},
            **({"error": report["error"]} if "error" in report else {})}
