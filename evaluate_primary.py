"""Evaluate the frozen 7B answering configuration on ten feedback questions."""

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from PIL import Image

from check_primary import (AUDIT_IMAGE_SHA256, BASELINE_FILES as PROFILE_FILES, CONCISE_CHECK_ID,
                           CONCISE_INSTRUCTION, MODEL, ROOT, SELECTED_ID, local_json)
from src.benchmarks import file_sha256, image_path, load_bundle
from src.evaluation import indexed_predictions, prediction_journal, recorded_source_hashes, select_records
from src.prompt_experiment import BASELINE_FILES
from src.repair_audit import image_content
from src.runtime import environment_snapshot, safe_error, write_json
from src.scoring import SCORING_ID, score_answer, summarize
from src.training_collection import read_json, tree_hashes


EXPERIMENT_ID = "qwen7b-feedback-10-v1"
DATASETS = ("vqa_rad", "slake")
OPTIONS = {"temperature": 0, "seed": 7, "num_predict": 512, "num_ctx": 4096, "num_thread": 4, "num_gpu": 0}


def same_file(path, digest):
    try:
        return file_sha256(path) == digest
    except OSError:
        return False


def frozen_profile(directory):
    directory = Path(directory)
    if (directory / "run.lock").exists():
        raise ValueError("The concise-prompt profile is still running")
    before = tree_hashes(directory)
    run, summary = read_json(directory / "run.json"), read_json(directory / "summary.json")
    request, response = read_json(directory / "request.json"), read_json(directory / "response.json")
    question = read_json(ROOT / "configs/continuation_case/saved_input.json")["question"]
    if (run.get("check_id") != CONCISE_CHECK_ID or summary.get("check_id") != CONCISE_CHECK_ID
            or run.get("role") != "train" or run.get("selected_id") != SELECTED_ID
            or summary.get("selected_id") != SELECTED_ID or summary.get("status") != "complete"
            or summary.get("generation_requests_started") != 1 or run.get("max_new_tokens") != 512
            or run.get("prompt_variant") != "concise-three-step-v1"
            or response.get("model") != MODEL or response.get("done") is not True
            or response.get("done_reason") != "stop" or not response.get("message", {}).get("content", "").strip()
            or summary.get("raw_response") != response["message"]["content"]
            or not run.get("model_digest") or run.get("model_digest") != summary.get("model_digest")
            or not run.get("ollama_version")):
        raise ValueError("Use the completed same-case concise-prompt 7B run as the frozen profile")
    messages = request.get("messages", [])
    if len(messages) != 1 or set(messages[0]) != {"role", "content", "images"} or len(messages[0]["images"]) != 1:
        raise ValueError("The frozen profile must contain exactly one reference-free image/question message")
    expected = {"model": MODEL, "messages": [{"role": "user", "content": question + "\n" + CONCISE_INSTRUCTION,
                 "images": messages[0]["images"]}], "stream": False, "keep_alive": "2m", "options": OPTIONS}
    image = base64.b64decode(messages[0]["images"][0], validate=True)
    if request != expected or hashlib.sha256(image).hexdigest() != AUDIT_IMAGE_SHA256:
        raise ValueError("The concise profile's prompt, image or generation controls changed")
    if tree_hashes(directory) != before:
        raise ValueError("Profile changed during validation")
    return run, request, before


def fixed_selection(data_dir, metadata, records, baseline_dirs):
    selected, evidence = [], {}
    for dataset in DATASETS:
        directory = Path(baseline_dirs[dataset]).resolve()
        if (directory / "run.lock").exists():
            raise ValueError("A selection baseline is still running")
        hashes = {name: file_sha256(directory / name) for name in BASELINE_FILES}
        signature = read_json(directory / "run.json")["signature"]
        rows = select_records(records, dataset, "feedback", metadata["protocol"]["seed"], 5)
        ids = [row["id"] for row in rows]
        if (len(rows) != 5 or signature.get("benchmark") != dataset or signature.get("role") != "feedback"
                or signature.get("limit") != 5 or signature.get("scope") != "subset"
                or signature.get("selected_ids") != ids or signature.get("protocol") != metadata["protocol"]
                or signature.get("dataset_metadata_sha256") != file_sha256(Path(data_dir) / "dataset.json")
                or recorded_source_hashes(signature).get("src/scoring.py") != file_sha256(ROOT / "src/scoring.py")):
            raise ValueError("Baseline IDs, feedback roles, dataset identity or scorer differ from the fixed five-question selection")
        previous = indexed_predictions(prediction_journal(directory / "predictions.jsonl"), ids)
        if len(previous) != 5 or any(row["status"] != "generated" for row in previous.values()):
            raise ValueError("Both original five-question feedback baselines must be complete")
        old_summary = summarize([{**previous[row["id"]], **score_answer(previous[row["id"]]["raw_response"], row["answer"])} for row in rows])
        if read_json(directory / "metrics.json")["summary"] != old_summary:
            raise ValueError("Original feedback metrics disagree with the saved predictions")
        resources = read_json(directory / "resources.json")
        if resources.get("worker_exit_code") != 0 or resources.get("limit_reached") is not None:
            raise ValueError("An original feedback worker did not finish cleanly")
        if any(file_sha256(directory / name) != expected for name, expected in hashes.items()):
            raise ValueError("A selection baseline changed during validation")
        evidence[dataset] = {"directory": str(directory), "file_sha256": hashes, "selected_ids": ids}
        selected.extend(rows)
    if len(selected) != 10 or len({row["id"] for row in selected}) != 10 or any(row["role"] != "feedback" for row in selected):
        raise ValueError("Exactly ten distinct feedback questions are required")
    return selected, evidence


def generation_request(image, question, profile_request):
    payload = deepcopy(profile_request)
    encoded = image_content(image)["image_url"]["url"].split(",", 1)[1]
    payload["messages"] = [{"role": "user", "content": question + "\n" + CONCISE_INSTRUCTION, "images": [encoded]}]
    return payload


def summarize_results(directory, selected, predictions, run, error=None):
    directory = Path(directory)
    indexed = indexed_predictions(predictions, [row["id"] for row in selected])
    scored = []
    for row in selected:
        prediction = indexed.get(row["id"], {"status": "missing"})
        result = {"id": row["id"], "dataset": row["dataset"], "question": row["question"],
                  "image_sha256": row["image_sha256"], "answer_type": row["answer_type"],
                  "reference": row["answer"], **prediction}
        if prediction["status"] == "generated":
            result.update(score_answer(prediction["raw_response"], row["answer"]))
        else:
            result.update(parsed_answer=None, parse_failed=False, strict_correct=False, normalized_correct=False, token_f1=0.0)
        scored.append(result)
    def metrics(rows):
        result = summarize(rows)
        complete = result["status"] == "complete"
        result["format_success_pct"] = 100 * (result["generated_count"] - result["parse_failure_count"]) / len(rows) if complete else None
        result["mean_request_seconds"] = result["generation_seconds"] / result["generated_count"] if result["generated_count"] else None
        return result
    report = {"experiment_id": EXPERIMENT_ID, "status": "complete" if len(predictions) == 10 and all(p["status"] == "generated" for p in predictions) and error is None else "incomplete",
              "purpose": "fixed feedback subset; exploratory baseline, not final benchmark evaluation",
              "output_folder": str(directory), "model": MODEL, "model_digest": run["model_digest"],
              "selected_count": 10, "generation_requests_started": run["generation_requests_started"],
              "scoring_id": SCORING_ID, "scorer_sha256": run["scorer_sha256"],
              "overall": metrics(scored), "datasets": {}, "training_examples_admitted": 0, "optimizer_steps": 0,
              "peak_ollama_ram_gib": None, "medical_validity": "not_independently_verified"}
    for dataset in DATASETS:
        rows = [r for r in scored if r["dataset"] == dataset]
        report["datasets"][dataset] = {"summary": metrics(rows), "by_answer_type": {
            kind: metrics([r for r in rows if r["answer_type"] == kind]) for kind in ("open", "closed")}}
    if error:
        report["error"] = error
    if report["status"] != "complete":
        groups = [report["overall"]]
        for value in report["datasets"].values():
            groups.extend([value["summary"], *value["by_answer_type"].values()])
        for group in groups:
            group.update(status="incomplete", strict_em_accuracy_pct=None, normalized_em_accuracy_pct=None,
                         mean_token_f1_pct=None, format_success_pct=None)
    with (directory / "scored.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(directory / "summary.json", report)
    lines = ["# 7B feedback evaluation", "", report["purpose"], "",
             "| Dataset | Generated / selected | Strict EM % | Normalized EM % | Token F1 % | Format % | Truncated |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    cell = lambda value: "incomplete" if value is None else f"{value:.1f}"
    for dataset, value in report["datasets"].items():
        m = value["summary"]
        lines.append(f"| {dataset} | {m['generated_count']} / {m['selected_count']} | {cell(m['strict_em_accuracy_pct'])} | {cell(m['normalized_em_accuracy_pct'])} | {cell(m['mean_token_f1_pct'])} | {cell(m['format_success_pct'])} | {m['token_limit_count']} |")
    lines += ["", f"Status: {report['status']}", "", "Five reused feedback questions per dataset do not establish full benchmark performance. The model, backend, token limit and prompt differ from the earlier small-model baseline; this is not a controlled model-size or MedEvo improvement comparison.",
              "", "The scorer is unchanged. Missing answer boxes are parse failures. Truncated responses remain recorded and are scored by the existing parser; truncation does not trigger another generation. Generated-only metrics on partial output must not be reported as complete-subset accuracy.",
              "", "No adapter training, candidate admission, auditor call or final-test inference is performed. Server allocation and available system RAM are not peak model RSS measurements."]
    if error:
        lines += ["", "Error: " + error]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_evaluation(root, data_dir, profile_dir, baseline_dirs, transport=local_json, available_ram=None):
    root, data_dir, profile_dir = Path(root), Path(data_dir).resolve(), Path(profile_dir).resolve()
    directory = root / "outputs/primary_feedback_10_v1"
    if directory.exists():
        raise ValueError(f"An attempt already exists: {directory}; inspect its summary before any repeat")
    profile, profile_request, profile_hashes = frozen_profile(profile_dir)
    metadata, records = load_bundle(data_dir, verify_images=True)
    if metadata["protocol"] != read_json(ROOT / "configs/benchmark_protocol.json"):
        raise ValueError("Prepared dataset protocol differs from the original fixed protocol")
    dataset_hash = file_sha256(data_dir / "dataset.json")
    selected, baselines = fixed_selection(data_dir, metadata, records, baseline_dirs)
    if available_ram is None:
        import psutil
        available_ram = lambda: psutil.virtual_memory().available / 1024**3
    ram_before = available_ram()
    if ram_before < 8:
        raise RuntimeError("The frozen 7B preflight guard requires at least 8 GiB available RAM")
    version = transport("/api/version").get("version")
    installed = transport("/api/tags").get("models", [])
    tag = next((m for m in installed if m.get("name") == MODEL or m.get("model") == MODEL), None)
    if not tag or tag.get("digest") != profile["model_digest"] or version != profile["ollama_version"]:
        raise ValueError("The installed model digest or server version differs from the frozen concise profile")
    directory.mkdir(parents=True)
    run = {"experiment_id": EXPERIMENT_ID, "started_utc": datetime.now(timezone.utc).isoformat(), "role": "feedback",
           "selected_ids": [r["id"] for r in selected], "model": MODEL, "model_digest": profile["model_digest"],
           "ollama_version": version, "prompt": CONCISE_INSTRUCTION, "options": OPTIONS,
           "profile_directory": str(profile_dir), "profile_sha256": profile_hashes, "selection_runs": baselines,
           "dataset_metadata_sha256": dataset_hash, "manifest_sha256": metadata["manifest_sha256"],
           "sources": metadata["sources"], "data_limitations": metadata.get("limitations", []),
           "script_sha256": file_sha256(__file__), "backend_script_sha256": file_sha256(ROOT / "check_primary.py"),
           "scorer_sha256": file_sha256(ROOT / "src/scoring.py"), "maximum_generation_requests": 10,
           "generation_requests_started": 0, "maximum_client_seconds": 1800, "peak_ollama_ram_gib": None}
    for label, source, names in [("profile", profile_dir, PROFILE_FILES)] + [("selection_" + d, Path(e["directory"]), BASELINE_FILES) for d, e in baselines.items()]:
        target = directory / label
        target.mkdir()
        expected = profile_hashes if label == "profile" else baselines[label.removeprefix("selection_")]["file_sha256"]
        for name in names:
            shutil.copy2(source / name, target / name)
            if file_sha256(target / name) != expected[name]:
                raise ValueError("An input snapshot changed while copying")
    write_json(directory / "run.json", run)
    write_json(directory / "environment.json", environment_snapshot(root))
    print(f"Feedback evaluation folder: {directory}", flush=True)
    predictions, error, loaded = [], None, None
    started = time.perf_counter()
    try:
        with (directory / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
            for index, row in enumerate(selected, 1):
                remaining = 1800 - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("The bounded evaluation reached its client time budget")
                if (tree_hashes(profile_dir) != profile_hashes or not same_file(data_dir / "dataset.json", dataset_hash)
                        or not same_file(ROOT / "src/scoring.py", run["scorer_sha256"])):
                    raise ValueError("Frozen profile or dataset metadata changed before generation")
                folder = directory / "cases" / f"{index:02d}"
                folder.mkdir(parents=True)
                path = image_path(data_dir, row["image_path"])
                if file_sha256(path) != row["image_file_sha256"]:
                    raise ValueError("A selected feedback image changed")
                with Image.open(path) as image:
                    payload = generation_request(image.convert("RGB"), row["question"], profile_request)
                write_json(folder / "request.json", payload)
                run["generation_requests_started"] += 1
                write_json(directory / "run.json", run)
                write_json(folder / "attempt.json", {"id": row["id"], "generation_requests_started": 1})
                request_started = time.perf_counter()
                try:
                    reply = transport("/api/chat", payload, timeout=min(600, remaining))
                    write_json(folder / "response.json", reply)
                    text = reply.get("message", {}).get("content")
                    if reply.get("model") != MODEL or reply.get("done") is not True or not isinstance(text, str):
                        raise ValueError("The selected model did not return a completed text response")
                    reason = reply.get("done_reason")
                    prediction = {"id": row["id"], "status": "generated", "raw_response": text,
                        "generated_tokens": reply.get("eval_count"), "done_reason": reason,
                        "hit_token_limit": reason == "length" or (reason not in ("stop", "length") and reply.get("eval_count", 0) >= 512),
                        "generation_seconds": round(time.perf_counter() - request_started, 3),
                        "provider_total_seconds": reply.get("total_duration", 0) / 1e9}
                except (Exception, KeyboardInterrupt) as failure:
                    error = safe_error(failure)
                    prediction = {"id": row["id"], "status": "generation_error", "error": error}
                predictions.append(prediction)
                journal.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(f"{index}/10 {row['id']}: {prediction['status']}", flush=True)
                if error:
                    break
        models = transport("/api/ps").get("models", [])
        write_json(directory / "loaded_models.json", {"models": models})
        loaded = next((m for m in models if m.get("name") == MODEL or m.get("model") == MODEL), None)
        if loaded and loaded.get("digest") and loaded["digest"] != profile["model_digest"]:
            raise ValueError("The loaded model digest differs from the frozen profile")
    except (Exception, KeyboardInterrupt) as failure:
        error = error or safe_error(failure)
    finally:
        run.update(elapsed_seconds=round(time.perf_counter() - started, 3), available_ram_before_gib=round(ram_before, 3),
                   available_ram_after_gib=round(available_ram(), 3), profile_preserved=tree_hashes(profile_dir) == profile_hashes)
        preserved = all(same_file(Path(e["directory"]) / name, digest) for e in baselines.values() for name, digest in e["file_sha256"].items())
        run["selection_runs_preserved"] = preserved
        data_preserved = same_file(data_dir / "dataset.json", dataset_hash) and all(
            same_file(data_dir / (role + ".jsonl"), digest) for role, digest in metadata["manifest_sha256"].items())
        controls_preserved = same_file(ROOT / "src/scoring.py", run["scorer_sha256"]) and same_file(__file__, run["script_sha256"])
        if not run["profile_preserved"] or not preserved or not data_preserved or not controls_preserved:
            error = "Frozen profile, selection run or dataset identity changed during evaluation"
        write_json(directory / "run.json", run)
        report = summarize_results(directory, selected, predictions, run, error)
        report.update(elapsed_seconds=run["elapsed_seconds"], available_ram_before_gib=run["available_ram_before_gib"],
                      available_ram_after_gib=run["available_ram_after_gib"], profile_preserved=run["profile_preserved"],
                      selection_runs_preserved=preserved, cpu_requested=True,
                      memory_note="System available RAM and server allocation are not a measured peak model RSS")
        if loaded:
            report.update(server_model_allocation_gib=round(loaded.get("size", 0) / 1024**3, 3),
                          server_vram_bytes=loaded.get("size_vram"), cpu_only_reported=loaded.get("size_vram") == 0)
        write_json(directory / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description="Run the approved fixed 10-question 7B feedback evaluation")
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--vqa-rad-run", type=Path, required=True)
    parser.add_argument("--slake-run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/benchmarks")
    args = parser.parse_args()
    try:
        report = run_evaluation(ROOT, args.data_dir, args.profile_dir, {"vqa_rad": args.vqa_rad_run, "slake": args.slake_run})
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "complete" else 2
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
