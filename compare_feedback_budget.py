"""Audit and run the expanded 1024-token feedback condition as separate stages."""

import argparse
import json
import os
from pathlib import Path
import shutil
import time

from PIL import Image

import evaluate_feedback as original
from feedback_scoring import SCORING_ID as FORMATTED_SCORER, score_formatted
from src.benchmarks import file_sha256, image_path, load_bundle
from src.evaluation import select_records
from src.runtime import environment_snapshot, safe_error, write_json
from src.scoring import score_answer
from src.training_collection import read_json


ROOT = Path(__file__).resolve().parent
EXPERIMENT = "qwen35-feedback-119-budget-v1"
OPTIONS = {**original.OPTIONS, "num_predict": 1024}
TOTAL_SECONDS = 10800
REQUEST_SECONDS = 300
PLAN_FOLDER = "qwen35_feedback_119_plan_v1"
RUN_FOLDER = "qwen35_feedback_119_1024_v1"
BASELINE_FILES = ("run.json", "summary.json", "selection.json", "predictions.jsonl", "scored.jsonl", "model_details.json")


def read_lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def unchanged(root, hashes):
    return all(file_sha256(Path(root) / name) == digest for name, digest in hashes.items())


def audit_inputs(data_dir, baseline_dir):
    """Offline validation only. No transport/model call is made here."""
    data_dir, baseline_dir = Path(data_dir).resolve(), Path(baseline_dir).resolve()
    metadata, records = load_bundle(data_dir, verify_images=True)
    baseline = read_json(baseline_dir / "run.json")
    summary = read_json(baseline_dir / "summary.json")
    if (baseline.get("experiment_id") != original.EXPERIMENT or summary.get("status") != "complete"
            or baseline.get("role") != "feedback" or baseline.get("model") != original.MODEL
            or baseline.get("prompt") != original.PROMPT or baseline.get("think") is not False
            or baseline.get("options") != original.OPTIONS or baseline.get("generation_requests_started") != 20
            or summary.get("model_digest") != baseline.get("model_digest")):
        raise ValueError("Use the completed original twenty-question 512-token feedback baseline")
    if not unchanged(ROOT, baseline["code_sha256"]) or not unchanged(data_dir, baseline["dataset_file_sha256"]):
        raise ValueError("Original evaluator, source, scorer or prepared data changed since the baseline")
    if metadata["protocol"] != read_json(ROOT / "configs/benchmark_protocol.json"):
        raise ValueError("Prepared protocol changed")
    selected = select_records(records, "vqa_rad", "feedback", 7)
    if len(selected) != 79:
        raise ValueError("The frozen design requires all 79 VQA-RAD feedback questions")
    slake = select_records(records, "slake", "feedback", 7, 40)
    if len(slake) != 40:
        raise ValueError("Need forty SLAKE feedback questions")
    selected += slake
    ids = [r["id"] for r in selected]
    feedback_ids = {r["id"] for r in records if r["role"] == "feedback"}
    other_ids = {r["id"] for r in records if r["role"] in ("train", "test")}
    if len(set(ids)) != 119 or not set(ids) <= feedback_ids or set(ids) & other_ids:
        raise ValueError("Selected IDs must belong exclusively to the feedback manifest")
    expected_old = original.select_feedback(metadata, records)
    if baseline["selected_ids"] != [r["id"] for r in expected_old] or read_json(baseline_dir / "selection.json") != expected_old:
        raise ValueError("Earlier twenty-question selection differs from the fixed manifest selection")
    previous = read_lines(baseline_dir / "predictions.jsonl")
    previous_scored = read_lines(baseline_dir / "scored.jsonl")
    if [p["id"] for p in previous] != baseline["selected_ids"] or len(previous_scored) != 20:
        raise ValueError("Original baseline predictions are missing, reordered or duplicated")
    for row, prediction, scored in zip(expected_old, previous, previous_scored):
        if prediction.get("status") != "generated" or any(scored.get(k) != v for k, v in prediction.items()):
            raise ValueError("Original scored records do not preserve the raw predictions")
        if scored.get("reference") != row["answer"] or scored.get("question") != row["question"]:
            raise ValueError("Original references/questions differ from the manifest")
        if any(scored.get(k) != v for k, v in score_answer(prediction["raw_response"], row["answer"]).items()):
            raise ValueError("Original saved scores do not reproduce")
    if sum(not p["hit_token_limit"] for p in previous) != 16:
        raise ValueError("Expected the sixteen originally nontruncated outputs")
    if not set(baseline["selected_ids"]) <= set(ids):
        raise ValueError("The expanded selection must include all earlier twenty questions")
    hashes = {name: file_sha256(baseline_dir / name) for name in BASELINE_FILES}
    for index, row in enumerate(expected_old, 1):
        with Image.open(image_path(data_dir, row["image_path"])) as image:
            expected_request = original.request_payload(image.convert("RGB"), row["question"])
        request_name = f"cases/{index:02d}/request.json"
        response_name = f"cases/{index:02d}/response.json"
        if read_json(baseline_dir / request_name) != expected_request:
            raise ValueError("A previous request differs from the frozen prompt, image or controls")
        response = read_json(baseline_dir / response_name)
        if response.get("message", {}).get("content") != previous[index - 1]["raw_response"]:
            raise ValueError("A previous raw response differs from its prediction")
        for name in (request_name, response_name): hashes[name] = file_sha256(baseline_dir / name)
    code = {**baseline["code_sha256"], "compare_feedback_budget.py": file_sha256(__file__),
            "feedback_scoring.py": file_sha256(ROOT / "feedback_scoring.py")}
    plan = {"experiment_id": EXPERIMENT, "data_directory": str(data_dir), "baseline_directory": str(baseline_dir),
            "selected_ids": ids, "counts": {"vqa_rad": 79, "slake": 40}, "role": "feedback",
            "all_ids_in_feedback_manifest": True, "training_role_overlap": 0, "test_role_overlap": 0,
            "earlier_twenty_included": True, "selected_count": 119,
            "model": original.MODEL, "model_digest": baseline["model_digest"], "ollama_version": baseline["ollama_version"],
            "prompt": original.PROMPT, "think": False, "options": OPTIONS,
            "original_options": original.OPTIONS, "changed_generation_field": "options.num_predict",
            "baseline_file_sha256": hashes, "dataset_file_sha256": baseline["dataset_file_sha256"],
            "code_sha256": code, "original_scorer": original.SCORING_ID, "companion_scorer": FORMATTED_SCORER,
            "decision_rule": {"denominator": 119, "maximum_missing_boxes": 11,
                              "pass": "adopt paper prompt with 1024-token cap",
                              "fail": "retain paper prompt with 512-token baseline; concise condition remains separate and unrun"},
            "maximum_generation_requests": 119, "request_timeout_seconds": REQUEST_SECONDS,
            "maximum_client_seconds": TOTAL_SECONDS}
    return plan, selected, previous


def inspect_selection(root, data_dir, baseline_dir):
    plan, selected, _ = audit_inputs(data_dir, baseline_dir)
    directory = Path(root) / "outputs" / PLAN_FOLDER
    if directory.exists():
        if read_json(directory / "plan.json") != plan or read_json(directory / "selection.json") != selected:
            raise ValueError("Existing plan differs; preserve it and inspect the changes")
    else:
        directory.mkdir(parents=True)
        write_json(directory / "plan.json", plan)
        write_json(directory / "selection.json", selected)
    return {"status": "passed", "plan_folder": str(directory), "selected_count": 119, "counts": plan["counts"],
            "all_ids_in_feedback_manifest": True, "training_role_overlap": 0, "test_role_overlap": 0,
            "earlier_twenty_included": True, "new_generation_calls": 0,
            "runtime_estimate_minutes": [80, 160], "client_budget_minutes": 180,
            "next_action": "review selection audit and obtain approval before running"}


def metrics(rows, scorer, valid):
    scored = []
    for row in rows:
        item = dict(row)
        item.update(scorer(row["raw_response"], row["reference"]) if row["status"] == "generated" else {
            "parsed_answer": None, "parse_failed": False, "strict_correct": False, "normalized_correct": False, "token_f1": 0.0})
        scored.append(item)
    return original.group_metrics(scored, valid), scored


def save_results(directory, selected, predictions, previous, plan, run, error):
    indexed = {r["id"]: r for r in predictions}
    rows = [{"id": r["id"], "dataset": r["dataset"], "question": r["question"], "reference": r["answer"],
             "answer_type": r["answer_type"], **indexed.get(r["id"], {"status": "missing"})} for r in selected]
    complete = error is None and len(predictions) == 119 and all(p["status"] == "generated" for p in predictions)
    result = {"experiment_id": EXPERIMENT, "status": "complete" if complete else "incomplete",
              "selected_count": 119, "generation_requests_started": run["generation_requests_started"],
              "elapsed_seconds": run["elapsed_seconds"], "model_digest": plan["model_digest"],
              "scorers": {}, "training_examples_admitted": 0, "optimizer_steps": 0, "test_generation_requests": 0,
              "peak_ollama_ram_gib": None, "output_folder": str(directory)}
    scored_versions = {}
    for label, scorer in ((original.SCORING_ID, score_answer), (FORMATTED_SCORER, score_formatted)):
        overall, scored = metrics(rows, scorer, error is None)
        scored_versions[label] = scored
        groups = {"overall": overall, "datasets": {}}
        for dataset in original.DATASETS:
            part = [r for r in rows if r["dataset"] == dataset]
            groups["datasets"][dataset] = {"summary": metrics(part, scorer, error is None)[0], "by_answer_type": {
                kind: metrics([r for r in part if r["answer_type"] == kind], scorer, error is None)[0]
                for kind in ("open", "closed")}}
        result["scorers"][label] = groups
    comparisons = []
    references = {r["id"]: r["answer"] for r in selected}
    for old in previous:
        new = indexed.get(old["id"], {})
        successful = new.get("status") == "generated"
        item = {"id": old["id"], "originally_nontruncated": not old["hit_token_limit"],
                "trial_available": successful, "raw_text_identical": old["raw_response"] == new.get("raw_response") if successful else None,
                "old_raw_response": old["raw_response"], "trial_raw_response": new.get("raw_response")}
        for label, scorer in ((original.SCORING_ID, score_answer), (FORMATTED_SCORER, score_formatted)):
            old_score = scorer(old["raw_response"], references[old["id"]])
            new_score = scorer(new["raw_response"], references[old["id"]]) if successful else None
            item[label] = {"old_score": old_score, "trial_score": new_score,
                           "parsed_answer_identical": old_score["parsed_answer"] == new_score["parsed_answer"] if successful else None}
        comparisons.append(item)
    nontruncated = [r for r in comparisons if r["originally_nontruncated"]]
    result["earlier_sixteen_reproduction"] = {"expected_count": 16, "available_count": sum(r["trial_available"] for r in nontruncated),
                                            "raw_text_identical_count": sum(r["raw_text_identical"] is True for r in nontruncated),
                                            "all_raw_text_identical": all(r["raw_text_identical"] is True for r in nontruncated)}
    result["earlier_sixteen_reproduction"]["parsed_answer_identical_counts"] = {
        label: sum(r[label]["parsed_answer_identical"] is True for r in nontruncated)
        for label in (original.SCORING_ID, FORMATTED_SCORER)}
    missing_boxes = result["scorers"][FORMATTED_SCORER]["overall"]["parse_failure_count"]
    result["decision"] = {"rule": plan["decision_rule"], "missing_boxes": missing_boxes,
                          "missing_box_pct": 100 * missing_boxes / 119 if complete else None,
                          "outcome": ("adopt_1024" if missing_boxes <= 11 else "retain_512") if complete else "undecided_incomplete"}
    if error:
        result["error"] = error
    for label, items in scored_versions.items():
        with (directory / ("scored_original.jsonl" if label == original.SCORING_ID else "scored_unwrapped.jsonl")).open("w", encoding="utf-8") as stream:
            for row in items: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(directory / "paired_earlier_twenty.json", comparisons)
    write_json(directory / "summary.json", result)
    lines = ["# Expanded feedback budget experiment", "", "Exploratory feedback results; no test generation or training.", "",
             "| Scorer | Dataset | Generated | Accuracy % | Boxed % | Truncated | Mean seconds | Mean tokens |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda x: "unavailable" if x is None else f"{x:.2f}"
    for label, groups in result["scorers"].items():
        for dataset, data in groups["datasets"].items():
            m = data["summary"]
            lines.append(f"| {label} | {dataset} | {m['generated_count']}/{m['selected_count']} | {fmt(m['strict_em_accuracy_pct'])} | "
                         f"{fmt(m['boxed_answer_rate_pct'])} | {m['token_limit_count']} | "
                         f"{fmt(m['request_seconds']['mean'] if m['request_seconds'] else None)} | {fmt(m['generated_tokens']['mean'] if m['generated_tokens'] else None)} |")
    lines += ["", "Decision: " + result["decision"]["outcome"], "", "Reproduction of earlier sixteen: " + str(result["earlier_sixteen_reproduction"]),
              "", "Full 119-question scores are not a paired improvement over the earlier twenty. Paired records separately rescore both budgets under both scorer versions. "
              "Exact text reproduction is measured, not guaranteed by temperature zero. Missing-box threshold concerns format, not medical competence. "
              "A concise-prompt condition is not run or selected by this script."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run_comparison(root, plan_dir, transport=original.local_json, available_ram=None):
    root, plan_dir = Path(root), Path(plan_dir).resolve()
    directory = root / "outputs" / RUN_FOLDER
    if directory.exists():
        raise ValueError("An expanded attempt already exists; inspect it before any repeat")
    plan = read_json(plan_dir / "plan.json")
    fresh, selected, previous = audit_inputs(plan["data_directory"], plan["baseline_directory"])
    if plan != fresh or read_json(plan_dir / "selection.json") != selected:
        raise ValueError("Plan or selection changed since the offline audit")
    if available_ram is None:
        import psutil
        available_ram = lambda: psutil.virtual_memory().available / 1024**3
    if available_ram() < 10:
        raise RuntimeError("Need 10 GiB available RAM before this run; this is not a fit guarantee")
    def server_identity():
        tags = transport("/api/tags").get("models", [])
        if transport("/api/version") != plan["ollama_version"] or not any(
                t.get("digest") == plan["model_digest"] and (t.get("name") == original.MODEL or t.get("model") == original.MODEL) for t in tags):
            raise ValueError("Ollama version or model digest differs from the original baseline")
    server_identity()
    details = transport("/api/show", {"model": original.MODEL})
    if "vision" not in details.get("capabilities", []) or not any(v is False for v in details.get("thinking", {}).get("values", [])):
        raise ValueError("Vision and think=false support must be confirmed")
    directory.mkdir()
    shutil.copytree(plan_dir, directory / "selection_audit")
    if read_json(directory / "selection_audit/plan.json") != plan or read_json(directory / "selection_audit/selection.json") != selected:
        raise ValueError("Selection plan changed while copying its snapshot")
    (directory / "baseline").mkdir()
    for name in BASELINE_FILES:
        shutil.copy2(Path(plan["baseline_directory"]) / name, directory / "baseline" / name)
        if file_sha256(directory / "baseline" / name) != plan["baseline_file_sha256"][name]:
            raise ValueError("Original baseline changed while copying its snapshot")
    write_json(directory / "environment.json", environment_snapshot(root))
    write_json(directory / "model_details.json", details)
    run = {"plan": plan, "generation_requests_started": 0, "available_ram_before_gib": available_ram()}
    write_json(directory / "run.json", run)
    print(f"Expanded feedback folder: {directory}", flush=True)
    predictions, error = [], None
    started = time.perf_counter()
    try:
        with (directory / "predictions.jsonl").open("x", encoding="utf-8") as journal:
            for index, row in enumerate(selected, 1):
                remaining = TOTAL_SECONDS - (time.perf_counter() - started)
                if remaining <= 0: raise TimeoutError("Three-hour client budget reached")
                if available_ram() < 4: raise RuntimeError("Available RAM fell below 4 GiB between requests")
                if not unchanged(ROOT, plan["code_sha256"]) or not unchanged(plan["data_directory"], plan["dataset_file_sha256"]) or not unchanged(plan["baseline_directory"], plan["baseline_file_sha256"]):
                    raise ValueError("Code, data or original baseline changed")
                path = image_path(Path(plan["data_directory"]), row["image_path"])
                if file_sha256(path) != row["image_file_sha256"]: raise ValueError("Selected image changed")
                with Image.open(path) as image:
                    payload = original.request_payload(image.convert("RGB"), row["question"])
                payload["options"] = dict(OPTIONS)
                case = directory / "cases" / f"{index:03d}"
                case.mkdir(parents=True)
                write_json(case / "request.json", payload)
                run["generation_requests_started"] += 1
                write_json(directory / "run.json", run)
                sent = time.perf_counter()
                try:
                    reply = transport("/api/chat", payload, timeout=min(REQUEST_SECONDS, remaining))
                    write_json(case / "response.json", reply)
                    prediction = original.response_prediction(reply, row["id"], time.perf_counter() - sent)
                    prediction["hit_token_limit"] = reply["done_reason"] == "length" or reply["eval_count"] >= 1024
                except (Exception, KeyboardInterrupt) as failure:
                    error = safe_error(failure)
                    prediction = {"id": row["id"], "status": "generation_error", "error": error}
                predictions.append(prediction)
                journal.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush(); os.fsync(journal.fileno())
                print(f"{index}/119 {row['id']}: {prediction['status']}", flush=True)
                if time.perf_counter() - started > TOTAL_SECONDS: error = error or "TimeoutError: Client budget exceeded"
                if error: break
        server_identity()
        if not unchanged(ROOT, plan["code_sha256"]) or not unchanged(plan["data_directory"], plan["dataset_file_sha256"]) or not unchanged(plan["baseline_directory"], plan["baseline_file_sha256"]):
            raise ValueError("Code, data or baseline changed before reporting")
        write_json(directory / "server_allocation.json", transport("/api/ps"))
    except (Exception, KeyboardInterrupt) as failure:
        error = error or safe_error(failure)
    run["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    run["available_ram_after_gib"] = available_ram()
    write_json(directory / "run.json", run)
    return save_results(directory, selected, predictions, previous, plan, run, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Offline selection audit; zero generation calls")
    inspect.add_argument("--data-dir", type=Path, default=ROOT / "data/benchmarks")
    inspect.add_argument("--baseline-dir", type=Path, default=ROOT / "outputs/qwen35_feedback_20_v1")
    run = commands.add_parser("run", help="Execute the reviewed plan; requires separate user approval")
    run.add_argument("--plan-dir", type=Path, default=ROOT / "outputs" / PLAN_FOLDER)
    args = parser.parse_args()
    try:
        result = inspect_selection(ROOT, args.data_dir, args.baseline_dir) if args.command == "inspect" else run_comparison(ROOT, args.plan_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] in ("passed", "complete") else 2
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "blocked", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
