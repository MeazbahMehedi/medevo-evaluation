"""Run a fixed, reference-free Qwen3.5 feedback baseline through local Ollama."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time

from PIL import Image

from check_primary import local_json
from src.benchmarks import file_sha256, image_path, load_bundle
from src.evaluation import select_records, source_fingerprints
from src.repair_audit import image_content
from src.runtime import environment_snapshot, safe_error, write_json
from src.scoring import SCORING_ID, score_answer, summarize
from src.training_collection import read_json


ROOT = Path(__file__).resolve().parent
MODEL = "qwen3.5:9b"
EXPERIMENT = "qwen35-feedback-20-v1"
PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
DATASETS = ("vqa_rad", "slake")
OPTIONS = {"temperature": 0, "seed": 7, "num_predict": 512, "num_ctx": 4096,
           "num_thread": 4, "num_gpu": 0, "presence_penalty": 1.5, "top_k": 20, "top_p": 0.95}
TOTAL_SECONDS = 1800
REQUEST_SECONDS = 300


def select_feedback(metadata, records):
    if metadata["protocol"] != read_json(ROOT / "configs/benchmark_protocol.json"):
        raise ValueError("Prepared bundle differs from the original dataset protocol")
    selected = []
    for dataset in DATASETS:
        rows = select_records(records, dataset, "feedback", 7, 10)
        if len(rows) != 10:
            raise ValueError(f"Need ten eligible feedback questions from {dataset}")
        selected.extend(rows)
    if len({r["id"] for r in selected}) != 20 or any(r["role"] != "feedback" for r in selected):
        raise ValueError("Only twenty distinct feedback records are permitted")
    return selected


def request_payload(image, question):
    encoded = image_content(image)["image_url"]["url"].split(",", 1)[1]
    return {"model": MODEL, "messages": [{"role": "user", "content": question + "\n" + PROMPT,
                                          "images": [encoded]}],
            "think": False, "stream": False, "keep_alive": "2m", "options": dict(OPTIONS)}


def response_prediction(reply, identity, seconds):
    message = reply.get("message", {})
    text = message.get("content")
    thinking = message.get("thinking", "")
    count = reply.get("eval_count")
    if (reply.get("model") != MODEL or reply.get("done") is not True or not isinstance(text, str)
            or reply.get("done_reason") not in ("stop", "length")
            or type(count) is not int or count < 0 or not isinstance(thinking, str)):
        raise ValueError("Expected a completed response with token count and a known stop reason")
    if thinking.strip() or "<think>" in text.lower() or "</think>" in text.lower():
        raise ValueError("Thinking output appeared despite think=false; stop and inspect the saved response")
    return {"id": identity, "status": "generated", "raw_response": text,
            "generated_tokens": count, "answer_characters": len(text), "answer_words": len(text.split()),
            "generation_seconds": round(seconds, 3), "done_reason": reply["done_reason"],
            "hit_token_limit": reply["done_reason"] == "length" or count >= OPTIONS["num_predict"],
            "thinking_characters": len(thinking),
            "provider_total_seconds": reply.get("total_duration", 0) / 1e9,
            "provider_load_seconds": reply.get("load_duration", 0) / 1e9,
            "provider_generation_seconds": reply.get("eval_duration", 0) / 1e9}


def group_metrics(rows, valid=True):
    result = summarize(rows)
    generated = [r for r in rows if r["status"] == "generated"]
    parsed = sum(not r["parse_failed"] for r in generated)
    complete = result["status"] == "complete" and valid
    result["boxed_answer_rate_pct"] = 100 * parsed / len(rows) if complete and rows else None
    result["generated_only_boxed_rate_pct"] = 100 * parsed / len(generated) if generated else None
    result["latency_scope"] = "client round trip including load, image processing and generation"
    for key, field in (("request_seconds", "generation_seconds"), ("generated_tokens", "generated_tokens"),
                       ("answer_characters", "answer_characters"), ("answer_words", "answer_words")):
        values = [r[field] for r in generated]
        result[key] = {"mean": statistics.mean(values), "median": statistics.median(values),
                       "min": min(values), "max": max(values)} if values else None
    if not complete:
        result.update(status="incomplete", strict_em_accuracy_pct=None,
                      normalized_em_accuracy_pct=None, mean_token_f1_pct=None)
    return result


def save_report(directory, selected, predictions, run, error):
    indexed = {p["id"]: p for p in predictions}
    scored = []
    for row in selected:
        prediction = indexed.get(row["id"], {"status": "missing"})
        item = {"id": row["id"], "dataset": row["dataset"], "question": row["question"],
                "reference": row["answer"], "answer_type": row["answer_type"], **prediction}
        item.update(score_answer(prediction["raw_response"], row["answer"]) if prediction["status"] == "generated"
                    else {"parsed_answer": None, "parse_failed": False, "strict_correct": False,
                          "normalized_correct": False, "token_f1": 0.0})
        scored.append(item)
    complete = error is None and len(predictions) == 20 and all(p["status"] == "generated" for p in predictions)
    result = {"experiment_id": EXPERIMENT, "status": "complete" if complete else "incomplete",
              "purpose": "exploratory fixed feedback subset; not final test evaluation or MedEvo improvement",
              "model": MODEL, "model_digest": run["model_digest"], "selected_count": 20,
              "generation_requests_started": run["generation_requests_started"],
              "overall": group_metrics(scored, error is None), "datasets": {},
              "scoring_id": SCORING_ID, "scorer_sha256": run["scorer_sha256"],
              "elapsed_seconds": run["elapsed_seconds"], "output_folder": str(directory),
              "peak_ollama_ram_gib": None, "training_examples_admitted": 0, "optimizer_steps": 0,
              "test_generation_requests": 0, "medical_validity": "not_independently_verified"}
    for dataset in DATASETS:
        rows = [r for r in scored if r["dataset"] == dataset]
        result["datasets"][dataset] = {"summary": group_metrics(rows, error is None), "by_answer_type": {
            kind: group_metrics([r for r in rows if r["answer_type"] == kind], error is None)
            for kind in ("open", "closed")}}
    if error:
        result["error"] = error
    with (directory / "scored.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for row in scored:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(directory / "summary.json", result)
    lines = ["# Qwen3.5 feedback baseline", "", result["purpose"], "",
             "| Dataset | Generated/selected | Strict EM % | Boxed % | Mean seconds/question | Mean tokens | Truncated |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    cell = lambda x: "unavailable" if x is None else f"{x:.2f}"
    for dataset in DATASETS:
        m = result["datasets"][dataset]["summary"]
        lines.append(f"| {dataset} | {m['generated_count']}/{m['selected_count']} | {cell(m['strict_em_accuracy_pct'])} | "
                     f"{cell(m['boxed_answer_rate_pct'])} | {cell(m['request_seconds']['mean'] if m['request_seconds'] else None)} | "
                     f"{cell(m['generated_tokens']['mean'] if m['generated_tokens'] else None)} | {m['token_limit_count']} |")
    lines += ["", f"Status: {result['status']}", "",
              "The paper's Appendix D.1 instruction is retained. Quantized CPU execution, think=false, temperature zero, "
              "512 output tokens and reduced image/context limits are study adaptations, not verified paper execution settings.",
              "", "Selections are deterministic feedback subsets and may include previously inspected questions. "
              "This is not a controlled comparison with earlier models or an unbiased full-benchmark estimate.",
              "", "Raw stop reasons, empty replies and malformed answer boxes are retained. No retries, repair, "
              "training or test generation occur. RAM is not measured as a model peak; do not substitute server allocation."]
    if error:
        lines += ["", "Stopped: " + error]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def run_feedback(root, data_dir, transport=local_json, available_ram=None):
    root, data_dir = Path(root), Path(data_dir).resolve()
    directory = root / "outputs/qwen35_feedback_20_v1"
    if directory.exists():
        raise ValueError(f"An attempt already exists: {directory}; inspect it before any repeat")
    metadata, records = load_bundle(data_dir, verify_images=True)
    selected = select_feedback(metadata, records)
    if available_ram is None:
        import psutil
        available_ram = lambda: psutil.virtual_memory().available / 1024**3
    ram_before = available_ram()
    if ram_before < 10:
        raise RuntimeError("Need at least 10 GiB available RAM for this preflight guard; this is not a fit guarantee")
    version = transport("/api/version")
    tags = transport("/api/tags").get("models", [])
    tag = next((m for m in tags if m.get("name") == MODEL or m.get("model") == MODEL), None)
    if not tag or not isinstance(tag.get("digest"), str) or not tag["digest"]:
        raise ValueError("The approved model must already be installed; this script does not pull weights")
    details = transport("/api/show", {"model": MODEL})
    if "vision" not in details.get("capabilities", []) or not any(v is False for v in details.get("thinking", {}).get("values", [])):
        raise ValueError("Model metadata must confirm vision and think=false support; inspect /api/show")
    (root / "outputs").mkdir(exist_ok=True)
    directory.mkdir()
    code = {**source_fingerprints(ROOT), "evaluate_feedback.py": file_sha256(__file__),
            "check_primary.py": file_sha256(ROOT / "check_primary.py")}
    data_hashes = {name: file_sha256(data_dir / name) for name in
                   ("dataset.json", "train.jsonl", "feedback.jsonl", "test.jsonl", "exclusions.jsonl")}
    run = {"experiment_id": EXPERIMENT, "started_utc": datetime.now(timezone.utc).isoformat(),
           "role": "feedback", "selected_ids": [r["id"] for r in selected], "model": MODEL,
           "model_digest": tag["digest"], "ollama_version": version, "prompt": PROMPT,
           "prompt_source": "MedEvo-Ant2026.pdf Appendix D.1, page 29", "think": False, "options": OPTIONS,
           "sources": metadata["sources"], "manifest_sha256": metadata["manifest_sha256"],
           "data_limitations": metadata.get("limitations", []), "dataset_file_sha256": data_hashes,
           "code_sha256": code, "scorer_sha256": file_sha256(ROOT / "src/scoring.py"),
           "maximum_generation_requests": 20, "generation_requests_started": 0,
           "request_timeout_seconds": REQUEST_SECONDS, "maximum_client_seconds": TOTAL_SECONDS,
           "available_ram_before_gib": ram_before, "peak_ollama_ram_gib": None}
    write_json(directory / "run.json", run)
    write_json(directory / "model_details.json", details)
    write_json(directory / "environment.json", environment_snapshot(root))
    write_json(directory / "selection.json", selected)
    print(f"Feedback output folder: {directory}", flush=True)
    predictions, error = [], None
    started = time.perf_counter()
    try:
        with (directory / "predictions.jsonl").open("x", encoding="utf-8", newline="\n") as journal:
            for index, row in enumerate(selected, 1):
                remaining = TOTAL_SECONDS - (time.perf_counter() - started)
                if remaining <= 0:
                    raise TimeoutError("Total client budget reached")
                if any(file_sha256(ROOT / name) != digest for name, digest in code.items()):
                    raise ValueError("Source changed during the experiment")
                if any(file_sha256(data_dir / name) != digest for name, digest in data_hashes.items()):
                    raise ValueError("Prepared bundle changed during the experiment")
                if available_ram() < 4:
                    raise RuntimeError("Available system RAM fell below the 4 GiB stop threshold")
                path = image_path(data_dir, row["image_path"])
                if file_sha256(path) != row["image_file_sha256"]:
                    raise ValueError("Selected image changed")
                with Image.open(path) as image:
                    payload = request_payload(image.convert("RGB"), row["question"])
                case = directory / "cases" / f"{index:02d}"
                case.mkdir(parents=True)
                write_json(case / "request.json", payload)
                run["generation_requests_started"] += 1
                write_json(directory / "run.json", run)
                request_started = time.perf_counter()
                try:
                    reply = transport("/api/chat", payload, timeout=min(REQUEST_SECONDS, remaining))
                    write_json(case / "response.json", reply)
                    prediction = response_prediction(reply, row["id"], time.perf_counter() - request_started)
                except (Exception, KeyboardInterrupt) as failure:
                    error = safe_error(failure)
                    prediction = {"id": row["id"], "status": "generation_error", "error": error,
                                  "generation_seconds": round(time.perf_counter() - request_started, 3)}
                predictions.append(prediction)
                journal.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(f"{index}/20 {row['id']}: {prediction['status']}", flush=True)
                if time.perf_counter() - started > TOTAL_SECONDS:
                    error = error or "TimeoutError: Total client budget exceeded; inspect the saved response"
                if error:
                    break
        # Detect model replacement or data/code changes without another generation.
        after = transport("/api/tags").get("models", [])
        if not any(m.get("digest") == run["model_digest"] and (m.get("name") == MODEL or m.get("model") == MODEL) for m in after):
            raise ValueError("Model identity changed; results require review")
        if transport("/api/version") != version:
            raise ValueError("Ollama version changed during the run")
        if any(file_sha256(ROOT / name) != digest for name, digest in code.items()) or any(
                file_sha256(data_dir / name) != digest for name, digest in data_hashes.items()):
            raise ValueError("Code or data changed before reporting")
        write_json(directory / "server_allocation.json", transport("/api/ps"))
    except (Exception, KeyboardInterrupt) as failure:
        error = error or safe_error(failure)
    run["elapsed_seconds"] = round(time.perf_counter() - started, 3)
    run["available_ram_after_gib"] = available_ram()
    write_json(directory / "run.json", run)
    return save_report(directory, selected, predictions, run, error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/benchmarks")
    args = parser.parse_args()
    try:
        result = run_feedback(ROOT, args.data_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "complete" else 2
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "blocked", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
