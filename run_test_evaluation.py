"""Evaluate one system on the frozen reserved test subset (or a small feedback check).

Systems
  baseline   Qwen3.5-9B via Ollama (paper base model, quantized)
  primary    Qwen3.5-2B in Transformers, unadapted
  adapted    Qwen3.5-2B with the trained LoRA adapter

Examples
  python run_test_evaluation.py --system baseline
  python run_test_evaluation.py --system primary --feedback 20
  python run_test_evaluation.py --system adapted --adapter-dir outputs/adapter_v1/adapter

Runs are resumable: re-run the same command after an interruption.
"""

import argparse
import json
from pathlib import Path
import sys
import time

from medevo_study.core import (DATA_DIR, OUTPUTS, PRIMARY_FIELD, PRIMARY_SCORER, PROTOCOL_DIR, append_jsonl,
                               code_hashes, data_hashes, file_sha256, grouped_summary, load_image, load_records,
                               read_json, read_jsonl, require_unchanged, safe_error, score_all, stratified_sample,
                               utc_now, write_json, write_jsonl)

FOLDERS = {"baseline": "qwen35_9b", "primary": "qwen35_2b_base", "adapted": "qwen35_2b_adapted"}
MAX_CONSECUTIVE_ERRORS = 3


def select(records, protocol, feedback_size):
    by_id = {r["id"]: r for r in records}
    if feedback_size:
        rows = []
        for dataset in ("vqa_rad", "slake"):
            rows += stratified_sample(records, "feedback", dataset, feedback_size // 2, 7, "feedback-check-v1")
        return rows
    return [by_id[i] for i in protocol["test_ids"]]


def make_backend(system, protocol, adapter_dir):
    budget = protocol["output_token_budget"]
    if system == "baseline":
        from medevo_study import ollama_backend
        identity = ollama_backend.server_identity()
        expected = protocol["baseline_model"]["model_digest"]
        if identity["model_digest"] != expected:
            raise ValueError("Installed qwen3.5:9b digest differs from the frozen protocol")

        def run(record):
            payload, reply, prediction = ollama_backend.answer(DATA_DIR, record, budget)
            return prediction
        return run, identity

    from medevo_study import transformers_backend as backend
    settings = read_json(PROTOCOL_DIR / "primary_model.json")
    if system == "adapted":
        if adapter_dir is None:
            raise ValueError("--adapter-dir is required for the adapted system")
        adapter_dir = Path(adapter_dir).resolve()
        if not (adapter_dir / "adapter_config.json").exists():
            raise ValueError(f"No adapter found in {adapter_dir}")
    model, processor, info = backend.load(settings["model_id"], settings["revision"], settings["dtype"],
                                          settings["threads"], adapter_dir if system == "adapted" else None)
    if system == "adapted":
        info["adapter_sha256"] = file_sha256(adapter_dir / "adapter_model.safetensors")
    user_suffix = "\n" + protocol["prompt"]

    def run(record):
        image = load_image(DATA_DIR, record)
        result = backend.generate(model, processor, image, record["question"] + user_suffix, budget)
        result.pop("generated_token_ids")
        return {"id": record["id"], "status": "generated", **result}
    return run, {**info, "settings": settings}


def scored_rows(selected, predictions):
    latest = {p["id"]: p for p in predictions}
    rows = []
    for record in selected:
        prediction = latest.get(record["id"], {"status": "missing"})
        row = {"id": record["id"], "dataset": record["dataset"], "answer_type": record["answer_type"],
               "question": record["question"], "reference": record["answer"], **prediction}
        if prediction.get("status") == "generated":
            row["scores"] = score_all(prediction["raw_response"], record["answer"])
        rows.append(row)
    return rows


def write_outputs(directory, run, selected, predictions):
    rows = scored_rows(selected, predictions)
    write_jsonl(directory / "scored.jsonl", rows)
    summary = {"system": run["system"], "role": run["role"], "status": "complete" if all(
        r.get("status") == "generated" for r in rows) else "incomplete",
        "primary_metric": f"{PRIMARY_SCORER}:{PRIMARY_FIELD}", "metrics": grouped_summary(rows),
        "updated_utc": utc_now(), "folder": str(directory)}
    write_json(directory / "summary.json", summary)
    main = summary["metrics"][f"{PRIMARY_SCORER}:{PRIMARY_FIELD}"]
    lines = [f"# {run['system']} on {run['role']} questions", "", f"Status: {summary['status']}", "",
             "| Dataset | Type | Generated | Accuracy % | 95% CI | Boxed % | Truncated | s/question |",
             "|---|---|---:|---:|---|---:|---:|---:|"]
    f = lambda x: "-" if x is None else f"{x:.1f}"
    for dataset, block in main["datasets"].items():
        for kind in ("all", "open", "closed"):
            m = block[kind]
            ci = "-" if not m["accuracy_ci95"] else f"{m['accuracy_ci95'][0]:.1f}-{m['accuracy_ci95'][1]:.1f}"
            lines.append(f"| {dataset} | {kind} | {m['generated']}/{m['selected']} | {f(m['accuracy_pct'])} | {ci} | "
                         f"{f(m['boxed_rate_pct'])} | {m['truncated']} | {f(m['mean_seconds'])} |")
    m = main["overall"]
    lines += ["", f"Overall: {f(m['accuracy_pct'])}% ({m['correct']}/{m['generated']}), boxed {f(m['boxed_rate_pct'])}%.",
              "", f"Primary metric: {summary['primary_metric']}. Other scorer/field combinations are in summary.json."]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def evaluate(system, feedback_size, adapter_dir, max_hours, summarize_only=False):
    protocol = read_json(PROTOCOL_DIR / "protocol.json")
    require_unchanged(protocol["dataset_file_sha256"], data_hashes(), "Prepared data")
    role = "feedback" if feedback_size else "test"
    name = ("feedback_check_" if feedback_size else "test_eval_") + FOLDERS[system] + "_v1"
    directory = OUTPUTS / name
    metadata, records = load_records(DATA_DIR)
    selected = select(records, protocol, feedback_size)
    if summarize_only:
        return write_outputs(directory, read_json(directory / "run.json"), selected, read_jsonl(directory / "predictions.jsonl"))
    settings = {"system": system, "role": role, "question_ids": [r["id"] for r in selected],
                "output_token_budget": protocol["output_token_budget"], "prompt": protocol["prompt"],
                "protocol_sha256": file_sha256(PROTOCOL_DIR / "protocol.json"),
                "adapter_dir": None if adapter_dir is None else str(Path(adapter_dir).resolve())}
    if directory.exists():
        run = read_json(directory / "run.json")
        if any(run[k] != v for k, v in settings.items()):
            raise ValueError(f"{directory} was started with different settings; it cannot be resumed")
        require_unchanged(run["code_sha256"], code_hashes(), "Code")
        print(f"Resuming {directory}", flush=True)
    else:
        directory.mkdir(parents=True)
        run = {**settings, "started_utc": utc_now(), "code_sha256": code_hashes(), "attempts": []}
        write_json(directory / "run.json", run)
        write_json(directory / "selection.json", selected)
        print(f"Output folder: {directory}", flush=True)
    backend, identity = make_backend(system, protocol, adapter_dir)
    run["attempts"].append({"started_utc": utc_now(), "backend": identity})
    write_json(directory / "run.json", run)
    predictions = read_jsonl(directory / "predictions.jsonl")
    done = {p["id"] for p in predictions if p.get("status") == "generated"}
    started, errors = time.perf_counter(), 0
    pending = [r for r in selected if r["id"] not in done]
    print(f"{len(done)} done, {len(pending)} to run", flush=True)
    for index, record in enumerate(pending, 1):
        if max_hours and time.perf_counter() - started > max_hours * 3600:
            print("Time budget reached; re-run the same command to continue.", flush=True)
            break
        try:
            prediction = backend(record)
            errors = 0
        except KeyboardInterrupt:
            print("Interrupted; re-run the same command to continue.", flush=True)
            break
        except Exception as failure:
            prediction = {"id": record["id"], "status": "generation_error", "error": safe_error(failure)}
            errors += 1
        prediction["recorded_utc"] = utc_now()
        append_jsonl(directory / "predictions.jsonl", prediction)
        predictions.append(prediction)
        mark = "ok" if prediction["status"] == "generated" else prediction["error"][:80]
        print(f"[{len(done) + index}/{len(selected)}] {record['id']}: {mark} "
              f"({prediction.get('generation_seconds', 0):.0f}s)", flush=True)
        if errors >= MAX_CONSECUTIVE_ERRORS:
            print("Stopping after repeated errors; inspect predictions.jsonl.", flush=True)
            break
    run["attempts"][-1]["ended_utc"] = utc_now()
    write_json(directory / "run.json", run)
    return write_outputs(directory, run, selected, predictions)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=sorted(FOLDERS), required=True)
    parser.add_argument("--feedback", type=int, default=0, help="Run N feedback questions instead of the test subset")
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--max-hours", type=float, default=0, help="Stop cleanly after this many hours (0 = no limit)")
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild summary files without generating")
    args = parser.parse_args()
    try:
        summary = evaluate(args.system, args.feedback, args.adapter_dir, args.max_hours, args.summarize_only)
        overall = summary["metrics"][summary["primary_metric"]]["overall"]
        print(json.dumps({"status": summary["status"], "folder": summary["folder"],
                          "accuracy_pct": overall["accuracy_pct"], "generated": overall["generated"],
                          "selected": overall["selected"]}, indent=2))
        return 0 if summary["status"] == "complete" else 2
    except Exception as error:
        print(json.dumps({"status": "blocked", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
