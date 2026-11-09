"""Run the approved schema-grounding check on the first saved training case."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import uuid

from src.audit_experiment import AUDIT_FILES, assess_response, validate_audit_inputs
from src.benchmarks import file_sha256
from src.evaluation import source_fingerprints
from src.readout_experiment import EVIDENCE_FILES
from src.repair_audit import AUDIT_VALIDATION_POLICY, request_json
from src.repair_pilot import read_pilot_config
from src.runtime import environment_snapshot, read_settings, safe_error, write_json
from src.training_collection import read_json, tree_hashes


ROOT = Path(__file__).resolve().parent
CHECK_ID = "single-audit-schema-grounding-v1"
MODEL_CHECK_ID = "single-audit-model-comparison-v1"
REPLACEMENT_MODEL = "qwen2.5vl:7b"


def grounded_request(original):
    trial = deepcopy(original)
    if trial["messages"][0]["role"] != "system":
        raise ValueError("The saved request must begin with the original system instruction")
    schema = trial["response_format"]["json_schema"]["schema"]
    trial["messages"][0]["content"] += (
        "\n\nOutput JSON schema (identical to response_format):\n"
        + json.dumps(schema, indent=2, ensure_ascii=False))
    return trial


def validate_model_baseline(directory, identity, settings, config, collection_hashes, expected_request):
    directory = Path(directory)
    before = tree_hashes(directory)
    if (directory / "run.lock").exists():
        raise ValueError("Wait until the baseline check has stopped")
    run, summary = read_json(directory / "run.json"), read_json(directory / "summary.json")
    if (run.get("check_id") != CHECK_ID or run.get("role") != "train"
            or run.get("selected_id") != identity or summary.get("selected_id") != identity
            or run.get("settings") != settings or run.get("config") != config
            or run.get("source_collection_sha256") != collection_hashes
            or summary.get("check_id") != CHECK_ID or summary.get("status") != "complete"
            or summary.get("new_audit_calls_started") != 1
            or read_json(directory / "trial/audit_request.json") != expected_request):
        raise ValueError("Use the completed same-case, same-request 3B schema-grounded baseline")
    read_json(directory / "trial/audit_response.json")
    if tree_hashes(directory) != before:
        raise ValueError("The baseline check changed during validation")
    return before


def check_one_case(root, data_dir, collection_dir, settings, config, *, replacement_model=None, baseline_check_dir=None):
    if bool(replacement_model) != bool(baseline_check_dir):
        raise ValueError("A replacement model and its baseline-check folder must be provided together")
    if replacement_model and (replacement_model != REPLACEMENT_MODEL or config["auditor_model"] != "qwen2.5vl:3b"):
        raise ValueError("The approved comparison replaces qwen2.5vl:3b with qwen2.5vl:7b only")
    root, collection_dir = Path(root).resolve(), Path(collection_dir).resolve()
    previous, hashes, _ = validate_audit_inputs(root, data_dir, collection_dir, settings, config)
    identity = previous["selected_ids"][0]
    source = collection_dir / "cases" / "01"
    original = read_json(source / "audit_request.json")
    payload = grounded_request(original)
    baseline_hashes = None
    effective_config = dict(config)
    if replacement_model:
        baseline_check_dir = Path(baseline_check_dir).resolve()
        baseline_hashes = validate_model_baseline(baseline_check_dir, identity, settings, config, hashes, payload)
        payload["model"] = replacement_model
        effective_config["auditor_model"] = replacement_model
    check_id = MODEL_CHECK_ID if replacement_model else CHECK_ID
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "auditor_model_comparison" if replacement_model else "single_audit"
    directory = root / "outputs" / f"{stamp}_{label}_{uuid.uuid4().hex[:6]}"
    evidence = directory / "evidence"
    evidence.mkdir(parents=True)
    for name in EVIDENCE_FILES + AUDIT_FILES:
        shutil.copy2(source / name, evidence / name)
        if file_sha256(evidence / name) != hashes["cases/01/" + name]:
            raise ValueError("Original case changed while copying evidence")
    if replacement_model:
        for name in ("run.json", "summary.json", "trial/audit_request.json", "trial/audit_response.json"):
            destination = directory / "baseline_3b" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(baseline_check_dir / name, destination)
            if file_sha256(destination) != baseline_hashes[name]:
                raise ValueError("The 3B baseline changed while copying")
    run = {"check_id": check_id, "started_utc": stamp, "role": "train", "selected_id": identity,
           "selection_rule": "first ID in the unchanged twelve-case collection; no replacement sampling",
           "settings": settings, "config": effective_config, "source_config": config, "source_collection": str(collection_dir),
           "source_collection_sha256": hashes, "source_sha256": source_fingerprints(root),
           "check_script_sha256": file_sha256(__file__),
           "changed_request_field": "model" if replacement_model else "messages[0].content",
           "audit_validation_policy": AUDIT_VALIDATION_POLICY, "maximum_new_audit_calls": 1,
           "limitations": ["Single post-inspection integration check; not a benchmark evaluation",
                           "Schema/hint validity does not establish clinical coherence or sound error localization",
                           "No primary-model generation, continuation, admission, training or automatic retry"]}
    if replacement_model:
        run.update(baseline_check_folder=str(baseline_check_dir), baseline_check_sha256=baseline_hashes,
                   baseline_model=config["auditor_model"], trial_model=replacement_model)
    write_json(directory / "run.json", run)
    write_json(directory / "environment.json", environment_snapshot(root))
    trial = directory / "trial"
    trial.mkdir()
    if tree_hashes(collection_dir) != hashes:
        raise ValueError("The original collection changed before the audit call")
    if replacement_model and tree_hashes(baseline_check_dir) != baseline_hashes:
        raise ValueError("The 3B baseline changed before the comparison call")
    print(f"Single-case output folder: {directory}", flush=True)
    started = time.perf_counter()
    error = None
    try:
        # Exactly one HTTP call; parsing failures remain captured in the raw response.
        request_json(payload, effective_config, trial, "audit")
    except (Exception, KeyboardInterrupt) as failure:
        error = safe_error(failure)
        write_json(trial / "request_error.json", {"error": error})
    steps = read_json(evidence / "reasoning_steps.json")
    reference = read_json(evidence / "verification.json")["reference"]
    result = assess_response(trial / "audit_response.json", steps, reference)
    baseline_response = directory / "baseline_3b/trial/audit_response.json" if replacement_model else evidence / "audit_response.json"
    summary = {"status": "complete" if result["response_available"] else "incomplete",
        "check_id": check_id, "output_folder": str(directory), "selected_id": identity,
        "purpose": "one-case auditor-model comparison" if replacement_model else "one-case original-instruction audit with explicit schema grounding",
        "original_condition": "schema-grounded 3B" if replacement_model else "original audit before explicit schema",
        "trial_model": effective_config["auditor_model"],
        "original": assess_response(baseline_response, steps, reference), "trial": result,
        "new_audit_calls_started": 1, "elapsed_seconds": round(time.perf_counter() - started, 3),
        "new_primary_reasoning_calls": 0, "new_readout_calls": 0, "new_continuations": 0,
        "training_examples_exported": 0, "optimizer_steps": 0, "adapters_created": False,
        "automatic_candidate_admission": False, "medical_validity": "not_independently_verified",
        "next_action": "review_this_response_before_any_continuation_or_model_change"}
    if error:
        summary["request_error"] = error
    if tree_hashes(collection_dir) != hashes:
        summary.update(status="incomplete", integrity_error="Original collection changed during the check")
    if replacement_model and tree_hashes(baseline_check_dir) != baseline_hashes:
        summary.update(status="incomplete", integrity_error="3B baseline changed during the comparison")
    write_json(directory / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Check explicit schema grounding on one saved medical audit")
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "benchmarks")
    parser.add_argument("--settings", type=Path, default=ROOT / "settings.json")
    parser.add_argument("--pilot-config", type=Path, default=ROOT / "configs" / "repair_pilot.json")
    parser.add_argument("--auditor-model", choices=(REPLACEMENT_MODEL,), help="Explicitly run the approved 7B replacement comparison")
    parser.add_argument("--baseline-check-dir", type=Path, help="Completed 3B schema-grounded one-case check")
    args = parser.parse_args()
    try:
        result = check_one_case(ROOT, args.data_dir.resolve(), args.collection_dir.resolve(),
                                read_settings(args.settings), read_pilot_config(args.pilot_config),
                                replacement_model=args.auditor_model, baseline_check_dir=args.baseline_check_dir)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "complete" else 2
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
