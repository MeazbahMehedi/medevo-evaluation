"""Paired auditor diagnostic: change only the saved request's system instruction."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from PIL import Image

from .benchmarks import file_sha256, image_path, load_bundle
from .evaluation import source_fingerprints
from .readout_experiment import EVIDENCE_FILES, validate_collection
from .repair_audit import (AUDIT_VALIDATION_POLICY, audit_payload, local_endpoint, request_json,
                           safe_hint, validate_medical_audit)
from .repair_model import reasoning_steps
from .runtime import environment_snapshot, safe_error, supervise, write_json
from .training_collection import read_json, tree_hashes


EXPERIMENT_ID = "short-auditor-instruction-v1"
SHORT_INSTRUCTION = (
    "Inspect the image independently, then audit the question and reasoning. No reference answer is given. "
    "Return only JSON matching the supplied schema.\n"
    "repair: identify the first consequential medical reasoning error supported by visible evidence. "
    "Set first_error_step to its 1-based step number (never 0). All earlier steps must be supported; "
    "set prefix_supported accordingly. Step 1 retains no prefix. "
    "Families: V visual misreading; R invalid inference; D differential comparison; K missing prerequisite knowledge.\n"
    "skip: no justified repair or only a format problem. unresolved: insufficient evidence. "
    "For either, use step 0 and an empty hint.\n"
    "A repair hint must be one short imperative inspection/reasoning operation. Only the hint reaches the model. "
    "Do not state a finding, diagnosis, answer, corrected rationale or new clinical evidence, or repeat a failed claim. "
    "Explain the error and boundary in evidence. Steps are heuristic spans. Entropy is a proposal, not proof.")
AUDIT_FILES = ("reasoning_steps.json", "audit_request.json", "audit_response.json")


def shorter_request(original):
    trial = deepcopy(original)
    if trial["messages"][0]["role"] != "system" or len(SHORT_INSTRUCTION) >= len(trial["messages"][0]["content"]):
        raise ValueError("The trial must replace a longer system instruction")
    trial["messages"][0]["content"] = SHORT_INSTRUCTION
    return trial


def validate_audit_inputs(root, data_dir, collection_dir, settings, config):
    previous, hashes, _ = validate_collection(root, data_dir, collection_dir, settings)
    if previous["config"] != config:
        raise ValueError("Auditor model, endpoint and limits must match the source collection")
    local_endpoint(config["auditor_base_url"])
    saved_sources = {key.replace("\\", "/"): value for key, value in previous["source_sha256"].items()}
    if saved_sources.get("src/repair_audit.py") != source_fingerprints(Path(root))["src/repair_audit.py"]:
        raise ValueError("The auditor request and validation implementation changed")
    _, records = load_bundle(data_dir, verify_images=True)
    by_id = {row["id"]: row for row in records}
    inputs = []
    for index, identity in enumerate(previous["selected_ids"], 1):
        folder = Path(collection_dir) / "cases" / f"{index:02d}"
        row = by_id[identity]
        initial = read_json(folder / "initial_reasoning.json")
        steps = read_json(folder / "reasoning_steps.json")
        if steps != reasoning_steps(initial["raw_response"]):
            raise ValueError("Saved audit spans differ from the unchanged reasoning")
        with Image.open(image_path(data_dir, row["image_path"])) as original:
            expected = audit_payload(original.convert("RGB"), row["question"], initial["raw_response"], steps,
                                     initial.get("entropy", {"status": "unavailable"}), config)
        payload = read_json(folder / "audit_request.json")
        if payload != expected:
            raise ValueError("The saved auditor request differs from the prepared image or frozen inputs")
        read_json(folder / "audit_response.json")
        inputs.append({"id": identity, "request": shorter_request(payload)})
    if tree_hashes(collection_dir) != hashes:
        raise ValueError("The source collection changed during audit validation")
    return previous, hashes, inputs


def assess_response(path, steps, reference):
    result = {"response_available": Path(path).exists(), "schema_valid": False, "action": None,
              "outcome": "not_recorded", "clinical_coherence": "requires_review", "medical_validity": "not_verified"}
    if not result["response_available"]:
        return result
    try:
        envelope = json.loads(read_json(path)["response_text"])
        choice = envelope["choices"][0]
        content = choice["message"]["content"]
        result.update(raw_content=content, provider_finish_reason=choice.get("finish_reason"), usage=envelope.get("usage"))
        audit = json.loads(content)
        result["raw_audit"] = audit
        validate_medical_audit(audit, steps)
        result.update(schema_valid=True, action=audit["action"], hint=audit["hint"], evidence=audit["evidence"],
                      first_error_step=audit["first_error_step"])
        if audit["action"] != "repair":
            result.update(outcome="auditor_" + audit["action"], hint_used_for_generation=False)
        else:
            allowed, reason = safe_hint(audit["hint"], reference)
            result.update(hint_check_passed=allowed, hint_rejection_reason=reason,
                          outcome="repair_requires_clinical_review" if allowed else "rejected_hint")
            if allowed:
                result["prefix_end"] = steps[audit["first_error_step"] - 1]["start"]
                result["repair_mode_if_later_approved"] = "full_regeneration" if result["prefix_end"] == 0 else "retained_prefix_continuation"
    except (ValueError, TypeError, KeyError, IndexError) as error:
        result.update(outcome="invalid_audit", error=safe_error(error))
    return result


def summarize_audit_experiment(directory, run, resources=None):
    directory = Path(directory)
    comparisons = []
    for index, identity in enumerate(run["selected_ids"], 1):
        evidence = directory / "evidence" / f"{index:02d}"
        steps = read_json(evidence / "reasoning_steps.json")
        reference = read_json(evidence / "verification.json")["reference"]
        trial_dir = directory / "audits" / f"{index:02d}"
        attempt = read_json(trial_dir / "case.json") if (trial_dir / "case.json").exists() else None
        comparisons.append({"id": identity, "question": read_json(evidence / "initial_request.json")["question"],
            "dataset": read_json(evidence / "case.json")["dataset"], "reference_for_local_checks_only": reference,
            "original": assess_response(evidence / "audit_response.json", steps, reference),
            "trial": assess_response(trial_dir / "audit_response.json", steps, reference), "trial_attempt": attempt})
    received = sum(row["trial"]["response_available"] for row in comparisons)
    attempts = sum(row["trial_attempt"] is not None for row in comparisons)
    complete = received == 12 and not (directory / "worker_error.json").exists()
    if resources and (resources["worker_exit_code"] != 0 or resources["limit_reached"]):
        complete = False
    def metrics(label):
        values = [row[label] for row in comparisons]
        present = sum(value["response_available"] for value in values)
        valid = sum(value["schema_valid"] for value in values)
        return {"selected_count": 12, "saved_response_count": present,
                "schema_valid_count": valid, "invalid_audit_count": present - valid,
                "schema_valid_pct": 100 * valid / 12 if (label == "original" or complete) and present == 12 else None,
                "outcomes": dict(Counter(value["outcome"] for value in values)),
                "schema_valid_repair_count": sum(value["action"] == "repair" for value in values),
                "repair_hint_pass_count": sum(value.get("hint_check_passed", False) for value in values),
                "clinical_coherence": "requires_review_of_all_raw_responses"}
    summary = {"status": "complete" if complete else "incomplete", "experiment_id": EXPERIMENT_ID,
        "output_folder": str(directory), "purpose": "paired training-case auditor instruction diagnostic; not benchmark evaluation",
        "original": metrics("original"), "trial": metrics("trial"),
        "new_audit_calls_started": attempts, "new_primary_reasoning_calls": 0, "new_readout_calls": 0,
        "new_continuations": 0, "new_review_calls": 0, "optimizer_steps": 0,
        "training_examples_exported": 0, "adapters_created": False, "automatic_candidate_admission": False,
        "audit_validation_policy": AUDIT_VALIDATION_POLICY, "medical_validity": "not_independently_verified",
        "next_action": "review_paired_decisions_and_coherence; no automatic retries or further prompt trials"}
    if resources:
        summary["worker"] = resources
    if (directory / "worker_error.json").exists():
        summary["error"] = read_json(directory / "worker_error.json")["error"]
    write_json(directory / "summary.json", summary)
    write_json(directory / "comparison.json", {**summary, "cases": comparisons})
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    lines = ["# Auditor instruction comparison", "", "Saved baseline versus shorter instruction. No continuation or training is performed.",
             "Schema and hint checks are structural safeguards, not clinical validation.", "",
             "| Case | Original outcome | Trial outcome | Trial hint | Trial evidence |", "|---|---|---|---|---|"]
    for row in comparisons:
        trial = row["trial"]
        lines.append("| " + " | ".join(cell(v) for v in (row["id"], row["original"]["outcome"], trial["outcome"],
                     trial.get("hint", ""), trial.get("evidence", trial.get("error", "")))) + " |")
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def audit_worker(root, directory):
    directory = Path(directory)
    lock = directory / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        run = read_json(directory / "run.json")
        if (run["experiment_id"] != EXPERIMENT_ID or run["source_sha256"] != source_fingerprints(Path(root))
                or file_sha256(directory / "inputs.json") != run["inputs_sha256"]
                or tree_hashes(directory / "evidence") != run["evidence_sha256"]):
            raise ValueError("Auditor experiment code, requests or evidence changed")
        local_endpoint(run["config"]["auditor_base_url"])
        inputs = read_json(directory / "inputs.json")
        if len(inputs) != 12 or [item["id"] for item in inputs] != run["selected_ids"]:
            raise ValueError("The audit workload must contain the twelve fixed IDs")
        for index, item in enumerate(inputs, 1):
            original = read_json(directory / "evidence" / f"{index:02d}" / "audit_request.json")
            if set(item) != {"id", "request"} or item["request"] != shorter_request(original):
                raise ValueError("Only the system instruction may change")
        if (directory / "audits").exists() and any((directory / "audits").iterdir()):
            raise ValueError("Audit attempts already exist; do not repeat this worker")
        for index, item in enumerate(inputs, 1):
            folder = directory / "audits" / f"{index:02d}"
            folder.mkdir(parents=True)
            attempt = {"id": item["id"], "status": "started", "new_audit_calls_started": 1}
            write_json(folder / "case.json", attempt)
            try:
                request_json(item["request"], run["config"], folder, "audit")
                attempt["status"] = "response_received"
            except (ValueError, TypeError, KeyError, IndexError) as error:
                if not (folder / "audit_response.json").exists():
                    raise
                attempt.update(status="invalid_response", error=safe_error(error))
            except Exception as error:
                attempt.update(status="transport_error", error=safe_error(error))
                raise
            finally:
                write_json(folder / "case.json", attempt)
            print(f"{index}/12 {item['id']}: auditor response recorded", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        write_json(directory / "worker_error.json", {"error": safe_error(error)})
        return 2
    finally:
        lock.unlink(missing_ok=True)


def start_audit_experiment(root, data_dir, collection_dir, settings, config):
    root, collection_dir = Path(root).resolve(), Path(collection_dir).resolve()
    previous, hashes, inputs = validate_audit_inputs(root, data_dir, collection_dir, settings, config)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "outputs" / f"{stamp}_audit_comparison_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    for name in ("run.json", "summary.json"):
        shutil.copy2(collection_dir / name, directory / ("original_" + name))
    for index in range(1, 13):
        folder = directory / "evidence" / f"{index:02d}"
        folder.mkdir(parents=True)
        for name in EVIDENCE_FILES + AUDIT_FILES:
            shutil.copy2(collection_dir / "cases" / f"{index:02d}" / name, folder / name)
            if file_sha256(folder / name) != hashes[f"cases/{index:02d}/{name}"]:
                raise ValueError("A source audit artifact changed while copying")
    write_json(directory / "inputs.json", inputs)
    run = {"experiment_id": EXPERIMENT_ID, "role": "train", "started_utc": stamp,
           "settings": settings, "config": config, "selected_ids": previous["selected_ids"],
           "source_collection": str(collection_dir), "source_collection_sha256": hashes,
           "source_sha256": source_fingerprints(root), "inputs_sha256": file_sha256(directory / "inputs.json"),
           "evidence_sha256": tree_hashes(directory / "evidence"), "trial_system_instruction": SHORT_INSTRUCTION,
           "audit_validation_policy": AUDIT_VALIDATION_POLICY, "changed_request_field": "messages[0].content",
           "limitations": ["Same model tag does not prove immutable historical Ollama weights",
               "Instruction selected after inspecting these training cases; exploratory diagnostic only",
               "Schema and hint validity do not establish sound medical reasoning or a correct boundary",
               "Coherence requires review; no continuation, candidate admission or training is performed"]}
    write_json(directory / "run.json", run)
    write_json(directory / "environment.json", environment_snapshot(root))
    if tree_hashes(collection_dir) != hashes:
        raise ValueError("The original collection changed before worker launch")
    print(f"Auditor comparison folder: {directory}", flush=True)
    command = [sys.executable, str(root / "evaluate.py"), "--audit-worker-dir", str(directory)]
    try:
        resources = supervise(command, {**settings, "timeout_seconds": config["worker_timeout_seconds"]}, directory / "console.log")
        resources["measurement_scope"] = "Python HTTP worker only; excludes external Ollama model, CPU and memory usage"
        resources["ollama_peak_memory_measured"] = False
        write_json(directory / "resources.json", resources)
        summary = summarize_audit_experiment(directory, run, resources)
    except (Exception, KeyboardInterrupt) as error:
        write_json(directory / "worker_error.json", {"error": safe_error(error)})
        summary = summarize_audit_experiment(directory, run)
    if tree_hashes(collection_dir) != hashes:
        summary.update(status="incomplete", error="Original collection changed during the auditor experiment")
        summary["trial"]["schema_valid_pct"] = None
        write_json(directory / "summary.json", summary)
        report = read_json(directory / "comparison.json")
        write_json(directory / "comparison.json", {**report, **summary})
    return summary
