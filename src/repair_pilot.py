from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

from PIL import Image

from .answer_diagnostic import PLAIN_SCORING_ID, normalize_plain
from .benchmarks import file_sha256, image_path, load_bundle, read_jsonl
from .evaluation import select_records, source_fingerprints
from .repair_audit import (AUDIT_VALIDATION_POLICY, audit_payload, local_endpoint, request_json, review_payload,
                           safe_hint, validate_medical_audit, validate_review)
from .repair_model import READOUT_INSTRUCTION, reasoning_steps
from .runtime import environment_snapshot, safe_error, supervise, write_json


def read_pilot_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"pilot_id", "selection", "rollout_max_new_tokens", "suffix_max_new_tokens", "readout_max_new_tokens",
                "auditor_base_url", "auditor_model", "auditor_max_tokens", "auditor_timeout_seconds", "worker_timeout_seconds"}
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("Pilot configuration fields do not match the declared schema")
    if value.get("pilot_id") != "training-repair-pilot-v1" or value.get("selection") != {"vqa_rad": 2, "slake": 1}:
        raise ValueError("The approved pilot selects two VQA-RAD and one SLAKE training questions")
    if any(type(count) is not int for count in value["selection"].values()):
        raise ValueError("Pilot selection counts must be integers")
    for name, maximum in (("rollout_max_new_tokens", 256), ("suffix_max_new_tokens", 256),
                          ("readout_max_new_tokens", 64), ("auditor_max_tokens", 1024),
                          ("auditor_timeout_seconds", 300), ("worker_timeout_seconds", 7200)):
        if type(value.get(name)) is not int or not 1 <= value[name] <= maximum:
            raise ValueError(f"Invalid pilot limit: {name}")
    local_endpoint(value["auditor_base_url"])
    if not isinstance(value.get("auditor_model"), str) or not value["auditor_model"].strip():
        raise ValueError("An explicit local auditor model is required")
    return value


def select_training_cases(records, seed):
    selected = select_records(records, "vqa_rad", "train", seed, 2) + select_records(records, "slake", "train", seed, 1)
    if len(selected) != 3 or any(row["role"] != "train" for row in selected):
        raise ValueError("The prepared training pool does not satisfy the three-question pilot")
    return selected


def matched_readout(value, reference):
    answer = value["raw_response"]
    return bool(answer.strip()) and not value["hit_token_limit"] and normalize_plain(answer) == normalize_plain(reference)


def run_case(row, image, instruction, config, directory, callbacks):
    directory = Path(directory)
    directory.mkdir(parents=True)
    result = {"id": row["id"], "dataset": row["dataset"], "role": "train", "outcome": "started",
              "audit_validation_policy": AUDIT_VALIDATION_POLICY,
              "audit_calls": 0, "review_calls": 0, "optimizer_steps": 0, "training_examples_exported": 0,
              "clinical_boundary_status": "not_independently_verified",
              "learnability_gate": {"status": "not_implemented_in_this_pilot", "ratio": None, "gap": None}}
    def save():
        write_json(directory / "case.json", result)
    def readout(reasoning, label):
        write_json(directory / (label + "_readout_request.json"), {
            "question": row["question"], "reasoning_instruction": instruction,
            "image_sha256": row["image_sha256"], "assistant_reasoning": reasoning,
            "readout_instruction": READOUT_INSTRUCTION})
        value = callbacks["readout"](image, row["question"], instruction, reasoning, config["readout_max_new_tokens"])
        write_json(directory / (label + "_readout.json"), value)
        return value
    save()
    try:
        write_json(directory / "initial_request.json", {"question": row["question"],
                   "reasoning_instruction": instruction, "image_sha256": row["image_sha256"]})
        initial = callbacks["reasoning"](image, row["question"], instruction, config["rollout_max_new_tokens"])
        write_json(directory / "initial_reasoning.json", initial)
        text = initial["raw_response"]
        steps = reasoning_steps(text)
        write_json(directory / "reasoning_steps.json", steps)
        if not text.strip() or not steps:
            result.update(outcome="rejected_empty_initial_response")
            return result
        initial_answer = readout(text, "initial")
        initial_match = matched_readout(initial_answer, row["answer"])
        verification = {"scoring_id": PLAIN_SCORING_ID, "reference": row["answer"],
                        "initial_answer_match": initial_match, "candidate_answer_match": None,
                        "scope": "training-only separate answer-readout verification; not final evaluation"}
        write_json(directory / "verification.json", verification)
        result["initial_answer_match"] = initial_match
        if initial_match:
            result.update(outcome="initial_answer_match", training_admission="not_exported_reasoning_quality_unreviewed")
            return result
        payload = audit_payload(image, row["question"], text, steps,
                                initial.get("entropy", {"status": "unavailable"}), config)
        result["audit_calls"] += 1; save()
        try:
            audit = validate_medical_audit(callbacks["audit"](payload, config, directory, "audit"), steps)
        except ValueError as error:
            result.update(outcome="rejected_invalid_audit", error=safe_error(error))
            return result
        write_json(directory / "audit.json", audit)
        result.update(audit_structure_valid=True, auditor_action=audit["action"])
        if audit["action"] != "repair":
            result.update(outcome="auditor_" + audit["action"],
                          unused_audit_fields_ignored=["first_error_step", "error_family", "prefix_supported", "hint"],
                          hint_used_for_generation=False)
            return result
        allowed, reason = safe_hint(audit["hint"], row["answer"])
        if not allowed:
            result.update(outcome="rejected_hint", rejection_reason=reason)
            return result
        boundary = steps[audit["first_error_step"] - 1]["start"]
        prefix = text[:boundary]
        result.update(first_error_step=audit["first_error_step"], prefix_end=boundary,
                      repair_mode="full_regeneration" if boundary == 0 else "retained_prefix_continuation",
                      error_family=audit["error_family"], boundary_structure_valid=True)
        write_json(directory / "continuation_request.json", {"question": row["question"],
            "reasoning_instruction": instruction, "image_sha256": row["image_sha256"],
            "preserved_prefix": prefix, "minimal_hint": audit["hint"]})
        candidate = callbacks["continuation"](image, row["question"], instruction, prefix,
                                             audit["hint"], config["suffix_max_new_tokens"])
        write_json(directory / "candidate_reasoning.json", candidate)
        expected_cue = "\nLet me recheck the evidence. " + audit["hint"] + "\n"
        if (candidate.get("preserved_prefix") != prefix or candidate.get("recovery_cue") != expected_cue
                or candidate.get("assistant_prefill") != prefix + expected_cue
                or candidate.get("complete_response") != prefix + expected_cue + candidate.get("suffix", "")
                or not candidate.get("assistant_prefill_verified") or not candidate.get("input_prefix_unchanged")
                or not candidate.get("retained_prefix_unchanged")):
            result.update(outcome="rejected_prefix_integrity")
            return result
        result["retained_prefix_unchanged"] = True
        suffix = candidate["suffix"]
        if not suffix.strip() or candidate["hit_token_limit"]:
            result.update(outcome="rejected_empty_or_truncated_suffix")
            return result
        if suffix.strip() == text[boundary:].strip():
            result.update(outcome="rejected_unchanged_erroneous_suffix")
            return result
        sentences = [normalize_plain(step["text"]) for step in reasoning_steps(suffix) if len(step["text"].split()) >= 4]
        if any(count >= 3 for count in Counter(sentences).values()):
            result.update(outcome="rejected_repetition")
            return result
        candidate_answer = readout(candidate["complete_response"], "candidate")
        candidate_match = matched_readout(candidate_answer, row["answer"])
        verification["candidate_answer_match"] = candidate_match
        write_json(directory / "verification.json", verification)
        result["candidate_answer_match"] = candidate_match
        if not candidate_match:
            result.update(outcome="rejected_candidate_answer_mismatch")
            return result
        payload = review_payload(image, row["question"], candidate["complete_response"], candidate_answer["raw_response"], config)
        result["review_calls"] += 1; save()
        try:
            review = validate_review(callbacks["review"](payload, config, directory, "review"))
        except ValueError as error:
            result.update(outcome="rejected_invalid_review", error=safe_error(error))
            return result
        write_json(directory / "review.json", review)
        if review["verdict"] != "supported" or review["hint_leakage"] != "none_detected":
            result.update(outcome="rejected_candidate_integrity")
            return result
        result.update(outcome="preliminary_candidate",
                      training_admission="not_exported_pending_full_admission_checks")
        return result
    except Exception as error:
        result.update(outcome="infrastructure_error", error=safe_error(error))
        return result
    finally:
        save()


def summarize_pilot(directory, run, resources=None):
    directory = Path(directory)
    cases = []
    for index in range(1, 4):
        path = directory / "cases" / f"{index:02d}" / "case.json"
        if path.exists():
            cases.append(json.loads(path.read_text(encoding="utf-8")))
    complete = len(cases) == 3 and all(value["outcome"] not in ("started", "infrastructure_error") for value in cases)
    if resources and (resources["worker_exit_code"] != 0 or resources["limit_reached"]):
        complete = False
    summary = {"status": "complete" if complete else "incomplete", "purpose": "training-only repair feasibility; not benchmark evaluation",
        "output_folder": str(directory), "selected_count": len(run["selected_ids"]), "recorded_count": len(cases),
        "outcomes": dict(Counter(value["outcome"] for value in cases)),
        "initial_answer_match_count": sum(value.get("initial_answer_match", False) for value in cases),
        "candidate_answer_match_count": sum(value.get("candidate_answer_match", False) for value in cases),
        "preliminary_candidate_count": sum(value["outcome"] == "preliminary_candidate" for value in cases),
        "audit_calls": sum(value["audit_calls"] for value in cases), "review_calls": sum(value["review_calls"] for value in cases),
        "validated_audit_count": sum(value.get("audit_structure_valid", False) for value in cases),
        "auditor_path_exercised": any(value["audit_calls"] for value in cases),
        "continuation_path_exercised": any(value.get("retained_prefix_unchanged", False) for value in cases),
        "training_examples_exported": 0, "optimizer_steps": 0, "adapters_created": False,
        "medical_validity": "not_independently_verified", "full_medEvo_admission": "not_implemented_in_pilot",
        "cases": [{"id": value["id"], "outcome": value["outcome"],
                   **({"error": value["error"]} if "error" in value else {})} for value in cases]}
    if resources:
        summary["worker"] = resources
    write_json(directory / "summary.json", summary)
    return summary


def pilot_worker(root, data_dir, directory):
    lock = Path(directory) / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        return _pilot_worker(root, data_dir, directory)
    finally:
        lock.unlink(missing_ok=True)


def _pilot_worker(root, data_dir, directory):
    from .model import load_primary
    from .repair_model import answer_readout, continue_reasoning, generate_reasoning

    directory = Path(directory)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    if file_sha256(Path(data_dir) / "dataset.json") != run["dataset_metadata_sha256"]:
        raise ValueError("Pilot dataset identity changed")
    if source_fingerprints(Path(root)) != run["source_sha256"]:
        raise ValueError("Pilot source code changed after creation")
    _, records = load_bundle(data_dir, verify_images=True)
    by_id = {row["id"]: row for row in records}
    if any(identity not in by_id or by_id[identity]["role"] != "train" for identity in run["selected_ids"]):
        raise ValueError("A selected pilot question is outside the training role")
    model, processor = load_primary(run["settings"])
    write_json(directory / "model.json", {"model_id": run["settings"]["model_id"], "revision": run["settings"]["revision"],
               "model_weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters())})
    callbacks = {"reasoning": lambda *args: generate_reasoning(model, processor, *args),
                 "readout": lambda *args: answer_readout(model, processor, *args),
                 "continuation": lambda *args: continue_reasoning(model, processor, *args),
                 "audit": request_json, "review": request_json}
    for index, identity in enumerate(run["selected_ids"], 1):
        row = by_id[identity]
        print(f"{index}/3 {identity}: initial reasoning and answer readout", flush=True)
        with Image.open(image_path(data_dir, row["image_path"])) as original:
            image = original.convert("RGB")
        result = run_case(row, image, run["reasoning_instruction"], run["config"], directory / "cases" / f"{index:02d}", callbacks)
        print(f"{index}/3 {identity}: {result['outcome']}", flush=True)
        if result["outcome"] == "infrastructure_error":
            break
    summary = summarize_pilot(directory, run)
    return 0 if summary["status"] == "complete" else 2


def start_repair_pilot(root, data_dir, settings, config):
    root, data_dir = Path(root).resolve(), Path(data_dir).resolve()
    local_endpoint(config["auditor_base_url"])
    if config["auditor_model"] == settings["model_id"]:
        raise ValueError("Select an auditor distinct from the primary model")
    import psutil
    if psutil.virtual_memory().available < settings["minimum_available_gib"] * 1024**3:
        raise RuntimeError("Available RAM is below the configured minimum")
    metadata, records = load_bundle(data_dir, verify_images=True)
    selected = select_training_cases(records, settings["seed"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "outputs" / f"{stamp}_repair_pilot_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    run = {"pilot_id": config["pilot_id"], "started_utc": stamp, "role": "train", "settings": settings, "config": config,
           "audit_validation_policy": AUDIT_VALIDATION_POLICY,
           "selected_ids": [row["id"] for row in selected], "dataset_metadata_sha256": file_sha256(data_dir / "dataset.json"),
           "sources": metadata["sources"], "source_sha256": source_fingerprints(root),
           "reasoning_instruction": metadata["protocol"]["prompt"], "readout_instruction": READOUT_INSTRUCTION,
           "verifier_id": PLAIN_SCORING_ID,
           "limitations": ["Heuristic sentence spans, not independently validated semantic steps",
                "Separate primary-model answer readout is a training-only adaptation, not the paper's answer parser",
                "Local Qwen auditor replaces the paper's auditor; medical quality is unverified",
                "One suffix attempt per failed rollout; no model updates or training-target export",
                "Full learnability screening, twenty error subtypes, feedback navigation and SFT admission are deferred"]}
    write_json(directory / "run.json", run)
    write_json(directory / "environment.json", environment_snapshot(root))
    print(f"Pilot folder: {directory}", flush=True)
    command = [sys.executable, str(root / "evaluate.py"), "--data-dir", str(data_dir), "--repair-worker-dir", str(directory)]
    try:
        resources = supervise(command, {**settings, "timeout_seconds": config["worker_timeout_seconds"]}, directory / "console.log")
        weights = json.loads((directory / "model.json").read_text())["model_weight_bytes"] if (directory / "model.json").exists() else 0
        resources.update(memory_measurement_plausible=bool(weights and (resources["peak_process_tree_rss_bytes"] or 0) >= .95 * weights),
                         measurement_scope="Python worker process tree; excludes the external Ollama service")
        if not resources["memory_measurement_plausible"]:
            resources["warning"] = "RAM sample requires target-PC verification; do not report as validated peak"
        write_json(directory / "resources.json", resources)
        return summarize_pilot(directory, run, resources)
    except (Exception, KeyboardInterrupt) as error:
        summary = summarize_pilot(directory, run)
        summary.update(status="incomplete", error=safe_error(error))
        write_json(directory / "summary.json", summary)
        return summary
