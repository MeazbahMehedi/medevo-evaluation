"""Bounded training evidence collection; this module does not train adapters."""

from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from PIL import Image

from .answer_diagnostic import PLAIN_SCORING_ID, normalize_plain
from .benchmarks import file_sha256, image_path, load_bundle
from .evaluation import select_records, source_fingerprints
from .repair_audit import AUDIT_VALIDATION_POLICY, audit_payload, validate_medical_audit
from .repair_model import READOUT_INSTRUCTION, reasoning_steps
from .repair_pilot import matched_readout, run_case, select_training_cases
from .runtime import environment_snapshot, safe_error, supervise, write_json


COLLECTION_ID = "training-collection-12-v1"
# These two archived implementations differ only in handling unused non-repair fields.
LEGACY_SOURCES = {
    "src/repair_pilot.py": "02464530abaffac6002d4895241113193baba7688e7c101d7c2fe04920c42fef",
    "src/repair_audit.py": "c152a894d7f46dca24aa73c2d44e3d6f1ce7e33bd9742d7f00957eb44fbcc608",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def tree_hashes(directory):
    directory = Path(directory)
    if directory.is_symlink():
        raise ValueError("Evidence folders must not be symbolic links")
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("Evidence must contain regular files, not symbolic links")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = file_sha256(path)
    return result


def select_collection_cases(records, seed):
    selected = [row for dataset in ("vqa_rad", "slake")
                for row in select_records(records, dataset, "train", seed, 6)]
    if (len(selected) != 12 or len({r["id"] for r in selected}) != 12
            or any(row["role"] != "train" for row in selected)):
        raise ValueError("Collection requires six eligible training questions from each dataset")
    return selected


def classify_readout(value, question, reference):
    """Diagnostic labels only; neither answers nor the verifier are changed."""
    if value is None:
        return "not_recorded"
    answer = value["raw_response"]
    if not isinstance(answer, str) or type(value["hit_token_limit"]) is not bool:
        raise ValueError("Readout evidence has invalid types")
    if not answer.strip():
        return "empty_readout"
    if value["hit_token_limit"]:
        return "truncated_readout"
    if normalize_plain(answer) == normalize_plain(question):
        return "question_echo"
    return "answer_match" if matched_readout(value, reference) else "answer_mismatch"


def validate_previous(root, data_dir, pilot_dir, settings, config, metadata, records):
    """Accept the three saved non-repair cases only when their controls agree."""
    pilot_dir = Path(pilot_dir)
    if (pilot_dir / "run.lock").exists():
        raise ValueError("The previous pilot is locked; wait until it stops")
    hashes = tree_hashes(pilot_dir)
    run = read_json(pilot_dir / "run.json")
    expected_ids = [r["id"] for r in select_training_cases(records, settings["seed"])]
    if (run.get("pilot_id") != "training-repair-pilot-v1" or run.get("role") != "train"
            or run.get("selected_ids") != expected_ids):
        raise ValueError("Reuse requires the original fixed three-question training pilot")
    if (run.get("settings") != settings or run.get("config") != config
            or run.get("dataset_metadata_sha256") != file_sha256(Path(data_dir) / "dataset.json")
            or run.get("reasoning_instruction") != metadata["protocol"]["prompt"]
            or run.get("readout_instruction") != READOUT_INSTRUCTION
            or run.get("verifier_id") != PLAIN_SCORING_ID):
        raise ValueError("Saved pilot model, data, prompts, limits or scoring differ from the collection")
    current = source_fingerprints(Path(root))
    saved = {name.replace("\\", "/"): digest for name, digest in run["source_sha256"].items()}
    for name in ("src/model.py", "src/repair_model.py", "src/answer_diagnostic.py", "src/scoring.py",
                 "src/benchmarks.py", "src/repair_pilot.py", "src/repair_audit.py"):
        allowed = {current[name]}
        if name in LEGACY_SOURCES:
            allowed.add(LEGACY_SOURCES[name])
        if saved.get(name) not in allowed:
            raise ValueError(f"Saved generation or verification implementation differs: {name}")
    by_id = {row["id"]: row for row in records}
    reused = {}
    for index, identity in enumerate(expected_ids, 1):
        case_dir = pilot_dir / "cases" / f"{index:02d}"
        row = by_id[identity]
        case = read_json(case_dir / "case.json")
        initial = read_json(case_dir / "initial_reasoning.json")
        steps = read_json(case_dir / "reasoning_steps.json")
        answer = read_json(case_dir / "initial_readout.json")
        verification = read_json(case_dir / "verification.json")
        request = {"question": row["question"], "reasoning_instruction": run["reasoning_instruction"],
                   "image_sha256": row["image_sha256"]}
        if (case.get("id") != identity or case.get("role") != "train" or case.get("dataset") != row["dataset"]
                or case.get("outcome") not in ("rejected_invalid_audit", "auditor_skip", "auditor_unresolved")
                or case.get("optimizer_steps") != 0 or case.get("training_examples_exported") != 0
                or case.get("audit_calls") != 1 or case.get("review_calls") != 0
                or steps != reasoning_steps(initial["raw_response"])
                or read_json(case_dir / "initial_request.json") != request
                or read_json(case_dir / "initial_readout_request.json") != {
                    **request, "assistant_reasoning": initial["raw_response"], "readout_instruction": READOUT_INSTRUCTION}
                or verification.get("reference") != row["answer"]
                or verification.get("scoring_id") != PLAIN_SCORING_ID
                or verification.get("initial_answer_match") is not False
                or matched_readout(answer, row["answer"])):
            raise ValueError(f"Saved case evidence disagrees with the training record: {identity}")
        with Image.open(image_path(data_dir, row["image_path"])) as original:
            payload = audit_payload(original.convert("RGB"), row["question"], initial["raw_response"], steps,
                                    initial.get("entropy", {"status": "unavailable"}), config)
        if read_json(case_dir / "audit_request.json") != payload:
            raise ValueError("Saved auditor request differs from the unchanged collection request")
        envelope = json.loads(read_json(case_dir / "audit_response.json")["response_text"])
        audit = validate_medical_audit(json.loads(envelope["choices"][0]["message"]["content"]), steps)
        if audit["action"] not in ("skip", "unresolved"):
            raise ValueError("This collection reuses only the three recorded non-repair decisions")
        if any((case_dir / name).exists() for name in ("continuation_request.json", "candidate_reasoning.json")):
            raise ValueError("A reused non-repair case unexpectedly contains continuation evidence")
        reused[identity] = {"source_case": f"cases/{index:02d}", "original_outcome": case["outcome"],
                            "outcome": "auditor_" + audit["action"], "action": audit["action"]}
    if tree_hashes(pilot_dir) != hashes:
        raise ValueError("Original pilot files changed during validation")
    return run, hashes, reused


def record_case(directory, row, reused=None):
    directory = Path(directory)
    case = read_json(directory / "case.json")
    classification = {}
    for label in ("initial", "candidate"):
        path = directory / (label + "_readout.json")
        classification[label] = classify_readout(read_json(path) if path.exists() else None,
                                                row["question"], row["answer"])
    record = {"id": row["id"], "dataset": row["dataset"], "role": "train", "reused": reused is not None,
              "original_outcome": case["outcome"], "outcome": case["outcome"],
              "readout_classification": classification, "audit_validation_policy": AUDIT_VALIDATION_POLICY}
    if reused:
        record.update(reused)
    else:
        record["action"] = case.get("auditor_action")
    write_json(directory / "collection_record.json", record)
    return record


def collect_cases(data_dir, directory, run, records, callbacks):
    """Process only the nine unrecorded IDs; no replacement questions or retries."""
    directory = Path(directory)
    by_id = {row["id"]: row for row in records}
    for index, identity in enumerate(run["selected_ids"], 1):
        case_dir = directory / "cases" / f"{index:02d}"
        if identity in run["reused_cases"]:
            if not (case_dir / "collection_record.json").exists():
                raise ValueError("Reused evidence is missing; collection cannot regenerate it")
            print(f"{index}/12 {identity}: reused saved non-repair decision", flush=True)
            continue
        row = by_id[identity]
        print(f"{index}/12 {identity}: collecting new training evidence", flush=True)
        with Image.open(image_path(data_dir, row["image_path"])) as original:
            image = original.convert("RGB")
        result = run_case(row, image, run["reasoning_instruction"], run["config"], case_dir, callbacks)
        record_case(case_dir, row)
        print(f"{index}/12 {identity}: {result['outcome']}", flush=True)
        if result["outcome"] == "infrastructure_error":
            break
    return summarize_collection(directory, run)


def summarize_collection(directory, run, resources=None):
    directory = Path(directory)
    records, candidates, attempts = [], [], []
    for index, identity in enumerate(run["selected_ids"], 1):
        case_dir = directory / "cases" / f"{index:02d}"
        if not (case_dir / "collection_record.json").exists():
            continue
        record = read_json(case_dir / "collection_record.json")
        if record["id"] != identity:
            raise ValueError("Collection case identity differs from the fixed selection")
        case = read_json(case_dir / "case.json")
        records.append(record)
        link = {"id": identity, "dataset": record["dataset"], "case_directory": f"cases/{index:02d}",
                "outcome": record["outcome"], "admission_status": "pending_full_admission_checks"}
        if record["outcome"] in ("initial_answer_match", "preliminary_candidate"):
            initial = read_json(case_dir / "initial_reasoning.json")
            candidates.append({**link, "kind": "correct_rollout" if record["outcome"] == "initial_answer_match" else "repaired_rollout",
                               "initial_reasoning_truncated": initial["hit_token_limit"],
                               "clinical_validity": "not_independently_verified"})
        if record.get("action") == "repair":
            attempts.append({**link, "admission_status": "pending" if record["outcome"] == "preliminary_candidate" else "rejected",
                             "continuation_performed": (case_dir / "candidate_reasoning.json").exists()})
        if case.get("optimizer_steps") != 0 or case.get("training_examples_exported") != 0:
            raise ValueError("Collection evidence unexpectedly contains training operations")
    fresh = [r for r in records if not r["reused"]]
    complete = (len(records) == 12 and len(fresh) == 9
                and all(r["outcome"] not in ("started", "infrastructure_error") for r in records))
    if resources and (resources["worker_exit_code"] != 0 or resources["limit_reached"]):
        complete = False
    calls = {}
    for kind in ("audit", "review"):
        calls[kind + "_calls_new"] = sum(read_json(directory / "cases" / f"{run['selected_ids'].index(r['id']) + 1:02d}" / "case.json")[kind + "_calls"] for r in fresh)
    summary = {"status": "complete" if complete else "incomplete", "collection_id": COLLECTION_ID,
               "purpose": "bounded training candidate collection; not benchmark evaluation or adapter training",
               "output_folder": str(directory), "selected_count": 12, "recorded_count": len(records),
               "reused_count": len(records) - len(fresh), "new_case_count": len(fresh),
               "outcomes": dict(Counter(r["outcome"] for r in records)),
               "initial_readout_categories": dict(Counter(r["readout_classification"]["initial"] for r in records)),
               "candidate_readout_categories": dict(Counter(r["readout_classification"]["candidate"] for r in records)),
               "potential_candidate_count": len(candidates), "repair_decision_count": len(attempts),
               "historical_audit_calls_reused": len(records) - len(fresh), **calls,
               "training_examples_exported": 0, "optimizer_steps": 0, "adapters_created": False,
               "full_admission": "not_implemented; candidate bank is an evidence index only",
               "medical_validity": "not_independently_verified",
               "next_action": ("finish_or_investigate_incomplete_collection" if not complete else
                               "review_candidates_and_implement_remaining_admission_checks" if candidates else
                               "reassess_model_and_auditor_before_training"), "cases": records}
    if resources:
        summary["worker"] = resources
    write_json(directory / "candidate_bank.json", {"training_targets_exported": 0, "candidates": candidates})
    write_json(directory / "repair_attempts.json", {"attempts": attempts})
    write_json(directory / "summary.json", summary)
    return summary


def collection_worker(root, data_dir, directory):
    directory = Path(directory)
    lock = directory / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        run = read_json(directory / "run.json")
        if (run["collection_id"] != COLLECTION_ID or run["role"] != "train"
                or source_fingerprints(Path(root)) != run["source_sha256"]
                or file_sha256(Path(data_dir) / "dataset.json") != run["dataset_metadata_sha256"]
                or tree_hashes(directory / "previous_pilot") != run["previous_pilot_sha256"]):
            raise ValueError("Collection data, source or reused evidence changed after creation")
        _, records = load_bundle(data_dir, verify_images=True)
        selected = select_collection_cases(records, run["settings"]["seed"])
        if [r["id"] for r in selected] != run["selected_ids"]:
            raise ValueError("Collection selection changed")
        by_id = {r["id"]: r for r in records}
        for index, identity in enumerate(run["selected_ids"], 1):
            if identity in run["reused_cases"]:
                case_dir = directory / "cases" / f"{index:02d}"
                source = directory / "previous_pilot" / run["reused_cases"][identity]["source_case"]
                expected = tree_hashes(source)
                actual = tree_hashes(case_dir)
                actual.pop("collection_record.json", None)
                if actual != expected:
                    raise ValueError("Reused case bytes changed")
                record_case(case_dir, by_id[identity], run["reused_cases"][identity])
        from .model import load_primary
        from .repair_model import answer_readout, continue_reasoning, generate_reasoning
        from .repair_audit import request_json
        model, processor = load_primary(run["settings"])
        write_json(directory / "model.json", {"model_id": run["settings"]["model_id"], "revision": run["settings"]["revision"],
                   "model_weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters())})
        callbacks = {"reasoning": lambda *a: generate_reasoning(model, processor, *a),
                     "readout": lambda *a: answer_readout(model, processor, *a),
                     "continuation": lambda *a: continue_reasoning(model, processor, *a),
                     "audit": request_json, "review": request_json}
        summary = collect_cases(data_dir, directory, run, records, callbacks)
        return 0 if summary["status"] == "complete" else 2
    finally:
        lock.unlink(missing_ok=True)


def start_collection(root, data_dir, pilot_dir, settings, config):
    root, data_dir, pilot_dir = Path(root).resolve(), Path(data_dir).resolve(), Path(pilot_dir).resolve()
    import psutil
    if psutil.virtual_memory().available < settings["minimum_available_gib"] * 1024**3:
        raise RuntimeError("Available RAM is below the configured minimum")
    metadata, records = load_bundle(data_dir, verify_images=True)
    selected = select_collection_cases(records, settings["seed"])
    previous, hashes, reused = validate_previous(root, data_dir, pilot_dir, settings, config, metadata, records)
    if len(set(r["id"] for r in selected) - set(reused)) != 9:
        raise ValueError("The fixed collection must contain exactly nine new cases")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "outputs" / f"{stamp}_training_collection_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    shutil.copytree(pilot_dir, directory / "previous_pilot")
    if tree_hashes(directory / "previous_pilot") != hashes or tree_hashes(pilot_dir) != hashes:
        raise ValueError("Previous pilot changed while copying evidence")
    run = {"collection_id": COLLECTION_ID, "role": "train", "started_utc": stamp,
           "settings": settings, "config": config, "selection": {"vqa_rad": 6, "slake": 6},
           "selected_ids": [r["id"] for r in selected], "reused_cases": reused, "maximum_new_cases": 9,
           "previous_pilot_folder": str(pilot_dir), "previous_pilot_sha256": hashes,
           "dataset_metadata_sha256": previous["dataset_metadata_sha256"], "source_sha256": source_fingerprints(root),
           "reasoning_instruction": previous["reasoning_instruction"], "readout_instruction": READOUT_INSTRUCTION,
           "verifier_id": PLAIN_SCORING_ID, "audit_validation_policy": AUDIT_VALIDATION_POLICY,
           "limitations": previous.get("limitations", []) + [
               "Question echoes are diagnostic labels, not new matching rules",
               "Three non-repair cases are reused, not regenerated",
               "Ollama model tags do not guarantee immutable weights; record the local model identity separately",
               "Potential candidates still require full admission; no adapters are trained"]}
    write_json(directory / "run.json", run)
    for index, row in enumerate(selected, 1):
        if row["id"] in reused:
            case_dir = directory / "cases" / f"{index:02d}"
            shutil.copytree(directory / "previous_pilot" / reused[row["id"]]["source_case"], case_dir)
            record_case(case_dir, row, reused[row["id"]])
    write_json(directory / "environment.json", environment_snapshot(root))
    print(f"Collection folder: {directory}", flush=True)
    command = [sys.executable, str(root / "evaluate.py"), "--data-dir", str(data_dir), "--collection-worker-dir", str(directory)]
    try:
        resources = supervise(command, {**settings, "timeout_seconds": config["worker_timeout_seconds"]}, directory / "console.log")
        weights = read_json(directory / "model.json")["model_weight_bytes"] if (directory / "model.json").exists() else 0
        resources.update(memory_measurement_plausible=bool(weights and (resources["peak_process_tree_rss_bytes"] or 0) >= .95 * weights),
                         measurement_scope="Python worker process tree; excludes the external Ollama service")
        if not resources["memory_measurement_plausible"]:
            resources["warning"] = "RAM sample requires target-PC verification; do not report as validated peak"
        write_json(directory / "resources.json", resources)
        summary = summarize_collection(directory, run, resources)
    except (Exception, KeyboardInterrupt) as error:
        summary = summarize_collection(directory, run)
        summary.update(status="incomplete", error=safe_error(error))
    if tree_hashes(pilot_dir) != hashes:
        summary.update(status="incomplete", error="Original pilot files changed during collection")
    write_json(directory / "summary.json", summary)
    return summary
