"""Test one saved audit proposal without admitting a training example."""

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from src.audit_experiment import AUDIT_FILES, validate_audit_inputs
from src.benchmarks import file_sha256, image_path, load_bundle
from src.evaluation import source_fingerprints
from src.readout_experiment import EVIDENCE_FILES
from src.repair_audit import safe_hint, validate_medical_audit
from src.repair_model import READOUT_INSTRUCTION, reasoning_steps
from src.repair_pilot import matched_readout, read_pilot_config
from src.runtime import environment_snapshot, read_settings, safe_error, supervise, write_json
from src.training_collection import read_json, tree_hashes


ROOT = Path(__file__).resolve().parent
CHECK_ID = "saved-audit-continuation-1-v1"
SELECTED_ID = "vqa_rad:train:000188"
ASSET_HASHES = {
    "saved_input.json": "dbd40cbe1c0c40eacd725c2458e20bb76fa6f7de8353392ef9ce5a0c2e2d9108",
    "approved_audit.json": "4835ae97a1bb61788c794f27522240fc783967d8a745fb773f9c8f2ea9ceb393",
}
AUDIT_IMAGE_SHA256 = "286b4a376afc7f4075cbd3306f1b45a623ee5436e3a91de259cb7c52a13006a6"


def approved_proposal(asset_dir, initial, steps, reference):
    asset_dir = Path(asset_dir)
    if any(file_sha256(asset_dir / name) != digest for name, digest in ASSET_HASHES.items()):
        raise ValueError("The sealed audit or its input changed; do not edit or substitute it")
    visible = read_json(asset_dir / "saved_input.json")
    audit = read_json(asset_dir / "approved_audit.json")
    if visible["response"] != initial["raw_response"] or visible["steps"] != steps or steps != reasoning_steps(initial["raw_response"]):
        raise ValueError("The saved reasoning or step boundaries differ from the reviewed input")
    legacy = validate_medical_audit(audit["legacy_audit"], steps)
    allowed, reason = safe_hint(legacy["hint"], reference)
    if legacy["action"] != "repair" or legacy["first_error_step"] != 2 or not allowed:
        raise ValueError("The selected audit is not the approved step-2 repair: " + str(reason))
    boundary = steps[1]["start"]
    prefix = initial["raw_response"][:boundary]
    if not prefix or audit["paper_audit"]["SAFE_PREFIX_END"] != {"step_index": 1, "verbatim_ending_span": prefix}:
        raise ValueError("The proposed nonempty prefix differs from the sealed boundary")
    return visible, legacy, prefix


def run_attempt(directory, request, reference, original_suffix, callbacks):
    """Record one generation; callbacks receive no reference answer or audit rationale."""
    directory = Path(directory)
    result = {"status": "complete", "check_id": CHECK_ID, "selected_id": SELECTED_ID,
              "outcome": "started", "continuation_calls_started": 0, "readout_calls_started": 0,
              "new_initial_reasoning_calls": 0, "new_audit_calls": 0, "new_review_calls": 0,
              "candidate_answer_match": None, "training_examples_exported": 0,
              "training_examples_admitted": 0, "optimizer_steps": 0, "adapters_created": False,
              "medical_validity": "not_independently_verified", "automatic_candidate_admission": False,
              "remaining_checks": ["reasoning correctness and readout fidelity", "independent candidate integrity review",
                                   "full learnability admission; not implemented in this pilot"]}
    def save():
        write_json(directory / "summary.json", result)
    # An interrupted or failed attempt cannot silently be repeated in this folder.
    with (directory / "attempt.json").open("x", encoding="utf-8") as stream:
        json.dump({"check_id": CHECK_ID, "selected_id": SELECTED_ID}, stream)
    save()
    try:
        prefix, hint = request["preserved_prefix"], request["minimal_hint"]
        result["continuation_calls_started"] = 1
        save()
        candidate = callbacks["continuation"](prefix, hint, 128)
        write_json(directory / "candidate_reasoning.json", candidate)
        cue = "\nLet me recheck the evidence. " + hint + "\n"
        if (candidate.get("preserved_prefix") != prefix or candidate.get("recovery_cue") != cue
                or candidate.get("assistant_prefill") != prefix + cue
                or candidate.get("complete_response") != prefix + cue + candidate.get("suffix", "")
                or not candidate.get("assistant_prefill_verified") or not candidate.get("input_prefix_unchanged")
                or not candidate.get("retained_prefix_unchanged")):
            result["outcome"] = "rejected_prefix_integrity"
            return result
        result["retained_prefix_unchanged"] = True
        suffix = candidate["suffix"]
        if not suffix.strip() or candidate["hit_token_limit"]:
            result["outcome"] = "rejected_empty_or_truncated_suffix"
            return result
        if suffix.strip() == original_suffix.strip():
            result["outcome"] = "rejected_unchanged_erroneous_suffix"
            return result
        from src.answer_diagnostic import normalize_plain
        sentences = [normalize_plain(step["text"]) for step in reasoning_steps(suffix) if len(step["text"].split()) >= 4]
        if any(count >= 3 for count in Counter(sentences).values()):
            result["outcome"] = "rejected_repetition"
            return result
        readout_request = {"question": request["question"], "reasoning_instruction": request["reasoning_instruction"],
                           "image_sha256": request["image_sha256"], "assistant_reasoning": candidate["complete_response"],
                           "readout_instruction": READOUT_INSTRUCTION, "max_new_tokens": 32}
        write_json(directory / "candidate_readout_request.json", readout_request)
        result["readout_calls_started"] = 1
        save()
        answer = callbacks["readout"](candidate["complete_response"], 32)
        write_json(directory / "candidate_readout.json", answer)
        matched = matched_readout(answer, reference)
        result["candidate_answer_match"] = matched
        result["outcome"] = "answer_match_pending_review" if matched else "candidate_answer_mismatch"
        write_json(directory / "verification.json", {"reference": reference, "candidate_answer_match": matched,
                   "scoring_id": "plain-whole-answer-em-v1", "scope": "training-only continuation diagnostic; not benchmark evaluation"})
        return result
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="incomplete", outcome="infrastructure_error", error=safe_error(error))
        return result
    finally:
        save()


def worker(root, directory):
    from PIL import Image
    from src.model import load_primary
    from src.repair_model import answer_readout, continue_reasoning

    directory = Path(directory)
    lock = directory / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        run = read_json(directory / "run.json")
        if (run["check_id"] != CHECK_ID or run["selected_id"] != SELECTED_ID
                or source_fingerprints(root) != run["source_sha256"]
                or file_sha256(__file__) != run["check_script_sha256"]
                or tree_hashes(directory / "evidence") != run["evidence_sha256"]
                or tree_hashes(directory / "approved") != run["approved_sha256"]
                or file_sha256(directory / "continuation_request.json") != run["request_sha256"]):
            raise ValueError("The continuation check inputs or implementation changed")
        if (directory / "attempt.json").exists():
            raise ValueError("This continuation attempt already started; no automatic rerun")
        request = read_json(directory / "continuation_request.json")
        row = run["record"]
        data_dir = Path(run["data_dir"])
        if file_sha256(data_dir / "dataset.json") != run["dataset_metadata_sha256"] or row["role"] != "train":
            raise ValueError("Training data identity changed")
        initial = read_json(directory / "evidence/initial_reasoning.json")
        steps = read_json(directory / "evidence/reasoning_steps.json")
        visible, audit, prefix = approved_proposal(directory / "approved", initial, steps, row["answer"])
        initial_request = read_json(directory / "evidence/initial_request.json")
        expected_request = {"question": visible["question"], "reasoning_instruction": initial_request["reasoning_instruction"],
                            "image_sha256": row["image_sha256"], "preserved_prefix": prefix,
                            "minimal_hint": audit["hint"], "max_new_tokens": 128}
        if (request != expected_request or row["id"] != SELECTED_ID or row["question"] != visible["question"]
                or read_json(directory / "evidence/verification.json")["reference"] != row["answer"]):
            raise ValueError("Only the selected case, original instruction, sealed prefix and hint may be used")
        from src.benchmarks import decoded_sha256
        with Image.open(image_path(data_dir, row["image_path"])) as original:
            image = original.convert("RGB")
        if decoded_sha256(image) != row["image_sha256"]:
            raise ValueError("The selected training image changed")
        model, processor = load_primary(run["settings"])
        write_json(directory / "model.json", {"model_id": run["settings"]["model_id"], "revision": run["settings"]["revision"],
                   "model_weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters())})
        question, instruction = request["question"], request["reasoning_instruction"]
        callbacks = {"continuation": lambda p, h, n: continue_reasoning(model, processor, image, question, instruction, p, h, n),
                     "readout": lambda text, n: answer_readout(model, processor, image, question, instruction, text, n)}
        result = run_attempt(directory, request, row["answer"], initial["raw_response"][steps[1]["start"]:], callbacks)
        return 0 if result["status"] == "complete" else 2
    except (Exception, KeyboardInterrupt) as error:
        write_json(directory / "worker_error.json", {"error": safe_error(error)})
        return 2
    finally:
        lock.unlink(missing_ok=True)


def start_check(root, data_dir, collection_dir, settings, config):
    root, data_dir, collection_dir = Path(root).resolve(), Path(data_dir).resolve(), Path(collection_dir).resolve()
    output = root / "outputs/continuation_case_v1"
    if output.exists():
        raise ValueError(f"The one-case output already exists: {output}. Review it before any new attempt")
    previous, hashes, _ = validate_audit_inputs(root, data_dir, collection_dir, settings, config)
    if config["suffix_max_new_tokens"] != 128 or config["readout_max_new_tokens"] != 32:
        raise ValueError("The approved continuation/readout budgets are 128/32 tokens")
    index = previous["selected_ids"].index(SELECTED_ID) + 1
    source = collection_dir / "cases" / f"{index:02d}"
    _, records = load_bundle(data_dir)
    row = next(record for record in records if record["id"] == SELECTED_ID)
    if row["role"] != "train":
        raise ValueError("Only the selected training case may be continued")
    initial, steps = read_json(source / "initial_reasoning.json"), read_json(source / "reasoning_steps.json")
    assets = root / "configs/continuation_case"
    visible, audit, prefix = approved_proposal(assets, initial, steps, row["answer"])
    original = read_json(source / "audit_request.json")
    blocks = original["messages"][1]["content"]
    saved_visible = json.loads(next(block["text"] for block in blocks if block["type"] == "text"))
    encoded = next(block["image_url"]["url"] for block in blocks if block["type"] == "image_url")
    if saved_visible != visible or hashlib.sha256(base64.b64decode(encoded.split(",", 1)[1], validate=True)).hexdigest() != AUDIT_IMAGE_SHA256:
        raise ValueError("The original auditor image or visible input differs from the independently reviewed packet")
    if row["question"] != visible["question"]:
        raise ValueError("The selected question differs from the reviewed case")
    import psutil
    if psutil.virtual_memory().available < settings["minimum_available_gib"] * 1024**3:
        raise RuntimeError("Available RAM is below the configured minimum")
    output.mkdir(parents=True)
    for label in ("evidence", "approved"):
        (output / label).mkdir()
    for name in EVIDENCE_FILES + AUDIT_FILES:
        shutil.copy2(source / name, output / "evidence" / name)
        if file_sha256(output / "evidence" / name) != hashes[f"cases/{index:02d}/{name}"]:
            raise ValueError("Original case changed while copying evidence")
    for name in ASSET_HASHES:
        shutil.copy2(assets / name, output / "approved" / name)
    request = {"question": row["question"], "reasoning_instruction": previous["reasoning_instruction"],
               "image_sha256": row["image_sha256"], "preserved_prefix": prefix, "minimal_hint": audit["hint"],
               "max_new_tokens": 128}
    write_json(output / "continuation_request.json", request)
    run = {"check_id": CHECK_ID, "started_utc": datetime.now(timezone.utc).isoformat(), "selected_id": SELECTED_ID,
           "role": "train", "settings": settings, "source_config": config, "record": row,
           "source_collection": str(collection_dir), "source_collection_sha256": hashes,
           "dataset_metadata_sha256": file_sha256(data_dir / "dataset.json"), "data_dir": str(data_dir),
           "source_sha256": source_fingerprints(root), "check_script_sha256": file_sha256(__file__),
           "evidence_sha256": tree_hashes(output / "evidence"), "approved_sha256": tree_hashes(output / "approved"),
           "request_sha256": file_sha256(output / "continuation_request.json"),
           "auditor_source": "sealed independent batch case_05; no new auditor call", "maximum_continuations": 1,
           "maximum_readouts": 1, "training_admission": "disabled; diagnostic only"}
    write_json(output / "run.json", run)
    write_json(output / "environment.json", environment_snapshot(root))
    if tree_hashes(collection_dir) != hashes:
        raise ValueError("Original collection changed before worker launch")
    print(f"Continuation output folder: {output}", flush=True)
    command = [sys.executable, str(root / "continue_case.py"), "--worker-dir", str(output)]
    resources = supervise(command, {**settings, "timeout_seconds": config["worker_timeout_seconds"]}, output / "console.log")
    resources["measurement_scope"] = "Python worker process tree; no external auditor is called"
    weights = read_json(output / "model.json")["model_weight_bytes"] if (output / "model.json").exists() else 0
    resources["memory_measurement_plausible"] = bool(weights and (resources["peak_process_tree_rss_bytes"] or 0) >= .95 * weights)
    write_json(output / "resources.json", resources)
    summary = read_json(output / "summary.json") if (output / "summary.json").exists() else {
        "status": "incomplete", "check_id": CHECK_ID, "selected_id": SELECTED_ID,
        "outcome": "worker_stopped_before_result", "training_examples_admitted": 0, "optimizer_steps": 0}
    if resources["worker_exit_code"] != 0 or resources["limit_reached"] or tree_hashes(collection_dir) != hashes:
        summary["status"] = "incomplete"
    if (output / "worker_error.json").exists():
        summary["worker_error"] = read_json(output / "worker_error.json")["error"]
    summary.update(output_folder=str(output), worker=resources)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run the approved one-case saved-audit continuation test")
    parser.add_argument("--collection-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/benchmarks")
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker_dir:
        return worker(ROOT, args.worker_dir.resolve())
    if args.collection_dir is None:
        parser.error("--collection-dir is required")
    try:
        result = start_check(ROOT, args.data_dir, args.collection_dir, read_settings(ROOT / "settings.json"),
                             read_pilot_config(ROOT / "configs/repair_pilot.json"))
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "complete" else 2
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
