"""Compare a text-only readout with saved training-case readouts."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import uuid

from .answer_diagnostic import PLAIN_SCORING_ID
from .benchmarks import file_sha256, load_bundle
from .evaluation import source_fingerprints
from .repair_model import generate_chat
from .repair_pilot import matched_readout
from .runtime import environment_snapshot, safe_error, supervise, write_json
from .training_collection import (COLLECTION_ID, classify_readout, read_json,
                                  select_collection_cases, tree_hashes)


EXPERIMENT_ID = "saved-reasoning-readout-v1"
TOKEN_LIMIT = 32
INSTRUCTION = (
    "Extract the answer already stated in the saved response to the question below. "
    "Use only the saved response. Do not solve the question again, add information, or correct the response. "
    "Return only a word, number, or short phrase. For a yes/no question return only yes or no "
    "if the response states one. If the response does not state an answer, return unresolved.")
EVIDENCE_FILES = ("case.json", "collection_record.json", "initial_request.json", "initial_reasoning.json",
                  "initial_readout.json", "verification.json")


def readout_request(question, reasoning):
    if not isinstance(question, str) or not question.strip() or not isinstance(reasoning, str) or not reasoning.strip():
        raise ValueError("A readout requires a recorded question and nonempty saved response")
    text = INSTRUCTION + "\n\nQuestion: " + question + "\n\nSaved response: " + reasoning + "\n\nAnswer:"
    return {"messages": [{"role": "user", "content": [{"type": "text", "text": text}]}],
            "max_new_tokens": TOKEN_LIMIT}


def validate_collection(root, data_dir, collection_dir, settings):
    collection_dir = Path(collection_dir)
    if (collection_dir / "run.lock").exists():
        raise ValueError("Wait until the source collection worker has stopped")
    before = tree_hashes(collection_dir)
    run, summary = read_json(collection_dir / "run.json"), read_json(collection_dir / "summary.json")
    if (run.get("collection_id") != COLLECTION_ID or run.get("role") != "train"
            or summary.get("status") != "complete" or summary.get("recorded_count") != 12
            or summary.get("new_case_count") != 9 or summary.get("reused_count") != 3):
        raise ValueError("Use the completed twelve-case training collection")
    if run.get("settings") != settings or run.get("config", {}).get("readout_max_new_tokens") != TOKEN_LIMIT:
        raise ValueError("The primary-model settings and 32-token readout budget must stay unchanged")
    if run.get("verifier_id") != PLAIN_SCORING_ID:
        raise ValueError("The saved training verifier differs from this diagnostic")
    current = source_fingerprints(Path(root))
    saved = {name.replace("\\", "/"): digest for name, digest in run["source_sha256"].items()}
    for name in ("src/model.py", "src/repair_model.py", "src/scoring.py", "src/answer_diagnostic.py",
                 "src/repair_pilot.py", "src/training_collection.py", "src/benchmarks.py"):
        if saved.get(name) != current[name]:
            raise ValueError(f"A frozen model or verification implementation changed: {name}")
    if file_sha256(Path(data_dir) / "dataset.json") != run["dataset_metadata_sha256"]:
        raise ValueError("The source collection used a different data bundle")
    _, records = load_bundle(data_dir)
    selected = select_collection_cases(records, settings["seed"])
    if [row["id"] for row in selected] != run.get("selected_ids"):
        raise ValueError("The twelve fixed training IDs do not match the prepared bundle")
    inputs = []
    for index, row in enumerate(selected, 1):
        folder = collection_dir / "cases" / f"{index:02d}"
        evidence = {name: read_json(folder / name) for name in EVIDENCE_FILES}
        request, verification = evidence["initial_request.json"], evidence["verification.json"]
        case, record = evidence["case.json"], evidence["collection_record.json"]
        reasoning, answer = evidence["initial_reasoning.json"], evidence["initial_readout.json"]
        if (case.get("id") != row["id"] or record.get("id") != row["id"]
                or case.get("role") != "train" or record.get("role") != "train"
                or case.get("optimizer_steps") != 0 or case.get("training_examples_exported") != 0
                or request.get("question") != row["question"] or request.get("image_sha256") != row["image_sha256"]
                or request.get("reasoning_instruction") != run["reasoning_instruction"]
                or verification.get("reference") != row["answer"] or verification.get("scoring_id") != PLAIN_SCORING_ID
                or verification.get("initial_answer_match") != matched_readout(answer, row["answer"])):
            raise ValueError("Saved case identity, question, reference or original verification disagrees")
        request = readout_request(row["question"], reasoning["raw_response"])
        inputs.append({"id": row["id"], "request": request})
    if tree_hashes(collection_dir) != before:
        raise ValueError("The collection changed during validation")
    return run, before, inputs


def summarize_experiment(directory, run, resources=None):
    directory = Path(directory)
    comparisons = []
    for index, identity in enumerate(run["selected_ids"], 1):
        evidence = directory / "evidence" / f"{index:02d}"
        question = read_json(evidence / "initial_request.json")["question"]
        reference = read_json(evidence / "verification.json")["reference"]
        original = read_json(evidence / "initial_readout.json")
        path = directory / "readouts" / f"{index:02d}.json"
        trial = read_json(path) if path.exists() else None
        comparisons.append({"id": identity, "dataset": read_json(evidence / "case.json")["dataset"],
            "question": question, "reference": reference,
            "saved_reasoning": read_json(evidence / "initial_reasoning.json")["raw_response"],
            "original_readout": original, "trial_readout": trial,
            "original_match": matched_readout(original, reference),
            "trial_match": matched_readout(trial, reference) if trial else None,
            "original_category": classify_readout(original, question, reference),
            "trial_category": classify_readout(trial, question, reference),
            "readout_fidelity": "requires_review_against_saved_reasoning",
            "training_admission": "not_assessed"})
    count = sum(row["trial_readout"] is not None for row in comparisons)
    complete = count == 12 and not (directory / "worker_error.json").exists()
    if resources and (resources["worker_exit_code"] != 0 or resources["limit_reached"]):
        complete = False
    def metrics(rows):
        done = sum(row["trial_readout"] is not None for row in rows)
        original = sum(row["original_match"] for row in rows)
        matches = sum(row["trial_match"] is True for row in rows)
        return {"selected_count": len(rows), "trial_readout_count": done,
                "original_match_count": original, "trial_match_count": matches,
                "original_whole_answer_em_pct": 100 * original / len(rows),
                "trial_whole_answer_em_pct": 100 * matches / len(rows) if complete and done == len(rows) else None,
                "gained_matches": sum(row["trial_match"] is True and not row["original_match"] for row in rows),
                "lost_matches": sum(row["trial_match"] is False and row["original_match"] for row in rows)}
    summary = {"status": "complete" if complete else "incomplete", "experiment_id": EXPERIMENT_ID,
        "output_folder": str(directory), "purpose": "exploratory saved-reasoning readout comparison; not benchmark evaluation",
        "scoring_id": PLAIN_SCORING_ID, "metrics": metrics(comparisons),
        "by_dataset": {dataset: metrics([r for r in comparisons if r["dataset"] == dataset]) for dataset in ("vqa_rad", "slake")},
        "new_readout_calls_recorded": count, "new_reasoning_calls": 0, "new_auditor_calls": 0,
        "training_examples_exported": 0, "optimizer_steps": 0, "adapters_created": False,
        "medical_validity": "not_independently_verified", "readout_fidelity": "requires_review",
        "automatic_candidate_admission": False}
    if resources:
        summary["worker"] = resources
    if (directory / "worker_error.json").exists():
        summary["error"] = read_json(directory / "worker_error.json")["error"]
    write_json(directory / "comparison.json", {**summary, "cases": comparisons})
    write_json(directory / "summary.json", summary)
    lines = ["# Saved-reasoning readout comparison", "", "Exploratory training-case diagnostic; original results remain unchanged.", "",
             "A matching answer is not proof of faithful extraction, medically correct reasoning or training admission.", "",
             "| Case | Original readout | Trial readout | Original match | Trial match |", "|---|---|---|---|---|"]
    def cell(text):
        return str(text).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    for row in comparisons:
        trial = row["trial_readout"]["raw_response"] if row["trial_readout"] else "not recorded"
        lines.append(f"| {cell(row['id'])} | {cell(row['original_readout']['raw_response'])} | {cell(trial)} | {row['original_match']} | {row['trial_match']} |")
    (directory / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def readout_worker(root, directory):
    directory = Path(directory)
    lock = directory / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        run = read_json(directory / "run.json")
        if (run["experiment_id"] != EXPERIMENT_ID or run["source_sha256"] != source_fingerprints(Path(root))
                or file_sha256(directory / "inputs.json") != run["inputs_sha256"]
                or tree_hashes(directory / "evidence") != run["evidence_sha256"]):
            raise ValueError("Readout experiment code or saved inputs changed after creation")
        inputs = read_json(directory / "inputs.json")
        if len(inputs) != 12 or [item["id"] for item in inputs] != run["selected_ids"]:
            raise ValueError("The readout workload must contain the twelve fixed IDs")
        for index, item in enumerate(inputs, 1):
            evidence = directory / "evidence" / f"{index:02d}"
            expected = readout_request(read_json(evidence / "initial_request.json")["question"],
                                       read_json(evidence / "initial_reasoning.json")["raw_response"])
            if set(item) != {"id", "request"} or item["request"] != expected:
                raise ValueError("Readout request differs from the reference-free text-only method")
        if (directory / "readouts").exists() and any((directory / "readouts").iterdir()):
            raise ValueError("Readout results already exist; do not rerun this worker")
        from .model import load_primary
        model, processor = load_primary(run["settings"])
        write_json(directory / "model.json", {"model_id": run["settings"]["model_id"], "revision": run["settings"]["revision"],
                   "model_weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters())})
        (directory / "readouts").mkdir(exist_ok=True)
        for index, item in enumerate(inputs, 1):
            request = item["request"]
            response = generate_chat(model, processor, request["messages"], request["max_new_tokens"])
            if (not isinstance(response.get("raw_response"), str) or type(response.get("hit_token_limit")) is not bool
                    or type(response.get("generated_tokens")) is not int or not 0 <= response["generated_tokens"] <= TOKEN_LIMIT):
                raise ValueError("Invalid readout generation record")
            write_json(directory / "readouts" / f"{index:02d}.json", response)
            print(f"{index}/12 {item['id']}: text-only readout recorded", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        write_json(directory / "worker_error.json", {"error": safe_error(error)})
        return 2
    finally:
        lock.unlink(missing_ok=True)


def start_readout_experiment(root, data_dir, collection_dir, settings):
    root, collection_dir = Path(root).resolve(), Path(collection_dir).resolve()
    import psutil
    if psutil.virtual_memory().available < settings["minimum_available_gib"] * 1024**3:
        raise RuntimeError("Available RAM is below the configured minimum")
    previous, hashes, inputs = validate_collection(root, data_dir, collection_dir, settings)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "outputs" / f"{stamp}_readout_comparison_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    for name in ("run.json", "summary.json"):
        shutil.copy2(collection_dir / name, directory / ("original_" + name))
    for index in range(1, 13):
        folder = directory / "evidence" / f"{index:02d}"
        folder.mkdir(parents=True)
        for name in EVIDENCE_FILES:
            shutil.copy2(collection_dir / "cases" / f"{index:02d}" / name, folder / name)
            relative = f"cases/{index:02d}/{name}"
            if file_sha256(folder / name) != hashes[relative]:
                raise ValueError("A saved case changed while evidence was copied")
    write_json(directory / "inputs.json", inputs)
    run = {"experiment_id": EXPERIMENT_ID, "role": "train", "started_utc": stamp, "settings": settings,
           "selected_ids": previous["selected_ids"], "readout_max_new_tokens": TOKEN_LIMIT,
           "source_collection": str(collection_dir), "source_collection_sha256": hashes,
           "source_sha256": source_fingerprints(root), "inputs_sha256": file_sha256(directory / "inputs.json"),
           "evidence_sha256": tree_hashes(directory / "evidence"), "readout_instruction": INSTRUCTION,
           "scoring_id": PLAIN_SCORING_ID, "limitations": [
               "Single-turn text-only extraction replaces the old multimodal conversation readout for this diagnostic only",
               "Exploratory method selected after inspecting these twelve training cases",
               "A reference match does not establish extraction fidelity or clinically valid reasoning",
               "No candidate admission, new reasoning, auditor calls, training or final-test evaluation"]}
    write_json(directory / "run.json", run)
    write_json(directory / "environment.json", environment_snapshot(root))
    if tree_hashes(collection_dir) != hashes:
        raise ValueError("Original collection changed before worker launch")
    print(f"Readout comparison folder: {directory}", flush=True)
    command = [sys.executable, str(root / "evaluate.py"), "--readout-worker-dir", str(directory)]
    try:
        resources = supervise(command, settings, directory / "console.log")
        weights = read_json(directory / "model.json")["model_weight_bytes"] if (directory / "model.json").exists() else 0
        resources["memory_measurement_plausible"] = bool(weights and (resources["peak_process_tree_rss_bytes"] or 0) >= .95 * weights)
        resources["measurement_scope"] = "Python worker process tree; no external auditor is called"
        write_json(directory / "resources.json", resources)
        summary = summarize_experiment(directory, run, resources)
    except (Exception, KeyboardInterrupt) as error:
        write_json(directory / "worker_error.json", {"error": safe_error(error)})
        summary = summarize_experiment(directory, run)
    if tree_hashes(collection_dir) != hashes:
        summary.update(status="incomplete", error="Original collection changed during the experiment")
        write_json(directory / "summary.json", summary)
        report = read_json(directory / "comparison.json")
        write_json(directory / "comparison.json", {**report, **summary})
    return summary
