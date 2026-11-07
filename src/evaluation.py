from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import uuid
import sys

from PIL import Image

from .benchmarks import file_sha256, image_path, load_bundle, read_jsonl
from .runtime import environment_snapshot, safe_error, supervise, write_json
from .scoring import SCORING_ID, score_answer, summarize


def read_protocol(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("scoring_id") != SCORING_ID or not value.get("prompt", "").strip():
        raise ValueError("A nonempty prompt and the supported scoring_id are required")
    for key, low, high in (("seed", 0, 2**32 - 1), ("max_new_tokens", 1, 512),
                           ("evaluation_timeout_seconds", 1, 86400)):
        if type(value.get(key)) is not int or not low <= value[key] <= high:
            raise ValueError(f"Invalid protocol setting: {key}")
    fraction = value.get("rad_feedback_fraction")
    if type(fraction) not in (int, float) or not 0 < fraction < 1:
        raise ValueError("Invalid rad_feedback_fraction")
    if value.get("training_policy") != "exclude_exact_test_and_feedback_images":
        raise ValueError("Unsupported training policy")
    return value


def select_records(records, dataset, role, seed, limit=None):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("Limit must be a positive integer")
    selected = [r for r in records if r["dataset"] == dataset and r["role"] == role]
    selected.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
    return selected if limit is None else selected[:limit]


def source_fingerprints(root):
    paths = sorted((root / "src").glob("*.py")) + [root / "evaluate.py"]
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in paths}


def recorded_source_hashes(signature):
    return {name.replace("\\", "/"): digest for name, digest in signature["code_sha256"].items()}


def create_run(root, data_dir, dataset, role, limit, settings, protocol, resume=None):
    metadata, records = load_bundle(data_dir, verify_images=True)
    if metadata["protocol"] != protocol:
        raise ValueError("Data preparation protocol differs from the current protocol")
    chosen = select_records(records, dataset, role, protocol["seed"], limit)
    if not chosen:
        raise ValueError("No records match this selection")
    environment = environment_snapshot(root)
    signature = {"benchmark": dataset, "role": role, "limit": limit,
                 "scope": "full" if limit is None else "subset",
                 "selected_ids": [r["id"] for r in chosen],
                 "dataset_metadata_sha256": file_sha256(Path(data_dir) / "dataset.json"),
                 "settings": settings, "protocol": protocol,
                 "code_sha256": source_fingerprints(root),
                 "packages": environment["packages"], "python": environment["python"]}
    if resume:
        directory = Path(resume).resolve()
        existing = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        if existing["signature"] != signature:
            raise ValueError("Resume rejected: selection, data, code, model or configuration changed")
        if (directory / "run.lock").exists():
            raise ValueError("Run is locked. Confirm its process has stopped before removing run.lock")
        return directory
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "outputs" / f"{stamp}_{dataset}_{role}_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    write_json(directory / "run.json", {"created_utc": stamp, "signature": signature,
               "purpose": "development feedback" if role == "feedback" else "held-out evaluation",
               "sources": metadata["sources"]})
    write_json(directory / "environment.json", environment)
    write_json(directory / "dependencies.json", dict(sorted(
        (distribution.metadata["Name"], distribution.version)
        for distribution in importlib.metadata.distributions() if distribution.metadata["Name"])))
    return directory


def prediction_journal(path, recover=False):
    path = Path(path)
    if not path.exists():
        return []
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        # An interrupted append can leave only the last record incomplete.
        try:
            json.loads(content.rsplit(b"\n", 1)[-1])
        except (ValueError, UnicodeDecodeError):
            if not recover:
                raise ValueError("Interrupted final journal record; use --resume to recover it")
            backup = path.with_name(path.name + ".interrupted-" + uuid.uuid4().hex[:8])
            backup.write_bytes(content)
            cut = content.rfind(b"\n") + 1
            path.write_bytes(content[:cut])
        else:
            if recover:
                with path.open("ab") as stream:
                    stream.write(b"\n")
    return list(read_jsonl(path))


def indexed_predictions(predictions, selected_ids):
    indexed = {}
    allowed = set(selected_ids)
    for row in predictions:
        identity = row.get("id")
        if identity not in allowed or identity in indexed:
            raise ValueError("Unknown or duplicate prediction ID")
        if row.get("status") not in ("generated", "generation_error"):
            raise ValueError("Invalid prediction status")
        if row["status"] == "generated":
            if not isinstance(row.get("raw_response"), str):
                raise ValueError("Generated predictions require a raw response")
            seconds = row.get("generation_seconds")
            if type(seconds) not in (int, float) or not 0 <= seconds < float("inf"):
                raise ValueError("Invalid generation duration")
            if type(row.get("hit_token_limit")) is not bool:
                raise ValueError("Generated predictions require a token-limit flag")
        indexed[identity] = row
    return indexed


def score_run(directory, data_dir):
    directory, data_dir = Path(directory), Path(data_dir)
    run = json.loads((directory / "run.json").read_text(encoding="utf-8"))
    signature = run["signature"]
    if signature["protocol"]["scoring_id"] != SCORING_ID:
        raise ValueError("Scoring protocol changed")
    current_scorer = file_sha256(Path(__file__).with_name("scoring.py"))
    recorded_scorer = recorded_source_hashes(signature).get("src/scoring.py")
    if recorded_scorer and recorded_scorer != current_scorer:
        raise ValueError("Scoring implementation changed; use the original code for this run")
    if signature["dataset_metadata_sha256"] != file_sha256(data_dir / "dataset.json"):
        raise ValueError("Dataset identity changed")
    _, records = load_bundle(data_dir)
    references = {r["id"]: r for r in records}
    ids = signature["selected_ids"]
    predictions = indexed_predictions(prediction_journal(directory / "predictions.jsonl"), ids)
    scored = []
    for identity in ids:
        reference = references[identity]
        row = {"id": identity, "dataset": reference["dataset"],
               "answer_type": reference["answer_type"], "reference": reference["answer"],
               **predictions.get(identity, {"status": "missing"})}
        if row["status"] == "generated":
            row.update(score_answer(row["raw_response"], reference["answer"]))
        scored.append(row)
    metrics = {"benchmark": signature["benchmark"], "role": signature["role"],
               "scope": signature["scope"], "scoring_id": SCORING_ID,
               "scorer_sha256": current_scorer,
               "summary": summarize(scored),
               "by_answer_type": {kind: summarize([r for r in scored if r["answer_type"] == kind])
                                  for kind in ("open", "closed")}}
    from .benchmarks import write_jsonl
    write_jsonl(directory / "scored.jsonl", scored)
    write_json(directory / "metrics.json", metrics)
    return metrics


def execute_run(root, directory, data_dir, settings, protocol):
    root, directory, data_dir = Path(root), Path(directory), Path(data_dir)
    os.environ.update(PYTHONUNBUFFERED="1", HF_HUB_DISABLE_XET="1")
    command = [sys.executable, str(root / "evaluate.py"), "--data-dir", str(data_dir),
               "--worker-dir", str(directory)]
    limits = {**settings, "timeout_seconds": protocol["evaluation_timeout_seconds"]}
    attempt = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    resources = supervise(command, limits, directory / f"console_{attempt}.log")
    resources.update(attempt_id=attempt, measurement_scope="current worker attempt; not cumulative")
    model_file = directory / "model.json"
    weight_bytes = json.loads(model_file.read_text())["model_weight_bytes"] if model_file.exists() else 0
    measured = resources["peak_process_tree_rss_bytes"] or 0
    resources["memory_measurement_plausible"] = bool(weight_bytes and measured >= .95 * weight_bytes)
    if not resources["memory_measurement_plausible"]:
        resources["warning"] = "RAM measurement requires target-PC verification; do not report this as a validated peak"
    write_json(directory / "resources.json", resources)
    with (directory / "resource_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(resources, allow_nan=False) + "\n")
    result = score_run(directory, data_dir)
    result["worker"] = resources
    return result


def model_input(image, question, instruction):
    # Keep references and answer-type metadata outside the generation interface.
    return image, question + "\n" + instruction


def evaluation_worker(directory, data_dir, load_model=None, generate=None):
    from .model import generate_response, load_primary

    directory = Path(directory)
    lock = directory / "run.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        run = json.loads((directory / "run.json").read_text(encoding="utf-8"))["signature"]
        if run["dataset_metadata_sha256"] != file_sha256(Path(data_dir) / "dataset.json"):
            raise ValueError("Dataset identity changed")
        _, rows = load_bundle(data_dir, verify_images=True)
        by_id = {r["id"]: r for r in rows}
        journal = directory / "predictions.jsonl"
        previous = indexed_predictions(prediction_journal(journal, recover=True), run["selected_ids"])
        pending = [i for i in run["selected_ids"] if i not in previous]
        if pending:
            model, processor = (load_model or load_primary)(run["settings"])
            weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
            write_json(directory / "model.json", {"model_weight_bytes": weight_bytes,
                       "model_id": run["settings"]["model_id"], "revision": run["settings"]["revision"]})
            with journal.open("a", encoding="utf-8", newline="\n") as stream:
                for index, identity in enumerate(pending, 1):
                    reference = by_id[identity]
                    try:
                        with Image.open(image_path(data_dir, reference["image_path"])) as original:
                            image = original.convert("RGB")
                        image, prompt = model_input(image, reference["question"], run["protocol"]["prompt"])
                        result = (generate or generate_response)(model, processor, image, prompt,
                                                run["protocol"]["max_new_tokens"])
                        prediction = {"id": identity, "status": "generated",
                                      **{key: result[key] for key in ("raw_response", "generated_tokens",
                                                                      "generation_seconds", "hit_token_limit")}}
                    except Exception as error:
                        prediction = {"id": identity, "status": "generation_error", "error": safe_error(error)}
                    stream.write(json.dumps(prediction, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())
                    print(f"{index}/{len(pending)} {identity}: {prediction['status']}", flush=True)
        metrics = score_run(directory, data_dir)
        return 0 if metrics["summary"]["status"] == "complete" else 2
    finally:
        lock.unlink(missing_ok=True)
