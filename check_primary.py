"""Check the installed 7B model on one saved training image through Ollama."""

import argparse
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.request

from continue_case import ASSET_HASHES, AUDIT_IMAGE_SHA256, ROOT, SELECTED_ID
from src.benchmarks import file_sha256
from src.runtime import safe_error, write_json
from src.scoring import score_answer
from src.training_collection import read_json, tree_hashes


MODEL = "qwen2.5vl:7b"
CHECK_ID = "larger-primary-single-image-v1"
EXTENDED_CHECK_ID = "larger-primary-output-budget-v1"
CONCISE_CHECK_ID = "larger-primary-concise-prompt-v1"
CONCISE_INSTRUCTION = ("Give at most three brief, image-grounded reasoning steps. "
                       "Then provide a short final answer inside \\boxed{}.")
BASE_URL = "http://127.0.0.1:11434"
BASELINE_FILES = ("request.json", "response.json", "run.json", "summary.json", "model_details.json")


def local_json(route, payload=None, timeout=20):
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args):
            return None
    request = urllib.request.Request(BASE_URL + route,
        data=None if payload is None else json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=timeout) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("Local response exceeds the size limit")
    return json.loads(raw.decode("utf-8"))


def prepare_case(collection_dir):
    collection_dir = Path(collection_dir)
    if (collection_dir / "run.lock").exists():
        raise ValueError("The original training collection is still running")
    run, summary = read_json(collection_dir / "run.json"), read_json(collection_dir / "summary.json")
    if (run.get("collection_id") != "training-collection-12-v1" or run.get("role") != "train"
            or summary.get("status") != "complete" or len(run.get("selected_ids", [])) != 12):
        raise ValueError("Use the completed original twelve-case training collection")
    index = run["selected_ids"].index(SELECTED_ID) + 1
    folder = collection_dir / "cases" / f"{index:02d}"
    case, initial = read_json(folder / "case.json"), read_json(folder / "initial_request.json")
    if case.get("id") != SELECTED_ID or case.get("role") != "train":
        raise ValueError("The selected record must be the approved training case")
    assets = ROOT / "configs/continuation_case"
    if file_sha256(assets / "saved_input.json") != ASSET_HASHES["saved_input.json"]:
        raise ValueError("The fixed input packet changed")
    visible = read_json(assets / "saved_input.json")
    original = read_json(folder / "audit_request.json")
    blocks = original["messages"][1]["content"]
    saved_visible = json.loads(next(b["text"] for b in blocks if b["type"] == "text"))
    image_url = next(b["image_url"]["url"] for b in blocks if b["type"] == "image_url")
    header, image = image_url.split(",", 1)
    if (saved_visible != visible or initial["question"] != visible["question"]
            or initial["reasoning_instruction"] != run["reasoning_instruction"]
            or header != "data:image/png;base64"
            or hashlib.sha256(base64.b64decode(image, validate=True)).hexdigest() != AUDIT_IMAGE_SHA256):
        raise ValueError("The saved image, question or original prompt differs from the approved case")
    payload = {"model": MODEL, "messages": [{"role": "user",
        "content": initial["question"] + "\n" + initial["reasoning_instruction"], "images": [image]}],
        "stream": False, "keep_alive": "2m",
        "options": {"temperature": 0, "seed": 7, "num_predict": 128, "num_ctx": 4096,
                    "num_thread": 4, "num_gpu": 0}}
    return folder, payload


def validate_baseline(directory, expected_request, collection_hashes, *, token_budget=128):
    directory = Path(directory)
    if (directory / "run.lock").exists():
        raise ValueError(f"Wait until the {token_budget}-token baseline has stopped")
    before = tree_hashes(directory)
    run, summary = read_json(directory / "run.json"), read_json(directory / "summary.json")
    response = read_json(directory / "response.json")
    expected_check = EXTENDED_CHECK_ID if token_budget == 512 else CHECK_ID
    if (read_json(directory / "request.json") != expected_request
            or run.get("check_id") != expected_check or summary.get("check_id") != expected_check
            or run.get("selected_id") != SELECTED_ID or summary.get("selected_id") != SELECTED_ID
            or run.get("role") != "train" or summary.get("status") != "complete"
            or summary.get("generation_requests_started") != 1
            or run.get("source_collection_sha256") != collection_hashes
            or response.get("model") != MODEL or response.get("done") is not True
            or response.get("done_reason") != "length" or response.get("eval_count") != token_budget
            or summary.get("raw_response") != response.get("message", {}).get("content")
            or summary.get("model_digest") != run.get("model_digest")
            or not run.get("model_digest") or not run.get("ollama_version")):
        raise ValueError(f"Use the completed same-case, same-request truncated {token_budget}-token baseline")
    if tree_hashes(directory) != before:
        raise ValueError("The source baseline changed during validation")
    return run, before


def run_check(root, collection_dir, transport=local_json, available_ram=None, *, max_new_tokens=128, baseline_dir=None, concise_prompt=False):
    if max_new_tokens not in (128, 512) or (max_new_tokens == 512) != (baseline_dir is not None):
        raise ValueError("The 512-token experiment requires its 128-token baseline; other budgets are not approved")
    if concise_prompt and max_new_tokens != 512:
        raise ValueError("The concise-prompt test requires 512 tokens and its original-prompt 512-token baseline")
    check_id = CONCISE_CHECK_ID if concise_prompt else (EXTENDED_CHECK_ID if max_new_tokens == 512 else CHECK_ID)
    baseline_budget = 512 if concise_prompt else 128
    changed_field = "messages[0].content" if concise_prompt else ("options.num_predict" if baseline_dir else None)
    root, collection_dir = Path(root), Path(collection_dir).resolve()
    output_name = "larger_primary_concise_v1" if concise_prompt else ("larger_primary_512_v1" if max_new_tokens == 512 else "larger_primary_v1")
    directory = root / "outputs" / output_name
    if directory.exists():
        raise ValueError(f"An attempt already exists: {directory}; review it before any repeat")
    before = tree_hashes(collection_dir)
    folder, payload = prepare_case(collection_dir)
    baseline_run, baseline_hashes = None, None
    if baseline_dir is not None:
        baseline_dir = Path(baseline_dir).resolve()
        expected_baseline = deepcopy(payload)
        expected_baseline["options"]["num_predict"] = baseline_budget
        baseline_run, baseline_hashes = validate_baseline(baseline_dir, expected_baseline, before, token_budget=baseline_budget)
        payload = deepcopy(payload)
        payload["options"]["num_predict"] = 512
        if concise_prompt:
            question = read_json(folder / "initial_request.json")["question"]
            payload["messages"][0]["content"] = question + "\n" + CONCISE_INSTRUCTION
    if available_ram is None:
        import psutil
        available_ram = lambda: psutil.virtual_memory().available / 1024**3
    ram_before = available_ram()
    if ram_before < 8:
        raise RuntimeError("This test requires at least 8 GiB available RAM as a conservative preflight guard")
    version = transport("/api/version")
    installed = transport("/api/tags")["models"]
    tag = next((m for m in installed if m.get("name") == MODEL or m.get("model") == MODEL), None)
    if tag is None:
        raise ValueError("The previously downloaded qwen2.5vl:7b is not installed in this Ollama service")
    if baseline_run and (tag.get("digest") != baseline_run["model_digest"]
                         or version.get("version") != baseline_run["ollama_version"]):
        raise ValueError(f"The installed model digest or Ollama version changed since the {baseline_budget}-token baseline")
    details = transport("/api/show", {"model": MODEL})
    if "vision" not in details.get("capabilities", []):
        raise ValueError("The installed model does not advertise image support")
    directory.mkdir(parents=True)
    if baseline_dir is not None:
        copied_baseline = directory / f"baseline_{baseline_budget}"
        copied_baseline.mkdir()
        for name in BASELINE_FILES:
            shutil.copy2(baseline_dir / name, copied_baseline / name)
            if file_sha256(copied_baseline / name) != baseline_hashes[name]:
                raise ValueError("The source baseline changed while copying evidence")
    write_json(directory / "model_details.json", details)
    write_json(directory / "request.json", payload)
    write_json(directory / "run.json", {"check_id": check_id, "selected_id": SELECTED_ID,
        "role": "train", "started_utc": datetime.now(timezone.utc).isoformat(), "model": MODEL,
        "model_digest": tag.get("digest"), "ollama_version": version.get("version"),
        "source_collection_sha256": before, "script_sha256": file_sha256(__file__),
        "max_new_tokens": max_new_tokens, "baseline_directory": str(baseline_dir) if baseline_dir else None,
        "baseline_sha256": baseline_hashes,
        "changed_request_field": changed_field,
        "prompt_variant": "concise-three-step-v1" if concise_prompt else "original",
        "purpose": "one-image larger primary model runtime/output check; not benchmark evaluation",
        "limitations": ["Ollama backend, quantization, templates and image processing differ from the original primary",
                        "No same-checkpoint continuation, training feasibility or model improvement is established"]})
    result = {"status": "incomplete", "check_id": check_id, "model": MODEL, "selected_id": SELECTED_ID,
        "max_new_tokens": max_new_tokens,
        "generation_requests_started": 0, "raw_response": None, "cpu_requested": True,
        "model_digest": tag.get("digest"), "ollama_version": version.get("version"),
        "available_ram_before_gib": round(ram_before, 3), "peak_ollama_ram_gib": None,
        "training_examples_admitted": 0, "optimizer_steps": 0, "benchmark_accuracy": None,
        "output_folder": str(directory)}
    started = time.perf_counter()
    try:
        if tree_hashes(collection_dir) != before:
            raise ValueError("Original collection changed before generation")
        if baseline_dir is not None and tree_hashes(baseline_dir) != baseline_hashes:
            raise ValueError("The source baseline changed before generation")
        result["generation_requests_started"] = 1
        write_json(directory / "summary.json", result)
        print(f"Larger-model output folder: {directory}", flush=True)
        print(f"Requesting one CPU answer from qwen2.5vl:7b ({max_new_tokens} tokens maximum)...", flush=True)
        reply = transport("/api/chat", payload, timeout=600)
        write_json(directory / "response.json", reply)
        text = reply.get("message", {}).get("content", "")
        if not isinstance(text, str) or reply.get("model") != MODEL or reply.get("done") is not True:
            raise ValueError("Ollama did not return a completed reply from the selected model")
        reason = reply.get("done_reason")
        truncated = reason == "length" or (reason not in ("stop", "length") and reply.get("eval_count", 0) >= max_new_tokens)
        result.update(status="complete", raw_response=text, generated_tokens=reply.get("eval_count"),
                      done_reason=reason, hit_token_limit=truncated,
                      outcome="empty_output" if not text.strip() else ("truncated_output" if truncated else "answer_generated_requires_review"),
                      model_inference_completed=True, finished_answer_available=bool(text.strip()) and not truncated,
                      provider_total_seconds=reply.get("total_duration", 0) / 1e9,
                      provider_load_seconds=reply.get("load_duration", 0) / 1e9,
                      provider_generation_seconds=reply.get("eval_duration", 0) / 1e9)
        # The reference is read only after generation and never enters the request.
        reference = read_json(folder / "verification.json")["reference"]
        write_json(directory / "single_case_verification.json", {"reference": reference,
            **score_answer(text, reference), "scope": "single training case; not benchmark accuracy"})
        try:
            models = transport("/api/ps").get("models", [])
            write_json(directory / "loaded_models.json", {"models": models})
            loaded = next((m for m in models if m.get("name") == MODEL or m.get("model") == MODEL), None)
            if loaded:
                result.update(server_model_allocation_gib=round(loaded.get("size", 0) / 1024**3, 3),
                              server_vram_bytes=loaded.get("size_vram"), cpu_only_reported=loaded.get("size_vram") == 0)
                if loaded.get("digest") and tag.get("digest") and loaded["digest"] != tag["digest"]:
                    result.update(status="incomplete", identity_error="Installed and loaded model digests differ")
        except Exception as error:
            result["allocation_check_error"] = safe_error(error)
    except (Exception, KeyboardInterrupt) as error:
        result.update(status="incomplete", error=safe_error(error),
                      next_action="Review the saved failure before retrying; a timed-out request may still run in Ollama")
    finally:
        result.update(elapsed_seconds=round(time.perf_counter() - started, 3),
                      available_ram_after_gib=round(available_ram(), 3),
                      memory_note="Server allocation and system available RAM are not a measured peak model RSS")
        if tree_hashes(collection_dir) != before:
            result.update(status="incomplete", integrity_error="Original collection changed during the check")
        if baseline_dir is not None:
            result["baseline_preserved"] = tree_hashes(baseline_dir) == baseline_hashes
            if not result["baseline_preserved"]:
                result.update(status="incomplete", baseline_error="The source baseline changed")
            result["changed_request_field"] = changed_field
        write_json(directory / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Check the installed 7B vision model on one saved training image")
    parser.add_argument("--collection-dir", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, choices=(128, 512), default=128)
    parser.add_argument("--baseline-dir", type=Path, help="128-token baseline for the budget extension, or original-prompt 512-token baseline with --concise-prompt")
    parser.add_argument("--concise-prompt", action="store_true", help="Approved three-step prompt test against the original-prompt 512-token run")
    args = parser.parse_args()
    try:
        result = run_check(ROOT, args.collection_dir, max_new_tokens=args.max_new_tokens,
                           baseline_dir=args.baseline_dir, concise_prompt=args.concise_prompt)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "complete" else 2
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
