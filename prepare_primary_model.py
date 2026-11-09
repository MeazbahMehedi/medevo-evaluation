"""Download, pin and check the primary model (Qwen3.5-2B) before it is evaluated or trained.

Writes outputs/frozen_protocol_v1/primary_model.json with the exact revision and settings.
Example:  python prepare_primary_model.py --dtype bfloat16 --threads 8
"""

import argparse
import json
import sys
import time
import traceback

from medevo_study.core import (DATA_DIR, OUTPUTS, PROTOCOL_DIR, load_image, load_records, primary_correct, read_json,
                               safe_error, stratified_sample, utc_now, write_json)
from medevo_study import transformers_backend as backend


def prepare(revision, dtype, threads, replace):
    target = PROTOCOL_DIR / "primary_model.json"
    protocol = read_json(PROTOCOL_DIR / "protocol.json")
    started_runs = [p for p in OUTPUTS.glob("*qwen35_2b*") if p.is_dir()] + [p for p in OUTPUTS.glob("training_bank*")]
    if target.exists() and not replace:
        raise ValueError(f"{target} exists. Use --replace only if no 2B evaluation or collection has started.")
    if target.exists() and started_runs:
        raise ValueError("Primary-model runs already exist; the primary model settings are frozen.")
    import psutil
    process = psutil.Process()
    sha = backend.resolve_revision(backend.MODEL_ID, revision)
    load_started = time.perf_counter()
    model, processor, info = backend.load(backend.MODEL_ID, sha, dtype, threads)
    load_seconds = time.perf_counter() - load_started
    _, records = load_records(DATA_DIR)
    record = stratified_sample(records, "feedback", "slake", 1, 7, "primary-readiness-v1")[0]
    image = load_image(DATA_DIR, record)
    result = backend.generate(model, processor, image, record["question"] + "\n" + protocol["prompt"],
                              protocol["output_token_budget"])
    nll = backend.continuation_nll(model, processor, image, record["question"] + "\n" + protocol["prompt"],
                                   "", result["generated_token_ids"][:64])
    settings = {"model_id": backend.MODEL_ID, "revision": sha, "requested_revision": revision, "dtype": dtype,
                "threads": threads, "max_new_tokens": protocol["output_token_budget"],
                "sampling_for_collection": backend.SAMPLING, "frozen_utc": utc_now(),
                "readiness_check": {"feedback_id": record["id"], "load_seconds": round(load_seconds, 1),
                                    "generation_seconds": result["generation_seconds"],
                                    "generated_tokens": result["generated_tokens"],
                                    "tokens_per_second": round(result["generated_tokens"] / max(result["generation_seconds"], 1e-6), 2),
                                    "hit_token_limit": result["hit_token_limit"],
                                    "boxed_answer_present": "\\boxed{" in result["raw_response"],
                                    "correct": primary_correct(result["raw_response"], record["answer"]),
                                    "log_likelihood_check_nats": nll,
                                    "process_rss_gib": round(process.memory_info().rss / 1024 ** 3, 2),
                                    "raw_response": result["raw_response"]},
                "loader": info}
    write_json(target, settings)
    return {"status": "ready", "file": str(target), "revision": sha,
            **{k: settings["readiness_check"][k] for k in ("load_seconds", "generation_seconds", "generated_tokens",
                                                           "tokens_per_second", "boxed_answer_present",
                                                           "process_rss_gib")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--revision", default="main", help="Hugging Face revision; resolved to a commit hash")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(prepare(args.revision, args.dtype, args.threads, args.replace), indent=2))
        return 0
    except Exception as error:
        traceback.print_exc()
        print(json.dumps({"status": "blocked", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
