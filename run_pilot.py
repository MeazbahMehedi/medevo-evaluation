import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import uuid

from src.runtime import environment_snapshot, read_settings, safe_error, supervise, write_json


ROOT = Path(__file__).resolve().parent


def worker(stage, settings, directory):
    try:
        if stage == "auditor":
            from src.auditor import auditor_probe
            result = auditor_probe(directory)
        else:
            from src.model import inference, load_primary, practice_image, training_probe
            if stage == "train":
                result = training_probe(settings, directory)
            else:
                model, processor = load_primary(settings)
                image = practice_image()
                image.save(directory / "practice_image.png")
                result = inference(model, processor, image, settings["max_new_tokens"])
                if not result["answer_generated"]:
                    raise ValueError("The model returned an empty decoded response")
            result["status"] = "passed"
    except (Exception, KeyboardInterrupt) as error:
        result = {"status": "failed", "error": safe_error(error)}
    write_json(directory / "result.json", result)
    print(f"{stage}: {result['status']}", flush=True)
    return 0 if result["status"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser(description="Phase 1 CPU feasibility checks; default is environment only")
    parser.add_argument("stage", nargs="?", default="environment",
                        choices=["environment", "inference", "train", "auditor"])
    parser.add_argument("--settings", type=Path, default=ROOT / "settings.json")
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        settings = read_settings(args.settings)
        if args.worker_dir:
            return worker(args.stage, settings, args.worker_dir)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        directory = ROOT / "outputs" / f"{stamp}_{args.stage}_{uuid.uuid4().hex[:6]}"
        directory.mkdir(parents=True)
        environment = environment_snapshot(ROOT)
        write_json(directory / "environment.json", environment)
        write_json(directory / "settings.json", settings)
        write_json(directory / "run.json", {"stage": args.stage, "started_utc": stamp,
                   "purpose": "Phase 1 feasibility; not a medical benchmark result"})
        print(f"Output folder: {directory}", flush=True)
        if args.stage == "environment":
            missing = [name for name, version in environment["packages"].items() if version is None]
            result = {"status": "blocked" if missing else "passed",
                      "missing_packages": missing, "environment": environment,
                      "scope": "Package presence and hardware inventory only; model operations not yet checked"}
        else:
            if environment["packages"]["psutil"] is None:
                raise RuntimeError("Install requirements.txt before running a model or auditor probe")
            if args.stage in ("inference", "train"):
                import psutil
                if psutil.virtual_memory().available < settings["minimum_available_gib"] * 1024**3:
                    raise RuntimeError("Available RAM is below the configured pilot minimum")
            environment_vars = {"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_XET": "1"}
            os.environ.update(environment_vars)
            command = [sys.executable, str(Path(__file__).resolve()), args.stage,
                       "--settings", str(args.settings.resolve()), "--worker-dir", str(directory)]
            resource = supervise(command, settings, directory / "console.log")
            result_file = directory / "result.json"
            result = json.loads(result_file.read_text()) if result_file.exists() else {"status": "failed"}
            weight_bytes = result.get("model_weight_bytes", 0)
            sampled = resource["peak_process_tree_rss_bytes"] or 0
            valid_memory = bool(sampled) and sampled >= 0.95 * weight_bytes
            resource["memory_measurement_plausible"] = valid_memory
            if not valid_memory and args.stage in ("inference", "train"):
                resource["warning"] = "Memory samples are unavailable or below the loaded model's weight size; verify process memory on the target PC"
                if result.get("status") == "passed":
                    result["operation_checks"] = "passed"
                    result["status"] = "blocked"
                    result["reason"] = "resource_measurement_unverified"
            write_json(directory / "resources.json", resource)
            if resource["limit_reached"] or resource["worker_exit_code"] != 0:
                result["status"] = "blocked" if result.get("status") == "blocked" else "failed"
                result["worker"] = resource
        write_json(directory / "result.json", result)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "passed" else 2
    except (Exception, KeyboardInterrupt) as error:
        result = {"status": "failed", "error": safe_error(error)}
        if "directory" in locals():
            write_json(directory / "result.json", result)
        print(json.dumps(result, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
