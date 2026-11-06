import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time


PACKAGES = ["torch", "torchvision", "transformers", "peft", "accelerate",
            "psutil", "Pillow", "huggingface-hub", "num2words", "socksio", "pyarrow"]


def read_settings(path):
    settings = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict):
        raise ValueError("Settings must be a JSON object")
    if settings.get("model_id") != "HuggingFaceTB/SmolVLM2-500M-Video-Instruct":
        raise ValueError("This pilot is configured for the proposed SmolVLM2 model")
    if not re.fullmatch(r"[a-f0-9]{40}", settings.get("revision", "")):
        raise ValueError("Use an immutable 40-character model commit")
    bounds = {"seed": (0, 2**32 - 1), "cpu_threads": (1, 64),
              "image_edge": (512, 512), "max_new_tokens": (1, 128),
              "timeout_seconds": (1, 7200), "memory_limit_gib": (1, 128),
              "minimum_available_gib": (1, 128)}
    for key, (low, high) in bounds.items():
        value = settings.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid {key}; expected an integer from {low} to {high}")
    rate = settings.get("learning_rate")
    if type(rate) not in (int, float) or not 0 < rate <= 0.01:
        raise ValueError("learning_rate must be greater than zero and at most 0.01")
    return settings


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def environment_snapshot(root):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.platform(), "cpu_count": os.cpu_count(),
              "free_disk_gib": round(shutil.disk_usage(root).free / 1024**3, 2),
              "packages": versions, "git_installed": bool(shutil.which("git")),
              "github_cli_installed": bool(shutil.which("gh"))}
    if versions["psutil"]:
        import psutil
        ram = psutil.virtual_memory()
        result.update(total_ram_gib=round(ram.total / 1024**3, 2),
                      available_ram_gib=round(ram.available / 1024**3, 2))
    return result


def safe_error(error):
    message = f"{type(error).__name__}: {error}"
    for name in ("AUDITOR_API_KEY", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[redacted]")
    return message[:1800]


def supervise(command, settings, log_path):
    import psutil

    started = time.monotonic()
    peak = 0
    reason = None
    with Path(log_path).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1)
        tracked = None

        def copy_output():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)

        reader = threading.Thread(target=copy_output, daemon=True)
        reader.start()
        try:
            while process.poll() is None:
                children = []
                try:
                    tracked = psutil.Process(process.pid)
                    children = tracked.children(recursive=True)
                    rss = sum(p.memory_info().rss for p in [tracked] + children)
                    peak = max(peak, rss)
                    if rss > settings["memory_limit_gib"] * 1024**3:
                        reason = "memory_limit"
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                if time.monotonic() - started > settings["timeout_seconds"]:
                    reason = "time_limit"
                if reason:
                    for child in reversed(children):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    process.kill()
                    break
                time.sleep(0.2)
            process.wait()
        finally:
            if process.poll() is None:
                try:
                    children = tracked.children(recursive=True) if tracked else []
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                process.kill()
                process.wait()
            reader.join(timeout=5)
            process.stdout.close()
    return {"worker_exit_code": process.returncode, "limit_reached": reason,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "peak_process_tree_rss_bytes": peak if peak else None,
            "peak_process_tree_rss_gib": round(peak / 1024**3, 6) if peak else None,
            "memory_measurement_available": peak > 0,
            "memory_sampling_seconds": 0.2}
