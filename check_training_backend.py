"""Check CPU NF4/PEFT operations on a tiny synthetic projection."""

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from src.benchmarks import file_sha256
from src.runtime import environment_snapshot, safe_error, supervise, write_json
from src.training_collection import read_json


ROOT = Path(__file__).resolve().parent
PROBE_ID = "cpu-nf4-lora-backend-v1"
BNB_VERSION = "0.50.2"
LIMITS = {"timeout_seconds": 180, "memory_limit_gib": 4}
PACKAGES = ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "safetensors", "psutil")


def package_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def run_operations(directory):
    # Set these in the fresh worker before importing Hub-dependent packages.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    import bitsandbytes as bnb
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file, save_file

    directory = Path(directory)
    result = {"status": "failed", "probe_id": PROBE_ID, "scope": "synthetic 64x64 CPU operator check",
              "synthetic_optimizer_steps": 0, "medical_optimizer_steps": 0, "training_examples_admitted": 0,
              "full_model_checked": False, "full_model_training_feasible": None,
              "weights_downloaded": False, "quantization": "NF4", "compute_dtype": "float32",
              "kernel_package_installed": importlib.util.find_spec("kernels") is not None,
              "limitations": ["Does not load Qwen or test its multimodal architecture",
                              "Does not estimate 7B training RAM, runtime or medical quality",
                              "Optional optimized kernel availability/performance is not established"]}
    started = time.perf_counter()
    try:
        if bnb.__version__ != BNB_VERSION:
            raise ValueError("Use the pinned bitsandbytes version 0.50.2 for this check")
        torch.set_num_threads(4)
        torch.manual_seed(7)
        torch.use_deterministic_algorithms(True)
        result["cpu_capability"] = torch.backends.cpu.get_cpu_capability()

        class ProjectionProbe(torch.nn.Module):
            def __init__(self, weight=None, packed_state=None):
                super().__init__()
                self.is_loaded_in_4bit = True
                self.proj = bnb.nn.Linear4bit(64, 64, bias=False, compute_dtype=torch.float32,
                                             quant_type="nf4", compress_statistics=True)
                if packed_state is None:
                    self.proj.load_state_dict({"weight": weight})
                    self.to("cpu")
                else:
                    self.proj.weight = bnb.nn.Params4bit.from_prequantized(
                        packed_state["weight"],
                        {key: value for key, value in packed_state.items() if key != "weight"},
                        requires_grad=False, device="cpu", module=self.proj)

            def forward(self, value):
                return self.proj(value)

        weight = torch.randn(64, 64, dtype=torch.float32) / 8
        base = ProjectionProbe(weight)
        if base.proj.weight.dtype != torch.uint8 or base.proj.weight.quant_state is None:
            raise ValueError("The CPU base was not actually quantized; a float fallback is not a pass")
        if base.proj.weight.device.type != "cpu" or base.proj.weight.quant_state.quant_type != "nf4":
            raise ValueError("Expected an NF4 base on CPU")
        frozen_before = {name: value.detach().clone() for name, value in base.proj.state_dict().items()}
        result.update(quantized_cpu_base_verified=True, original_weight_parameters=4096,
                      packed_weight_bytes=base.proj.weight.numel() * base.proj.weight.element_size())
        model = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["proj"],
                                              lora_dropout=0.0, bias="none"))
        layer = model.base_model.model.proj
        if not type(layer).__module__.startswith("peft.tuners.lora.bnb"):
            raise ValueError("PEFT did not attach its bitsandbytes LoRA layer")
        trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
        if not trainable or any("lora_" not in name for name in trainable):
            raise ValueError("Only the intended LoRA parameters may be trainable")
        adapter_before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
        inputs = torch.randn(4, 64, dtype=torch.float32, requires_grad=True)
        targets = torch.randn(4, 64, dtype=torch.float32)
        optimizer = torch.optim.AdamW(trainable.values(), lr=0.001)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output_before = model(inputs)
        loss = torch.nn.functional.mse_loss(output_before, targets)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Nonfinite synthetic loss")
        loss.backward()
        if inputs.grad is None or not bool(torch.isfinite(inputs.grad).all()) or not bool(inputs.grad.abs().sum() > 0):
            raise ValueError("No finite nonzero input gradient through the quantized CPU layer")
        if any(parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()) for parameter in trainable.values()):
            raise ValueError("Missing or nonfinite LoRA gradient")
        optimizer.step()
        result["synthetic_optimizer_steps"] = 1
        changed = [name for name, parameter in trainable.items() if not torch.equal(parameter.detach(), adapter_before[name])]
        frozen_after = layer.get_base_layer().state_dict()
        if not changed or set(frozen_after) != set(frozen_before) or any(not torch.equal(frozen_after[name], value) for name, value in frozen_before.items()):
            raise ValueError("The adapter did not update or the frozen quantized base changed")
        # Keep the training path (dropout is zero). AVX512 inference can repack
        # scales into another format; that separate optimization is out of scope.
        model.train()
        with torch.inference_mode():
            expected = model(inputs.detach()).float().clone()
        if not bool(torch.isfinite(expected).all()) or torch.equal(expected, output_before.detach()):
            raise ValueError("The finite adapter update did not affect the output")
        # Reload the packed base itself, rather than quantizing floating weights again.
        save_file(frozen_before, str(directory / "synthetic_base.safetensors"))
        model.save_pretrained(directory / "adapter", safe_serialization=True)
        saved = load_file(str(directory / "synthetic_base.safetensors"), device="cpu")
        restored = PeftModel.from_pretrained(ProjectionProbe(packed_state=saved), directory / "adapter", is_trainable=False).train()
        with torch.inference_mode():
            actual = restored(inputs.detach()).float()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
        restored_base = restored.base_model.model.proj.get_base_layer().state_dict()
        if set(restored_base) != set(frozen_before):
            raise ValueError("The restored synthetic base differs from the original NF4 base")
        for name, value in frozen_before.items():
            if "quant_state.bitsandbytes__" in name:
                # Metadata is JSON stored as uint8; object key order has no meaning.
                same = json.loads(bytes(restored_base[name].tolist())) == json.loads(bytes(value.tolist()))
            else:
                same = torch.equal(restored_base[name], value)
            if not same:
                raise ValueError(f"The restored synthetic base changed: {name}")
        result.update(status="passed", finite_input_gradient=True, frozen_base_unchanged=True,
                      adapter_output_changed=True, adapter_reload_passed=True, packed_base_reload_passed=True,
                      output_comparison_mode="training mode with zero dropout; inference repacking not tested",
                      changed_adapter_tensors=len(changed), trainable_parameters=sum(p.numel() for p in trainable.values()),
                      loss_before_update=float(loss.detach()), reload_max_abs_error=float((actual - expected).abs().max()),
                      adapter_sha256=file_sha256(directory / "adapter/adapter_model.safetensors"))
    except (Exception, KeyboardInterrupt) as error:
        result["error"] = safe_error(error)
    finally:
        result["operation_elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json(directory / "probe.json", result)
    return result


def worker(directory):
    directory = Path(directory)
    run = read_json(directory / "run.json")
    if run.get("probe_id") != PROBE_ID or run.get("script_sha256") != file_sha256(__file__) or run.get("packages") != package_versions():
        raise ValueError("The backend probe implementation or dependencies changed")
    with (directory / "attempt.json").open("x", encoding="utf-8") as stream:
        json.dump({"probe_id": PROBE_ID, "attempts": 1}, stream)
    result = run_operations(directory)
    return 0 if result["status"] == "passed" else 2


def start_probe(root):
    root = Path(root)
    directory = root / "outputs/cpu_4bit_backend_v1"
    if directory.exists():
        raise ValueError(f"An attempt already exists: {directory}; inspect it before any repeat")
    packages = package_versions()
    missing = [name for name, version in packages.items() if version is None]
    if missing or packages["bitsandbytes"] != BNB_VERSION:
        return {"status": "blocked", "probe_id": PROBE_ID, "missing_packages": missing,
                "required_bitsandbytes": BNB_VERSION, "packages": packages,
                "next_action": "Install the pinned bitsandbytes wheel without changing existing dependencies"}
    directory.mkdir(parents=True)
    write_json(directory / "run.json", {"probe_id": PROBE_ID, "packages": packages, "script_sha256": file_sha256(__file__),
               "limits": LIMITS, "seed": 7, "cpu_threads": 4, "data": "synthetic random tensors only"})
    write_json(directory / "environment.json", environment_snapshot(root))
    print(f"CPU backend output folder: {directory}", flush=True)
    command = [sys.executable, str(Path(__file__).resolve()), "--worker-dir", str(directory.resolve())]
    resources = supervise(command, LIMITS, directory / "console.log")
    resources["measurement_scope"] = "Sampled Python worker process-tree RSS; synthetic operator test, not 7B training memory"
    resources["memory_measurement_plausible"] = bool(resources.get("memory_measurement_available") and
                                                    resources.get("peak_process_tree_rss_bytes", 0) >= 64 * 1024**2)
    if not resources["memory_measurement_plausible"]:
        resources["memory_warning"] = "RSS sampling was unavailable or implausibly low; memory use is not validated"
    write_json(directory / "resources.json", resources)
    result = read_json(directory / "probe.json") if (directory / "probe.json").exists() else {
        "status": "failed", "probe_id": PROBE_ID, "error": "Worker stopped before producing a probe result",
        "full_model_checked": False, "full_model_training_feasible": None}
    if resources["worker_exit_code"] != 0 or resources["limit_reached"]:
        result["status"] = "failed"
    result.update(worker=resources, packages=packages, output_folder=str(directory))
    write_json(directory / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description="Check one CPU NF4/PEFT adapter update using tiny synthetic tensors")
    parser.add_argument("--worker-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args.worker_dir:
            return worker(args.worker_dir)
        result = start_probe(ROOT)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "passed" else 2
    except (Exception, KeyboardInterrupt) as error:
        print(json.dumps({"status": "failed", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
