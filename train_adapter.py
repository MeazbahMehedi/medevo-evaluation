"""Phase 4: train a LoRA adapter on the admitted training bank (fixed schedule, declared in advance).

Only the language-model Linear layers are adapted; the vision encoder and base weights stay frozen.
Prompt tokens are masked so only the assistant answer is learned.

Example:  python train_adapter.py --epochs 2
          python train_adapter.py --device cuda      (on a machine with a CUDA GPU)
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time

from medevo_study.core import (DATA_DIR, OUTPUTS, PROTOCOL_DIR, append_jsonl, code_hashes, file_sha256, load_image,
                               load_records, read_json, read_jsonl, safe_error, utc_now, write_json)

BANKS = [OUTPUTS / "training_bank_v2" / "training_bank.jsonl", OUTPUTS / "training_bank_v1" / "training_bank.jsonl"]
FOLDER = OUTPUTS / "adapter_v1"


def default_bank():
    return next((path for path in BANKS if path.exists()), BANKS[-1])


def train(epochs, learning_rate, rank, alpha, accumulation, device, dtype, max_hours, seed, bank_path=None):
    import torch
    from peft import LoraConfig, get_peft_model
    from medevo_study import transformers_backend as backend

    if FOLDER.exists():
        raise ValueError(f"{FOLDER} exists; keep it as evidence and choose a new folder name for another run")
    bank_path = bank_path or default_bank()
    bank = read_jsonl(bank_path)
    if not bank:
        raise ValueError("The training bank is empty; complete Phase 3 first")
    _, records = load_records(DATA_DIR)
    roles = {r["id"]: r["role"] for r in records}
    if any(roles.get(item["id"]) != "train" for item in bank):
        raise ValueError("Training bank contains non-training questions")
    protocol = read_json(PROTOCOL_DIR / "protocol.json")
    settings = read_json(PROTOCOL_DIR / "primary_model.json")
    torch.manual_seed(seed)
    random.seed(seed)
    model, processor, info = backend.load(settings["model_id"], settings["revision"], dtype, settings["threads"],
                                          device=device)
    pattern, leaves = backend.language_lora_pattern(model)
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=pattern, bias="none")
    model = get_peft_model(model, config)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate, weight_decay=0.0)
    steps_total = max(1, (len(bank) * epochs) // accumulation)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / max(1, steps_total // 10)) * max(0.0, 1 - step / steps_total))
    FOLDER.mkdir(parents=True)
    run = {"started_utc": utc_now(), "bank_file": str(bank_path), "bank_sha256": file_sha256(bank_path), "bank_size": len(bank),
           "bank_kinds": {k: sum(i["kind"] == k for i in bank) for k in ("correct_rollout", "repair")},
           "base_model": settings, "device": device, "train_dtype": dtype, "seed": seed,
           "lora": {"rank": rank, "alpha": alpha, "dropout": 0.05, "target_pattern": pattern, "target_leaves": leaves},
           "optimizer": {"name": "AdamW", "learning_rate": learning_rate, "weight_decay": 0.0,
                         "schedule": "linear warmup 10% then linear decay"},
           "epochs": epochs, "gradient_accumulation": accumulation, "batch_size": 1, "planned_optimizer_steps": steps_total,
           "trainable_parameters": trainable, "total_parameters": total, "code_sha256": code_hashes(),
           "checkpoint_policy": "final adapter after the fixed schedule; no test-based or feedback-based selection"}
    write_json(FOLDER / "run.json", run)
    print(f"Training {trainable:,} of {total:,} parameters on {len(bank)} examples, {epochs} epochs", flush=True)
    model.train()
    started, step, micro = time.perf_counter(), 0, 0
    suffix = "\n" + protocol["prompt"]
    stopped_early = False
    for epoch in range(epochs):
        order = list(range(len(bank)))
        random.Random(seed + epoch).shuffle(order)
        for position, index in enumerate(order):
            if max_hours and time.perf_counter() - started > max_hours * 3600:
                stopped_early = True
                break
            item = bank[index]
            image = load_image(DATA_DIR, item)
            inputs, counts = backend.training_example(processor, model, image, item["question"] + suffix, item["target_text"])
            inputs = {k: v.to(device) for k, v in inputs.items()}
            loss = model(**inputs).loss
            (loss / accumulation).backward()
            micro += 1
            append_jsonl(FOLDER / "train_log.jsonl", {"epoch": epoch, "position": position, "id": item["id"],
                                                     "kind": item["kind"], "loss": float(loss), **counts,
                                                     "elapsed_seconds": round(time.perf_counter() - started, 1)})
            if micro % accumulation == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                step += 1
                print(f"epoch {epoch + 1} step {step}/{steps_total} loss {float(loss):.4f} "
                      f"({time.perf_counter() - started:.0f}s)", flush=True)
        if stopped_early:
            break
    if micro % accumulation:
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        step += 1
    model.save_pretrained(str(FOLDER / "adapter"))
    adapter_hash = file_sha256(FOLDER / "adapter" / "adapter_model.safetensors")
    run.update(ended_utc=utc_now(), optimizer_steps=step, examples_seen=micro, stopped_early=stopped_early,
               elapsed_seconds=round(time.perf_counter() - started, 1), adapter_sha256=adapter_hash)
    write_json(FOLDER / "run.json", run)
    return {"status": "trained" if not stopped_early else "stopped_at_time_limit", "adapter": str(FOLDER / "adapter"),
            "optimizer_steps": step, "examples_seen": micro, "elapsed_seconds": run["elapsed_seconds"],
            "adapter_sha256": adapter_hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-hours", type=float, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bank", type=Path, help="Training bank (default: v2 if present, else v1)")
    args = parser.parse_args()
    try:
        result = train(args.epochs, args.learning_rate, args.rank, args.alpha, args.gradient_accumulation,
                       args.device, args.dtype, args.max_hours, args.seed, args.bank)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "blocked", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
