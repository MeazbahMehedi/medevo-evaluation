import gc
import hashlib
from pathlib import Path
import time


QUESTION = "Name the color and shape in this image. Answer in a few words."
TARGET = "red circle"


def practice_image():
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (128, 128), "white")
    ImageDraw.Draw(image).ellipse((24, 24, 104, 104), fill="red")
    return image


def load_primary(settings):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, set_seed

    torch.set_num_threads(settings["cpu_threads"])
    set_seed(settings["seed"])
    source = settings["model_id"]
    revision = settings["revision"]
    processor = AutoProcessor.from_pretrained(source, revision=revision,
                                              trust_remote_code=False)
    # A single 512-pixel view keeps CPU memory bounded; this is a study limitation.
    processor.image_processor.size = {"longest_edge": settings["image_edge"]}
    processor.image_processor.do_image_splitting = False
    model = AutoModelForImageTextToText.from_pretrained(
        source, revision=revision, dtype=torch.float32,
        attn_implementation="eager", trust_remote_code=False
    )
    return model.cpu().eval(), processor


def user_messages(image, question=QUESTION):
    return [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question}
    ]}]


def generate_response(model, processor, image, question, max_new_tokens):
    import torch

    inputs = processor.apply_chat_template(
        user_messages(image, question), tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                do_sample=False, use_cache=True)
    generated = output[0, inputs["input_ids"].shape[1]:]
    text = processor.decode(generated, skip_special_tokens=True).strip()
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    ended = bool(generated.numel()) and int(generated[-1]) in eos_ids
    return {"question": question, "raw_response": text,
            "model_weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters()),
            "generated_tokens": generated.numel(),
            "hit_token_limit": generated.numel() >= max_new_tokens and not ended,
            "generation_seconds": round(time.perf_counter() - started, 3),
            "input_shapes": {k: list(v.shape) for k, v in inputs.items()
                             if hasattr(v, "shape")},
            "answer_generated": bool(text),
            "benchmark_accuracy": None}


def inference(model, processor, image, max_new_tokens):
    return generate_response(model, processor, image, QUESTION, max_new_tokens)


def supervised_batch(processor, image, target=TARGET):
    import torch

    messages = user_messages(image)
    prompt = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt"
    )
    complete = messages + [{"role": "assistant", "content": [
        {"type": "text", "text": target}
    ]}]
    batch = processor.apply_chat_template(
        complete, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt"
    )
    length = prompt["input_ids"].shape[1]
    if not torch.equal(batch["input_ids"][:, :length], prompt["input_ids"]):
        raise ValueError("Chat-template boundary changed; cannot safely mask the prompt")
    if batch["input_ids"].shape[1] <= length:
        raise ValueError("The assistant target has no tokens")
    labels = batch["input_ids"].clone()
    labels[:, :length] = -100
    labels[batch["attention_mask"] == 0] = -100
    batch["labels"] = labels
    return batch


def attach_adapter(model):
    from peft import LoraConfig, TaskType, get_peft_model

    names = [name for name, _ in model.named_modules()
             if ".text_model." in name and name.endswith((".q_proj", ".v_proj"))]
    if not names:
        raise ValueError("No text attention projections found; check model compatibility")
    config = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0,
                        target_modules=names, bias="none",
                        task_type=TaskType.CAUSAL_LM)
    return get_peft_model(model, config)


def training_step(model, batch, learning_rate):
    import torch

    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not trainable or any("lora_" not in name for name in trainable):
        raise ValueError("Expected only LoRA parameters to be trainable")
    before = {name: p.detach().clone() for name, p in trainable.items()}
    frozen_name, frozen = next((n, p) for n, p in model.named_parameters()
                               if not p.requires_grad and p.numel() > 0)
    frozen_before = frozen.detach().clone()
    optimizer = torch.optim.AdamW(trainable.values(), lr=learning_rate)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    loss = model(**batch, use_cache=False).loss
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite training loss")
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0)
    if not torch.isfinite(norm):
        raise ValueError("Nonfinite gradient norm")
    optimizer.step()
    changed = [name for name, p in trainable.items()
               if not torch.equal(before[name], p.detach())]
    frozen_unchanged = torch.equal(frozen_before, frozen.detach())
    if not changed or not frozen_unchanged:
        raise ValueError("Parameter update verification failed")
    return {"loss_before_update": float(loss.detach()),
            "update_seconds": round(time.perf_counter() - started, 3),
            "trainable_parameters": sum(p.numel() for p in trainable.values()),
            "changed_parameter_tensors": len(changed),
            "frozen_parameter_sample": frozen_name,
            "frozen_parameter_sample_unchanged": frozen_unchanged}


def training_probe(settings, directory):
    import torch
    from peft import PeftModel
    from .runtime import write_json

    image = practice_image()
    image.save(Path(directory) / "practice_image.png")
    model, processor = load_primary(settings)
    batch = supervised_batch(processor, image)
    model = attach_adapter(model)
    result = training_step(model, batch, settings["learning_rate"])
    result["model_weight_bytes"] = sum(p.numel() * p.element_size() for p in model.parameters())
    model.eval()
    with torch.inference_mode():
        expected = model(**batch, use_cache=False).logits[:, -1, :].float().clone()
    adapter_path = Path(directory) / "adapter"
    model.save_pretrained(adapter_path)
    processor.save_pretrained(Path(directory) / "processor")
    write_json(Path(directory) / "base_model.json", {
        "model_id": settings["model_id"], "revision": settings["revision"]
    })
    del model
    gc.collect()
    base, _ = load_primary(settings)
    restored = PeftModel.from_pretrained(base, adapter_path).eval()
    with torch.inference_mode():
        actual = restored(**batch, use_cache=False).logits[:, -1, :].float()
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    digest = hashlib.sha256((adapter_path / "adapter_model.safetensors").read_bytes()).hexdigest()
    result.update(adapter_reload_passed=True, adapter_sha256=digest,
                  supervised_tokens=int((batch["labels"] != -100).sum()),
                  training_example="synthetic red-circle fixture; not medical data")
    return result
