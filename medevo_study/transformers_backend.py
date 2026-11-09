"""Primary model (Qwen3.5-2B) in Transformers: generation, prefix continuation, log-likelihood, LoRA.

All token sequences are built as  <generation prompt> + <assistant text tokens>, so generation,
scoring and training use one consistent tokenization path.
"""

import hashlib
import re
import time

MODEL_ID = "Qwen/Qwen3.5-2B"
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}  # model card, non-thinking VL settings
EXCLUDED_MODULES = ("visual", "vision", "merger", "lm_head", "embed")


def resolve_revision(model_id=MODEL_ID, revision="main"):
    from huggingface_hub import HfApi
    return HfApi().model_info(model_id, revision=revision).sha


def load(model_id, revision, dtype="bfloat16", threads=8, adapter_dir=None, device="cpu"):
    import torch
    import transformers

    torch.set_num_threads(threads)
    torch_dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    processor = transformers.AutoProcessor.from_pretrained(model_id, revision=revision)
    loader = getattr(transformers, "AutoModelForMultimodalLM", None) or transformers.AutoModelForImageTextToText
    model = loader.from_pretrained(model_id, revision=revision, dtype=torch_dtype).to(device)
    if adapter_dir is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.eval()
    info = {"model_id": model_id, "revision": revision, "dtype": dtype, "threads": threads, "device": device,
            "loader": loader.__name__, "transformers": transformers.__version__, "torch": torch.__version__,
            "adapter_dir": None if adapter_dir is None else str(adapter_dir),
            "parameters": sum(p.numel() for p in model.parameters())}
    return model, processor, info


def _prompt_inputs(processor, image, user_text):
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": user_text}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                         return_tensors="pt", enable_thinking=False)


def _extend(inputs, extra_ids):
    """Append text tokens and extend every per-token tensor the processor returned
    (attention mask, token-type ids such as mm_token_type_ids) so all lengths stay equal."""
    import torch
    if not extra_ids:
        return dict(inputs)
    length = inputs["input_ids"].shape[1]
    extended = {}
    for key, value in inputs.items():
        if key == "input_ids":
            extended[key] = torch.cat([value, torch.tensor([extra_ids], dtype=value.dtype)], dim=1)
        elif torch.is_tensor(value) and value.dim() == 2 and value.shape == (1, length) and "pixel" not in key \
                and "grid" not in key:
            fill = 1 if key == "attention_mask" else 0  # text tokens are type 0
            extended[key] = torch.cat([value, torch.full((1, len(extra_ids)), fill, dtype=value.dtype)], dim=1)
        else:
            extended[key] = value
    return extended


def _text_ids(processor, text):
    return processor.tokenizer(text, add_special_tokens=False)["input_ids"] if text else []


def _end_ids(processor, model):
    tokenizer = processor.tokenizer
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eos = model.generation_config.eos_token_id
    ids = set(eos if isinstance(eos, list) else [eos])
    if isinstance(end, int) and end != tokenizer.unk_token_id:
        ids.add(end)
    return ids


def strip_thinking(text):
    return re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)


def generate(model, processor, image, user_text, max_new_tokens, prefill="", sample=False, seed=7):
    import torch

    inputs = _prompt_inputs(processor, image, user_text)
    prompt_length = inputs["input_ids"].shape[1]
    inputs = _extend(inputs, _text_ids(processor, prefill))
    length = inputs["input_ids"].shape[1]
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    options = {"max_new_tokens": max_new_tokens, "use_cache": True}
    if sample:
        torch.manual_seed(seed)
        options.update(do_sample=True, **SAMPLING)
    else:
        options.update(do_sample=False, temperature=None, top_p=None, top_k=None)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **options)
    tokens = output[0, length:].tolist()
    end_ids = _end_ids(processor, model)
    ended = bool(tokens) and tokens[-1] in end_ids
    text = processor.decode(tokens, skip_special_tokens=True)
    return {"raw_response": strip_thinking(text), "generated_token_ids": tokens,
            "generated_tokens": len(tokens), "hit_token_limit": len(tokens) >= max_new_tokens and not ended,
            "generation_seconds": round(time.perf_counter() - started, 3), "prompt_tokens": prompt_length,
            "prefill_tokens": length - prompt_length, "sampled": sample, "seed": seed if sample else None,
            "input_sha256": hashlib.sha256(str(output[0, :length].tolist()).encode()).hexdigest()}


def _empty_prompt_inputs(processor):
    """Chat prompt with an empty user turn: the context without the task input x (MedEvo Eq. 10)."""
    messages = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                                         return_tensors="pt", enable_thinking=False)


def text_ids(processor, text):
    return _text_ids(processor, text)


def continuation_nll(model, processor, image, user_text, prefill, continuation_ids, include_input=True):
    """Mean negative log-likelihood (nats/token) of fixed continuation tokens after the prefill.

    With include_input=False the image and question are omitted and only the prefill is kept.
    """
    import torch

    if not continuation_ids:
        return None
    base = _prompt_inputs(processor, image, user_text) if include_input else _empty_prompt_inputs(processor)
    inputs = _extend(base, _text_ids(processor, prefill) + list(continuation_ids))
    total = inputs["input_ids"].shape[1]
    start = total - len(continuation_ids)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[0, start - 1:total - 1].float()
    target = inputs["input_ids"][0, start:total]
    return float(torch.nn.functional.cross_entropy(logits, target, reduction="mean"))


def training_example(processor, model, image, user_text, target_text):
    """Inputs plus labels with the prompt masked; only the assistant answer is learned."""
    import torch

    inputs = _prompt_inputs(processor, image, user_text)
    prompt_length = inputs["input_ids"].shape[1]
    end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    tail = [end] if isinstance(end, int) and end != processor.tokenizer.unk_token_id else \
        [model.generation_config.eos_token_id if isinstance(model.generation_config.eos_token_id, int)
         else model.generation_config.eos_token_id[0]]
    inputs = _extend(inputs, _text_ids(processor, target_text) + tail)
    labels = inputs["input_ids"].clone()
    labels[:, :prompt_length] = -100
    inputs["labels"] = labels
    return inputs, {"prompt_tokens": prompt_length, "target_tokens": int(labels.shape[1] - prompt_length)}


def language_lora_pattern(model):
    """Regex selecting Linear layers of the language model only (vision encoder stays frozen)."""
    import torch

    leaves = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not any(part in name for part in EXCLUDED_MODULES):
            leaves.add(name.rsplit(".", 1)[-1])
    if not leaves:
        raise ValueError("No language-model Linear layers found for LoRA")
    excluded = "|".join(EXCLUDED_MODULES)
    return rf"^(?!.*({excluded})).*\.({'|'.join(sorted(leaves))})$", sorted(leaves)
