import hashlib
import math
import re
import time

from .model import user_messages


READOUT_INSTRUCTION = ("Answer the original question directly using a word, number, or short phrase. "
                       "Return only the answer, without reasoning, labels, or formatting.")


def assistant_message(text):
    return {"role": "assistant", "content": [{"type": "text", "text": text}]}


def reasoning_steps(text):
    ends = [match.end() for match in re.finditer(r"[.!?](?:[\"')\]]*)(?=\s|$)|\n\s*\n", text)]
    if not ends or ends[-1] != len(text):
        ends.append(len(text))
    start, steps = 0, []
    for end in sorted(set(ends)):
        if text[start:end].strip():
            steps.append({"step": len(steps) + 1, "start": start, "end": end, "text": text[start:end]})
            start = end
    if start < len(text) and steps:
        steps[-1].update(end=len(text), text=text[steps[-1]["start"]:])
    return steps


def entropy_proposal(text, token_ends, entropies):
    steps = reasoning_steps(text)
    if not steps or token_ends is None or len(token_ends) != len(entropies):
        return {"status": "unavailable", "reason": "visible token/character alignment unavailable"}
    values = []
    previous_end = 0
    token_positions = []
    for end, entropy in zip(token_ends, entropies):
        if end > previous_end:
            token_positions.append((end - 1, entropy))
        previous_end = end
    for step in steps:
        scores = [value for position, value in token_positions if step["start"] <= position < step["end"]]
        if not scores or any(not math.isfinite(value) or value < 0 for value in scores):
            return {"status": "unavailable", "reason": "a step has no valid visible-token entropy"}
        ordered = sorted(scores)
        position = .9 * (len(scores) - 1)
        low, high = math.floor(position), math.ceil(position)
        quantile = ordered[low] + (position - low) * (ordered[high] - ordered[low])
        mean = sum(scores) / len(scores)
        values.append({"step": step["step"], "visible_tokens": len(scores), "mean_nats": mean,
                       "q90_nats": quantile, "uncertainty": .5 * mean + .5 * quantile})
    uncertainty = [value["uncertainty"] for value in values]
    surge = [0.0] + [uncertainty[i] - sum(uncertainty[max(0, i - 3):i]) / min(3, i)
                     for i in range(1, len(values))]
    def normalized(sequence):
        low, high = min(sequence), max(sequence)
        return [(value - low) / (high - low) if high > low else 0.0 for value in sequence]
    normalized_uncertainty, normalized_surge = normalized(uncertainty), normalized(surge)
    for index, value in enumerate(values):
        value.update(surge=surge[index], normalized_uncertainty=normalized_uncertainty[index],
                     normalized_surge=normalized_surge[index])
    anchor = max(range(len(values)), key=lambda i: normalized_uncertainty[i] + normalized_surge[i]) + 1
    return {"status": "measured", "lambda": .5, "history": 3, "steps": values,
            "anchor": anchor, "window": list(range(max(1, anchor - 1), min(len(values), anchor + 1) + 1)),
            "alignment_policy": "visible token assigned by last decoded character; zero-length special tokens excluded"}


def generate_chat(model, processor, messages, max_tokens, prefill=None, capture_entropy=False):
    import torch

    kwargs = {"add_generation_prompt": prefill is None, "continue_final_message": prefill is not None}
    rendered = processor.apply_chat_template(messages, tokenize=False, **kwargs)
    if prefill is not None and not rendered.endswith(prefill):
        raise ValueError("The chat template changed the assistant prefill")
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", **kwargs)
    if prefill is not None:
        decoded_input = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
        if not decoded_input.endswith(prefill):
            raise ValueError("The tokenized assistant prefill is not verbatim")
    length = inputs["input_ids"].shape[1]
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False, use_cache=True,
                                return_dict_in_generate=True, output_logits=capture_entropy)
    if not torch.equal(output.sequences[:, :length], inputs["input_ids"]):
        raise ValueError("Generation changed the input token prefix")
    tokens = output.sequences[0, length:]
    text = processor.decode(tokens, skip_special_tokens=True)
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else [eos])
    ended = bool(tokens.numel()) and int(tokens[-1]) in eos_ids
    result = {"raw_response": text, "generated_tokens": tokens.numel(),
              "generation_seconds": round(time.perf_counter() - started, 3),
              "hit_token_limit": tokens.numel() >= max_tokens and not ended,
              "input_token_count": length,
              "input_ids_sha256": hashlib.sha256(inputs["input_ids"].cpu().numpy().tobytes()).hexdigest(),
              "input_prefix_unchanged": True,
              "assistant_prefill_verified": prefill is not None}
    if capture_entropy:
        entropies = []
        for logits in output.logits:
            log_probabilities = logits[0].float().log_softmax(dim=-1)
            probabilities = log_probabilities.exp()
            entropy = -torch.where(probabilities > 0, probabilities * log_probabilities, 0).sum()
            entropies.append(float(entropy))
        ends, valid = [], True
        for count in range(1, tokens.numel() + 1):
            decoded = processor.decode(tokens[:count], skip_special_tokens=True)
            if not text.startswith(decoded):
                valid = False
            ends.append(len(decoded))
        if ends != sorted(ends):
            valid = False
        result.update(token_entropies_nats=entropies,
                      entropy=entropy_proposal(text, ends if valid else None, entropies))
    return result


def generate_reasoning(model, processor, image, question, instruction, max_tokens):
    return generate_chat(model, processor, user_messages(image, question + "\n" + instruction),
                         max_tokens, capture_entropy=True)


def answer_readout(model, processor, image, question, instruction, reasoning, max_tokens):
    messages = user_messages(image, question + "\n" + instruction) + [assistant_message(reasoning),
                {"role": "user", "content": [{"type": "text", "text": READOUT_INSTRUCTION}]}]
    return generate_chat(model, processor, messages, max_tokens)


def continue_reasoning(model, processor, image, question, instruction, prefix, hint, max_tokens):
    cue = "\nLet me recheck the evidence. " + hint + "\n"
    prefill = prefix + cue
    messages = user_messages(image, question + "\n" + instruction) + [assistant_message(prefill)]
    result = generate_chat(model, processor, messages, max_tokens, prefill=prefill)
    result.update(preserved_prefix=prefix, recovery_cue=cue, assistant_prefill=prefill,
                  suffix=result["raw_response"], complete_response=prefill + result["raw_response"],
                  retained_prefix_unchanged=True)
    return result
