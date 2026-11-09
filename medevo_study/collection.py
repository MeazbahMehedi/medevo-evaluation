"""Phase 3: rollouts, screening, reference-free audit, prefix repair and admission for one question.

The reference answer is used only to verify correctness and to reject hints that reveal it.
It is never sent to the primary model or the auditor.
"""

from .core import hint_leaks_answer, integrity_problems, primary_correct, reasoning_steps

ADMISSION = {
    "rollouts_per_question": 4,
    "repair_attempts": 3,
    "max_hint_free_nll": 1.5,       # nats/token of the repaired suffix without the hint
    "max_nll_ratio": 2.0,           # hint-free NLL / hinted NLL
    "recovery_cue": "\nLet me recheck the evidence. {hint}\n",
}


def screen_correct(rollout, reference):
    problems = integrity_problems(rollout["raw_response"], rollout["hit_token_limit"])
    if not primary_correct(rollout["raw_response"], reference):
        problems.append("incorrect")
    return problems


def choose_failure(rollouts):
    """First failed rollout that finished and produced an answer box; else None."""
    for index, rollout in enumerate(rollouts):
        text = rollout["raw_response"]
        if not rollout["hit_token_limit"] and "\\boxed{" in text and not integrity_problems(text, False):
            return index
    return None


def check_suffix(suffix_text, hit_limit, reference, hint):
    problems = integrity_problems(suffix_text, hit_limit)
    lowered = suffix_text.lower()
    if "hint" in lowered:
        problems.append("mentions_hint")
    if len(hint.split()) >= 6 and hint.lower().strip(" .") in lowered:
        problems.append("copies_hint")
    if not primary_correct(suffix_text, reference):
        problems.append("incorrect_after_repair")
    return problems


def process_question(record, primary, auditor, config=ADMISSION):
    """primary: generate(record, prefill, sample, seed) and nll(record, prefill, ids).
    auditor: audit(record, response_text, steps) -> (decision, reason, raw_value, seconds)."""
    reference = record["answer"]
    case = {"id": record["id"], "dataset": record["dataset"], "answer_type": record["answer_type"],
            "rollouts": [], "admitted": []}
    for k in range(config["rollouts_per_question"]):
        result = primary.generate(record, prefill="", sample=True, seed=k + 1)
        result["correct"] = primary_correct(result["raw_response"], reference)
        case["rollouts"].append(result)

    correct = [i for i, r in enumerate(case["rollouts"]) if r["correct"]]
    case["correct_rollouts"] = len(correct)
    for index in correct:
        problems = screen_correct(case["rollouts"][index], reference)
        case["rollouts"][index]["screening"] = problems or ["passed"]
        if not problems:
            case["admitted"].append({"kind": "correct_rollout", "rollout": index,
                                     "target_text": case["rollouts"][index]["raw_response"]})
            break
    if correct:
        case["outcome"] = "correct_rollout_admitted" if case["admitted"] else "correct_rollouts_failed_screening"
        return case

    index = choose_failure(case["rollouts"])
    if index is None:
        case["outcome"] = "no_auditable_failure"
        return case
    failed = case["rollouts"][index]["raw_response"]
    steps = reasoning_steps(failed)
    decision, reason, value, seconds = auditor.audit(record, failed, steps)
    case["audit"] = {"rollout": index, "steps": len(steps), "decision": decision, "reason": reason,
                     "value": value, "seconds": seconds}
    if decision != "repair":
        case["outcome"] = f"audit_{decision}"
        return case
    hint = value["hint"].strip()
    if hint_leaks_answer(hint, reference):
        case["outcome"] = "hint_rejected_leak"
        return case
    step = value["first_error_step"]
    prefix = failed[:steps[step - 1]["start"]]
    cue = config["recovery_cue"].format(hint=hint)
    case["repair"] = {"prefix_characters": len(prefix), "first_error_step": step,
                      "error_type": value["error_type"], "attempts": []}
    for attempt in range(config["repair_attempts"]):
        result = primary.generate(record, prefill=prefix + cue, sample=True, seed=100 + attempt)
        suffix = result["raw_response"]
        problems = check_suffix(suffix, result["hit_token_limit"], reference, hint)
        entry = {"attempt": attempt, "suffix": suffix, "generated_tokens": result["generated_tokens"],
                 "seconds": result["generation_seconds"], "problems": problems}
        if not problems:
            ids = result["generated_token_ids"]
            hinted = primary.nll(record, prefix + cue, ids)
            free = primary.nll(record, prefix, ids)
            ratio = free / hinted if hinted else None
            entry.update(nll_hinted=hinted, nll_hint_free=free, nll_ratio=ratio)
            if free > config["max_hint_free_nll"]:
                problems.append("not_learnable_nll")
            if ratio is None or ratio > config["max_nll_ratio"]:
                problems.append("hint_dependent")
        case["repair"]["attempts"].append(entry)
        if not problems:
            case["admitted"].append({"kind": "repair", "attempt": attempt, "target_text": prefix + suffix,
                                     "prefix_characters": len(prefix), "error_type": value["error_type"]})
            case["outcome"] = "repair_admitted"
            return case
    case["outcome"] = "repair_attempts_rejected"
    return case


# ---------------------------------------------------------------- repair v2 (MedEvo Sections 3.2.2-3.2.4)

REPAIR_V2 = {
    "audit_instruction": "v2, aligned with MedEvo Appendix D.3",
    "continuation_attempts": 5,              # paper: up to five suffixes, stop at first admitted
    "repair_ratio_range": [0.35, 0.85],      # D_repair = L(s+|x,p_b) / L(s+|p_b), Eq. 10
    "repair_gap_range": [0.02, 0.30],        # Delta_repair = mean L(s+|x,p_b) - mean L(s-|x,p_b), Eq. 11
    "threshold_source": "paper values, not recalibrated on our feedback set",
    "recovery_cue": "\nLet me recheck the evidence. {hint}\n",
    "training_target": "prefix + recovery cue + new suffix (Eq. 13)",
}


def _finite(value):
    return value is not None and value == value and abs(value) != float("inf")


def repair_failed_rollout(record, failed_text, primary, auditor, config=REPAIR_V2):
    """primary: generate(record, prefill, sample, seed), nll(record, prefill, ids, include_input), ids(text).
    auditor: audit(record, steps) -> (decision, reason, value, seconds)."""
    reference = record["answer"]
    steps = reasoning_steps(failed_text)
    decision, reason, value, seconds = auditor.audit(record, steps)
    result = {"audit": {"steps": len(steps), "decision": decision, "reason": reason, "value": value,
                        "seconds": seconds}, "admitted": []}
    if decision != "repair":
        result["outcome"] = f"audit_{decision}"
        return result
    hint = value["hint"].strip()
    if hint_leaks_answer(hint, reference):
        result["outcome"] = "hint_rejected_leak"
        return result
    step = value["first_error_step"]
    prefix = failed_text[:steps[step - 1]["start"]]
    wrong_ids = primary.ids(failed_text[len(prefix):])
    cue = config["recovery_cue"].format(hint=hint)
    low_ratio, high_ratio = config["repair_ratio_range"]
    low_gap, high_gap = config["repair_gap_range"]
    result["repair"] = {"first_error_step": step, "safe_prefix_steps": step - 1, "prefix_characters": len(prefix),
                        "error_type": value["error_type"], "hint": hint, "attempts": []}
    wrong_nll = primary.nll(record, prefix, wrong_ids, True)
    for attempt in range(config["continuation_attempts"]):
        generated = primary.generate(record, prefill=prefix + cue, sample=True, seed=200 + attempt)
        suffix = generated["raw_response"]
        problems = check_suffix(suffix, generated["hit_token_limit"], reference, hint)
        entry = {"attempt": attempt, "suffix": suffix, "generated_tokens": generated["generated_tokens"],
                 "seconds": generated["generation_seconds"], "problems": problems}
        if not problems:
            ids = generated["generated_token_ids"]
            with_input = primary.nll(record, prefix, ids, True)
            without_input = primary.nll(record, prefix, ids, False)
            if not all(_finite(v) for v in (with_input, without_input, wrong_nll)) or without_input <= 0:
                problems.append("nonfinite_or_nonpositive_loss")
            else:
                ratio = with_input / without_input
                gap = with_input - wrong_nll
                entry.update(nll_with_input=with_input, nll_without_input=without_input,
                             nll_wrong_suffix=wrong_nll, repair_ratio=ratio, repair_gap=gap)
                if not low_ratio <= ratio <= high_ratio:
                    problems.append("repair_ratio_out_of_range")
                if not low_gap <= gap <= high_gap:
                    problems.append("repair_gap_out_of_range")
        result["repair"]["attempts"].append(entry)
        if not problems:
            result["admitted"].append({"kind": "repair", "attempt": attempt, "target_text": prefix + cue + suffix,
                                       "prefix_characters": len(prefix), "error_type": value["error_type"]})
            result["outcome"] = "repair_admitted"
            return result
    result["outcome"] = "repair_attempts_rejected"
    return result
