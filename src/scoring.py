from collections import Counter
import unicodedata


SCORING_ID = "boxed-strict-em-v1"


def boxed_answer(text):
    if not isinstance(text, str):
        return None
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    start += len(marker)
    depth = 1
    for position in range(start, len(text)):
        if text[position] == "{" and (position == 0 or text[position - 1] != "\\"):
            depth += 1
        elif text[position] == "}" and (position == 0 or text[position - 1] != "\\"):
            depth -= 1
            if depth == 0:
                answer = text[start:position].strip()
                return answer if answer else None
    return None


def normalize_answer(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().strip().rstrip(".!?").split())


def token_f1(prediction, reference):
    left = Counter(normalize_answer(prediction).split())
    right = Counter(normalize_answer(reference).split())
    overlap = sum((left & right).values())
    if not left or not right or not overlap:
        return 0.0
    precision, recall = overlap / sum(left.values()), overlap / sum(right.values())
    return 2 * precision * recall / (precision + recall)


def score_answer(raw_response, reference):
    parsed = boxed_answer(raw_response)
    return {"parsed_answer": parsed, "parse_failed": parsed is None,
            "strict_correct": parsed is not None and parsed.lower().strip() == reference.lower().strip(),
            "normalized_correct": parsed is not None and normalize_answer(parsed) == normalize_answer(reference),
            "token_f1": token_f1(parsed, reference) if parsed is not None else 0.0}


def summarize(scored):
    total = len(scored)
    complete = [r for r in scored if r["status"] == "generated"]
    n = len(complete)
    values = {"strict_em_accuracy_pct": 100 * sum(r["strict_correct"] for r in complete) / n if n else None,
              "normalized_em_accuracy_pct": 100 * sum(r["normalized_correct"] for r in complete) / n if n else None,
              "mean_token_f1_pct": 100 * sum(r["token_f1"] for r in complete) / n if n else None}
    return {"selected_count": total, "generated_count": n,
            "error_count": sum(r["status"] == "generation_error" for r in scored),
            "missing_count": sum(r["status"] == "missing" for r in scored),
            "parse_failure_count": sum(r["parse_failed"] for r in complete),
            "token_limit_count": sum(r.get("hit_token_limit", False) for r in complete),
            "coverage_pct": 100 * n / total if total else None,
            "status": "complete" if total and n == total else "incomplete",
            **{key: value if total and n == total else None for key, value in values.items()},
            "generated_only_metrics": values,
            "generation_seconds": sum(r.get("generation_seconds", 0) for r in complete)}
