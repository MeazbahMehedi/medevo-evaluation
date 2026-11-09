"""Shared helpers: data, selection, scoring, statistics, journals and integrity checks."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re

from PIL import Image

from feedback_scoring import SCORING_ID as UNWRAPPED_SCORER, score_formatted
from src.benchmarks import file_sha256, image_path, load_bundle
from src.runtime import environment_snapshot, safe_error, write_json
from src.scoring import SCORING_ID as ORIGINAL_SCORER, normalize_answer, score_answer


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "benchmarks"
OUTPUTS = ROOT / "outputs"
PROTOCOL_DIR = OUTPUTS / "frozen_protocol_v1"
PAPER_PROMPT = r"Please reason step by step, and put your final answer within \boxed{}."
DATASETS = ("vqa_rad", "slake")
ANSWER_TYPES = ("open", "closed")
IMAGE_EDGE = 512

# Primary metric chosen on feedback data before any test generation:
# normalized exact match of the last boxed answer after removing \text{}/\textbf{} wrappers.
PRIMARY_SCORER = UNWRAPPED_SCORER
PRIMARY_FIELD = "normalized_correct"

# Amendment 01 (secondary metric, decided on feedback data before any test generation):
# for questions whose reference is yes/no, a boxed 1/0 (or true/false) is read as yes/no.
BINARY_SCORER = "boxed-binary-mapped-em-v1"
BINARY_MAP = {"1": "yes", "0": "no", "true": "yes", "false": "no"}


def score_binary_mapped(raw_response, reference):
    result = score_formatted(raw_response, reference)
    parsed = result["parsed_answer"]
    if parsed is None or normalize_answer(reference) not in ("yes", "no"):
        return result
    mapped = BINARY_MAP.get(normalize_answer(parsed).strip("$ "))
    if mapped is None:
        return result
    return {**score_answer(r"\boxed{" + mapped + "}", reference), "parsed_answer": parsed, "mapped_answer": mapped}


SCORERS = {ORIGINAL_SCORER: score_answer, UNWRAPPED_SCORER: score_formatted, BINARY_SCORER: score_binary_mapped}

# Paper Table (Qwen3.5-9B, base -> MedEvo R3), accuracy %.
PAPER_REFERENCE = {"vqa_rad": {"base": 68.10, "medevo": 70.40},
                   "slake": {"base": 76.80, "medevo": 78.23}}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def code_hashes():
    """Hashes of the code that determines results. Checked on resume and before reporting."""
    names = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "medevo_study").glob("*.py"))
    names += ["src/scoring.py", "feedback_scoring.py", "src/benchmarks.py", "evaluate_feedback.py"]
    return {name: file_sha256(ROOT / name) for name in names}


def data_hashes(data_dir=DATA_DIR):
    return {name: file_sha256(Path(data_dir) / name) for name in
            ("dataset.json", "train.jsonl", "feedback.jsonl", "test.jsonl", "exclusions.jsonl")}


def require_unchanged(recorded, current, what):
    changed = sorted(k for k in set(recorded) | set(current) if recorded.get(k) != current.get(k))
    if changed:
        raise ValueError(f"{what} changed since this run started: {', '.join(changed[:6])}")


def load_records(data_dir=DATA_DIR):
    metadata, records = load_bundle(data_dir, verify_images=False)
    return metadata, records


def seeded_order(rows, seed, salt):
    return sorted(rows, key=lambda r: hashlib.sha256(f"{seed}:{salt}:{r['id']}".encode()).hexdigest())


def stratified_sample(records, role, dataset, size, seed, salt):
    """Proportional allocation over answer types, deterministic order inside each stratum."""
    pool = [r for r in records if r["role"] == role and r["dataset"] == dataset]
    if size >= len(pool):
        return seeded_order(pool, seed, salt)
    strata = {kind: seeded_order([r for r in pool if r["answer_type"] == kind], seed, salt)
              for kind in ANSWER_TYPES}
    exact = {kind: size * len(rows) / len(pool) for kind, rows in strata.items()}
    counts = {kind: int(value) for kind, value in exact.items()}
    # Largest-remainder rounding, ties broken by name for determinism.
    for kind in sorted(exact, key=lambda k: (-(exact[k] - counts[k]), k))[:size - sum(counts.values())]:
        counts[kind] += 1
    chosen = []
    for kind in ANSWER_TYPES:
        chosen.extend(strata[kind][:counts[kind]])
    return seeded_order(chosen, seed, salt + ":order")


def load_image(data_dir, record):
    path = image_path(Path(data_dir), record["image_path"])
    if file_sha256(path) != record["image_file_sha256"]:
        raise ValueError(f"Image changed for {record['id']}")
    with Image.open(path) as image:
        clean = Image.frombytes("RGB", image.size, image.convert("RGB").tobytes())
    clean.thumbnail((IMAGE_EDGE, IMAGE_EDGE))
    return clean


def score_all(raw_response, reference):
    return {label: scorer(raw_response, reference) for label, scorer in SCORERS.items()}


def primary_correct(raw_response, reference):
    return bool(score_formatted(raw_response, reference)[PRIMARY_FIELD])


# ---------------------------------------------------------------- statistics

def bootstrap_ci(values, resamples=10000, seed=7, level=0.95):
    values = [float(v) for v in values]
    if not values:
        return None
    n = len(values)
    try:
        import numpy
        rng = numpy.random.default_rng(seed)
        array = numpy.asarray(values)
        means = sorted(array[rng.integers(0, n, size=(resamples, n))].mean(axis=1).tolist())
    except ImportError:
        rng = random.Random(seed)
        means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    low = means[int((1 - level) / 2 * resamples)]
    high = means[min(resamples - 1, int((1 + level) / 2 * resamples))]
    return [100 * low, 100 * high]


def paired_bootstrap(base, other, resamples=10000, seed=7):
    """base/other: dicts id -> 0/1. Returns difference (other - base) in points with 95% CI."""
    ids = sorted(set(base) & set(other))
    if not ids:
        return None
    diffs = [float(other[i]) - float(base[i]) for i in ids]
    ci = bootstrap_ci(diffs, resamples, seed)
    b = sum(1 for i in ids if base[i] and not other[i])
    c = sum(1 for i in ids if other[i] and not base[i])
    return {"paired_count": len(ids), "difference_points": 100 * sum(diffs) / len(ids),
            "ci95_points": ci, "base_only_correct": b, "other_only_correct": c,
            "mcnemar_exact_p": mcnemar_exact(b, c)}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    from math import comb
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def summarize_scored(rows, field=PRIMARY_FIELD, scorer=PRIMARY_SCORER):
    """rows: scored items with 'status', 'scores', 'hit_token_limit', 'dataset', 'answer_type'."""
    total = len(rows)
    done = [r for r in rows if r.get("status") == "generated"]
    correct = [1 if r["scores"][scorer][field] else 0 for r in done]
    boxed = [0 if r["scores"][scorer]["parse_failed"] else 1 for r in done]
    complete = total > 0 and len(done) == total
    seconds = [r["generation_seconds"] for r in done if "generation_seconds" in r]
    tokens = [r["generated_tokens"] for r in done if "generated_tokens" in r]
    f1 = [r["scores"][scorer]["token_f1"] for r in done]
    return {"selected": total, "generated": len(done), "complete": complete,
            "accuracy_pct": 100 * sum(correct) / len(done) if done else None,
            "accuracy_ci95": bootstrap_ci(correct) if done else None,
            "correct": sum(correct), "boxed_rate_pct": 100 * sum(boxed) / len(done) if done else None,
            "truncated": sum(bool(r.get("hit_token_limit")) for r in done),
            "mean_token_f1_pct": 100 * sum(f1) / len(f1) if f1 else None,
            "mean_seconds": sum(seconds) / len(seconds) if seconds else None,
            "mean_tokens": sum(tokens) / len(tokens) if tokens else None}


def grouped_summary(rows):
    result = {}
    for scorer in SCORERS:
        for field in ("normalized_correct", "strict_correct"):
            key = f"{scorer}:{field}"
            block = {"overall": summarize_scored(rows, field, scorer), "datasets": {}}
            for dataset in DATASETS:
                part = [r for r in rows if r["dataset"] == dataset]
                block["datasets"][dataset] = {"all": summarize_scored(part, field, scorer)}
                for kind in ANSWER_TYPES:
                    block["datasets"][dataset][kind] = summarize_scored(
                        [r for r in part if r["answer_type"] == kind], field, scorer)
            result[key] = block
    return result


# ---------------------------------------------------------------- text checks

def reasoning_steps(text):
    """Sentence/paragraph spans, identical policy to src/repair_model.reasoning_steps."""
    ends = [m.end() for m in re.finditer(r"[.!?](?:[\"')\]]*)(?=\s|$)|\n\s*\n", text)]
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


def repetition_ratio(text, n=8):
    """Share of word 8-grams that are repeats. High values indicate looping output."""
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
    if not grams:
        return 0.0
    counts = Counter(grams)
    return sum(c - 1 for c in counts.values()) / len(grams)


YES_NO = {"yes", "no"}


def hint_leaks_answer(hint, reference):
    """Conservative leak guard. Gold is used only here and for verification, never sent to a model."""
    h = " " + normalize_answer(hint) + " "
    ref = normalize_answer(reference)
    if not ref:
        return False
    if ref in YES_NO:
        return bool(re.search(r"\b(answer|answer is|should be|is)\s*[:\-]?\s*(yes|no)\b", h)) or \
            bool(re.search(r"\b(yes|no)\s*[.!]?\s*$", h.strip()))
    tokens = [t for t in re.split(r"[\s,;/]+", ref) if len(t) > 2]
    if " " + ref + " " in h:
        return True
    return bool(tokens) and all(" " + t + " " in h for t in tokens)


def integrity_problems(text, max_tokens_hit):
    problems = []
    if max_tokens_hit:
        problems.append("truncated")
    if repetition_ratio(text) > 0.15:
        problems.append("repetitive")
    if "<think>" in text.lower() or "</think>" in text.lower():
        problems.append("thinking_markup")
    if text.count(r"\boxed{") > 3:
        problems.append("multiple_boxes")
    if len(text.split()) < 5:
        problems.append("too_short")
    return problems


def run_header(kind, extra):
    return {"kind": kind, "started_utc": utc_now(), "code_sha256": code_hashes(),
            "dataset_file_sha256": data_hashes(), "environment": environment_snapshot(ROOT), **extra}


__all__ = [name for name in dir() if not name.startswith("_")] + ["file_sha256", "safe_error", "write_json"]
