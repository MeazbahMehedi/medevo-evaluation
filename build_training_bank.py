"""Phase 3: build the admitted training bank from training-role questions only.

For each question the primary model samples rollouts. A clean correct rollout is admitted. If all
rollouts fail, Qwen3.5-9B audits one failure without the reference answer, the primary model continues
from the supported prefix with the hint, and the repair is admitted only if it passes every check.

Example:  python build_training_bank.py --questions 100 --max-hours 8
Resumable: re-run the same command to continue.
"""

import argparse
import json
import sys
import time

from medevo_study.collection import ADMISSION, process_question
from medevo_study.core import (DATA_DIR, OUTPUTS, PROTOCOL_DIR, append_jsonl, code_hashes, data_hashes, file_sha256, load_image,
                               load_records, read_json, read_jsonl, require_unchanged, safe_error, stratified_sample,
                               utc_now, write_json, write_jsonl)

FOLDER = OUTPUTS / "training_bank_v1"
SALT = "training-bank-v1"


class PrimaryModel:
    def __init__(self, protocol):
        from medevo_study import transformers_backend as backend
        settings = read_json(PROTOCOL_DIR / "primary_model.json")
        self.backend = backend
        self.budget = protocol["output_token_budget"]
        self.suffix = "\n" + protocol["prompt"]
        self.model, self.processor, self.info = backend.load(settings["model_id"], settings["revision"],
                                                             settings["dtype"], settings["threads"])
        self.images = {}

    def _image(self, record):
        if record["id"] not in self.images:
            self.images = {record["id"]: load_image(DATA_DIR, record)}
        return self.images[record["id"]]

    def generate(self, record, prefill, sample, seed):
        return self.backend.generate(self.model, self.processor, self._image(record), record["question"] + self.suffix,
                                     self.budget, prefill=prefill, sample=sample, seed=seed)

    def nll(self, record, prefill, ids):
        return self.backend.continuation_nll(self.model, self.processor, self._image(record),
                                             record["question"] + self.suffix, prefill, ids)


class Auditor:
    def __init__(self, protocol, think):
        from medevo_study import ollama_backend
        self.backend = ollama_backend
        self.think = think
        identity = ollama_backend.server_identity()
        if identity["model_digest"] != protocol["baseline_model"]["model_digest"]:
            raise ValueError("Auditor model digest differs from the frozen protocol")
        self.info = identity

    def audit(self, record, response_text, steps):
        payload, reply, value, seconds = self.backend.audit(DATA_DIR, record, response_text, steps, think=self.think)
        decision, reason = self.backend.validate_audit(value, steps)
        return decision, reason, value, seconds


def strip_ids(case):
    for rollout in case["rollouts"]:
        rollout.pop("generated_token_ids", None)
    return case


def summarize(cases):
    outcomes = {}
    for case in cases:
        outcomes[case["outcome"]] = outcomes.get(case["outcome"], 0) + 1
    admitted = [a for c in cases for a in c["admitted"]]
    return {"questions_processed": len(cases), "outcomes": dict(sorted(outcomes.items())),
            "admitted_total": len(admitted),
            "admitted_correct_rollouts": sum(a["kind"] == "correct_rollout" for a in admitted),
            "admitted_repairs": sum(a["kind"] == "repair" for a in admitted),
            "audits_run": sum("audit" in c for c in cases),
            "by_dataset": {d: sum(len(c["admitted"]) for c in cases if c["dataset"] == d) for d in ("vqa_rad", "slake")}}


def export_bank(cases, records):
    by_id = {r["id"]: r for r in records}
    bank = []
    for case in cases:
        record = by_id[case["id"]]
        for item in case["admitted"]:
            bank.append({"id": case["id"], "kind": item["kind"], "dataset": record["dataset"],
                         "answer_type": record["answer_type"], "question": record["question"],
                         "image_path": record["image_path"], "image_file_sha256": record["image_file_sha256"],
                         "target_text": item["target_text"], "role": record["role"]})
    write_jsonl(FOLDER / "training_bank.jsonl", bank)
    return bank


def build(questions, max_hours, think):
    protocol = read_json(PROTOCOL_DIR / "protocol.json")
    require_unchanged(protocol["dataset_file_sha256"], data_hashes(), "Prepared data")
    _, records = load_records(DATA_DIR)
    selected = []
    for dataset in ("vqa_rad", "slake"):
        selected += stratified_sample(records, "train", dataset, questions // 2, 7, SALT)
    if any(r["role"] != "train" for r in selected):
        raise ValueError("Only training-role questions may be used")
    settings = {"question_ids": [r["id"] for r in selected], "admission": ADMISSION, "auditor_think": think,
                "protocol_sha256": file_sha256(PROTOCOL_DIR / "protocol.json")}
    if FOLDER.exists():
        run = read_json(FOLDER / "run.json")
        if any(run[k] != v for k, v in settings.items()):
            raise ValueError("training_bank_v1 was started with different settings; it cannot be resumed")
        require_unchanged(run["code_sha256"], code_hashes(), "Code")
    else:
        FOLDER.mkdir(parents=True)
        run = {**settings, "started_utc": utc_now(), "code_sha256": code_hashes(), "attempts": []}
        write_json(FOLDER / "selection.json", selected)
    primary = PrimaryModel(protocol)
    auditor = Auditor(protocol, think)
    run["attempts"].append({"started_utc": utc_now(), "primary": primary.info, "auditor": auditor.info})
    write_json(FOLDER / "run.json", run)
    cases = read_jsonl(FOLDER / "cases.jsonl")
    done = {c["id"] for c in cases}
    import psutil
    started = time.perf_counter()
    for record in [r for r in selected if r["id"] not in done]:
        if max_hours and time.perf_counter() - started > max_hours * 3600:
            print("Time budget reached; re-run the same command to continue.", flush=True)
            break
        if psutil.virtual_memory().available / 1024 ** 3 < 3:
            print("Less than 3 GiB RAM available; stopping safely.", flush=True)
            break
        t0 = time.perf_counter()
        try:
            case = strip_ids(process_question(record, primary, auditor))
        except KeyboardInterrupt:
            print("Interrupted; re-run the same command to continue.", flush=True)
            break
        except Exception as failure:
            case = {"id": record["id"], "dataset": record["dataset"], "answer_type": record["answer_type"],
                    "outcome": "error", "error": safe_error(failure), "rollouts": [], "admitted": []}
        case["seconds"] = round(time.perf_counter() - t0, 1)
        case["recorded_utc"] = utc_now()
        append_jsonl(FOLDER / "cases.jsonl", case)
        cases.append(case)
        print(f"[{len(cases)}/{len(selected)}] {record['id']}: {case['outcome']} ({case['seconds']:.0f}s)", flush=True)
    run["attempts"][-1]["ended_utc"] = utc_now()
    write_json(FOLDER / "run.json", run)
    bank = export_bank(cases, records)
    summary = {**summarize(cases), "selected_questions": len(selected), "bank_file": str(FOLDER / "training_bank.jsonl"),
               "status": "complete" if len(cases) == len(selected) else "incomplete", "updated_utc": utc_now()}
    write_json(FOLDER / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=int, default=100, help="Training questions, split evenly by dataset")
    parser.add_argument("--max-hours", type=float, default=0)
    parser.add_argument("--auditor-think", action="store_true", help="Enable Qwen3.5 thinking for audits (slower)")
    args = parser.parse_args()
    try:
        print(json.dumps(build(args.questions, args.max_hours, args.auditor_think), indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "blocked", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
