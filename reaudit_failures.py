"""Phase 3, v2: re-audit the failed questions of training bank v1 with the MedEvo audit prompt.

Bank v1 audited failures with an instruction that did not state the trajectory had failed, and every
case was skipped. This run reuses the saved failed rollouts (no new rollouts), audits them with the
instruction aligned to MedEvo Appendix D.3, and applies the paper's continuation and admission rules.
The v2 bank combines the v1 correct rollouts with any admitted repairs. Bank v1 is left unchanged.

Example:  python reaudit_failures.py
Resumable: re-run the same command to continue.
"""

import argparse
import json
import sys
import time
import traceback

from medevo_study.collection import REPAIR_V2, repair_failed_rollout
from medevo_study.core import (DATA_DIR, OUTPUTS, PROTOCOL_DIR, append_jsonl, code_hashes, data_hashes, file_sha256,
                               load_image, load_records, read_json, read_jsonl, require_unchanged, safe_error, utc_now,
                               write_json, write_jsonl)

SOURCE = OUTPUTS / "training_bank_v1"
FOLDER = OUTPUTS / "training_bank_v2"


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

    def nll(self, record, prefill, ids, include_input):
        return self.backend.continuation_nll(self.model, self.processor, self._image(record),
                                             record["question"] + self.suffix, prefill, ids, include_input)

    def ids(self, text):
        return self.backend.text_ids(self.processor, text)


class Auditor:
    def __init__(self, protocol, think):
        from medevo_study import ollama_backend
        self.backend = ollama_backend
        self.think = think
        self.info = ollama_backend.server_identity()
        if self.info["model_digest"] != protocol["baseline_model"]["model_digest"]:
            raise ValueError("Auditor model digest differs from the frozen protocol")

    def audit(self, record, steps):
        _, _, value, seconds = self.backend.audit_v2(DATA_DIR, record, steps, think=self.think)
        decision, reason = self.backend.validate_audit_v2(value, steps)
        return decision, reason, value, seconds


def failed_cases():
    cases = read_jsonl(SOURCE / "cases.jsonl")
    if not cases:
        raise ValueError("Training bank v1 not found; run build_training_bank.py first")
    return cases, [c for c in cases if c["outcome"] == "audit_skip"]


def export(source_cases, repaired, records):
    by_id = {r["id"]: r for r in records}
    bank = []

    def add(case_id, item):
        record = by_id[case_id]
        bank.append({"id": case_id, "kind": item["kind"], "dataset": record["dataset"],
                     "answer_type": record["answer_type"], "question": record["question"],
                     "image_path": record["image_path"], "image_file_sha256": record["image_file_sha256"],
                     "target_text": item["target_text"], "role": record["role"]})

    for case in source_cases:
        for item in case["admitted"]:
            add(case["id"], item)
    for case in repaired:
        for item in case["admitted"]:
            add(case["id"], item)
    write_jsonl(FOLDER / "training_bank.jsonl", bank)
    return bank


def run(max_hours, think):
    protocol = read_json(PROTOCOL_DIR / "protocol.json")
    require_unchanged(protocol["dataset_file_sha256"], data_hashes(), "Prepared data")
    source_cases, targets = failed_cases()
    _, records = load_records(DATA_DIR)
    by_id = {r["id"]: r for r in records}
    settings = {"source_cases_sha256": file_sha256(SOURCE / "cases.jsonl"), "target_ids": [c["id"] for c in targets],
                "repair": REPAIR_V2, "auditor_think": think,
                "protocol_sha256": file_sha256(PROTOCOL_DIR / "protocol.json")}
    if FOLDER.exists():
        state = read_json(FOLDER / "run.json")
        if any(state[k] != v for k, v in settings.items()):
            raise ValueError("training_bank_v2 was started with different settings; it cannot be resumed")
        require_unchanged(state["code_sha256"], code_hashes(), "Code")
    else:
        FOLDER.mkdir(parents=True)
        state = {**settings, "started_utc": utc_now(), "code_sha256": code_hashes(), "attempts": []}
    primary = PrimaryModel(protocol)
    auditor = Auditor(protocol, think)
    state["attempts"].append({"started_utc": utc_now(), "primary": primary.info, "auditor": auditor.info})
    write_json(FOLDER / "run.json", state)
    repaired = read_jsonl(FOLDER / "cases.jsonl")
    done = {c["id"] for c in repaired}
    import psutil
    started = time.perf_counter()
    for case in [c for c in targets if c["id"] not in done]:
        if max_hours and time.perf_counter() - started > max_hours * 3600:
            print("Time budget reached; re-run the same command to continue.", flush=True)
            break
        if psutil.virtual_memory().available / 1024 ** 3 < 3:
            print("Less than 3 GiB RAM available; stopping safely.", flush=True)
            break
        record = by_id[case["id"]]
        index = case["audit"]["rollout"]
        failed_text = case["rollouts"][index]["raw_response"]
        t0 = time.perf_counter()
        try:
            result = repair_failed_rollout(record, failed_text, primary, auditor)
        except KeyboardInterrupt:
            print("Interrupted; re-run the same command to continue.", flush=True)
            break
        except Exception as failure:
            result = {"outcome": "error", "error": safe_error(failure), "admitted": []}
        entry = {"id": case["id"], "dataset": case["dataset"], "answer_type": case["answer_type"],
                 "source_rollout": index, **result, "seconds": round(time.perf_counter() - t0, 1),
                 "recorded_utc": utc_now()}
        append_jsonl(FOLDER / "cases.jsonl", entry)
        repaired.append(entry)
        print(f"[{len(repaired)}/{len(targets)}] {case['id']}: {entry['outcome']} ({entry['seconds']:.0f}s)", flush=True)
    state["attempts"][-1]["ended_utc"] = utc_now()
    write_json(FOLDER / "run.json", state)
    bank = export(source_cases, repaired, records)
    outcomes = {}
    for entry in repaired:
        outcomes[entry["outcome"]] = outcomes.get(entry["outcome"], 0) + 1
    summary = {"status": "complete" if len(repaired) == len(targets) else "incomplete",
               "reaudited": len(repaired), "targets": len(targets), "outcomes": dict(sorted(outcomes.items())),
               "admitted_repairs": sum(e["outcome"] == "repair_admitted" for e in repaired),
               "admitted_correct_rollouts": sum(len(c["admitted"]) for c in source_cases),
               "bank_size": len(bank), "bank_file": str(FOLDER / "training_bank.jsonl"), "updated_utc": utc_now()}
    write_json(FOLDER / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-hours", type=float, default=0)
    parser.add_argument("--auditor-think", action="store_true", help="Enable Qwen3.5 thinking for audits (slower)")
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.max_hours, args.auditor_think), indent=2))
        return 0
    except Exception as error:
        traceback.print_exc()
        print(json.dumps({"status": "blocked", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
