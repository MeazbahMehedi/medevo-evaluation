"""Freeze the evaluation protocol and the reserved test subsets before any test-set generation.

The output budget comes from the predeclared rule applied to the 119-question feedback run.
Run once:  python freeze_protocol.py --per-dataset 200
"""

import argparse
import json
import sys

from medevo_study.core import (DATA_DIR, DATASETS, OUTPUTS, PAPER_PROMPT, PRIMARY_FIELD, PRIMARY_SCORER,
                               PROTOCOL_DIR, SCORERS, code_hashes, data_hashes, file_sha256, load_records,
                               read_json, safe_error, stratified_sample, utc_now, write_json)
import evaluate_feedback as feedback

FEEDBACK_RUN = OUTPUTS / "qwen35_feedback_119_1024_v1"
SELECTION_SEED = 7
SELECTION_SALT = "reserved-test-subset-v1"


def decide_budget(override, reason):
    summary_path = FEEDBACK_RUN / "summary.json"
    summary = read_json(summary_path) if summary_path.exists() else None
    outcome = (summary or {}).get("decision", {}).get("outcome")
    if override is not None:
        if not reason:
            raise ValueError("--budget requires --reason so the deviation is recorded")
        return override, {"source": "manual override", "reason": reason, "feedback_outcome": outcome}
    if outcome == "adopt_1024":
        budget = 1024
    elif outcome == "retain_512":
        budget = 512
    else:
        raise ValueError(f"The 119-question feedback run has no decision (outcome={outcome!r}). "
                         "Finish it, or pass --budget with --reason.")
    return budget, {"source": "predeclared feedback rule", "feedback_run": str(FEEDBACK_RUN),
                    "feedback_summary_sha256": file_sha256(summary_path), "feedback_outcome": outcome,
                    "missing_boxes": summary["decision"]["missing_boxes"]}


def freeze(per_dataset, override, reason):
    if PROTOCOL_DIR.exists():
        raise ValueError(f"{PROTOCOL_DIR} already exists. The protocol is frozen; do not overwrite it.")
    budget, decision = decide_budget(override, reason)
    plan = read_json(OUTPUTS / "qwen35_feedback_119_plan_v1" / "plan.json")
    metadata, records = load_records(DATA_DIR)
    subsets = {d: stratified_sample(records, "test", d, per_dataset, SELECTION_SEED, SELECTION_SALT) for d in DATASETS}
    used_elsewhere = {r["id"] for r in records if r["role"] != "test"}
    for dataset, rows in subsets.items():
        if {r["id"] for r in rows} & used_elsewhere or any(r["role"] != "test" for r in rows):
            raise ValueError(f"{dataset} subset contains non-test records")
    ordered = [row for pair in zip(*subsets.values()) for row in pair]
    longest = max(subsets.values(), key=len)
    ordered += [r for r in longest if r not in ordered]
    options = dict(feedback.OPTIONS, num_predict=budget)
    protocol = {
        "protocol_id": "medevo-study-frozen-v1", "frozen_utc": utc_now(),
        "prompt": PAPER_PROMPT, "prompt_source": "MedEvo paper, Appendix D.1",
        "output_token_budget": budget, "budget_decision": decision,
        "baseline_model": {"backend": "ollama", "model": feedback.MODEL, "model_digest": plan["model_digest"],
                           "ollama_version": plan["ollama_version"], "think": False, "options": options},
        "primary_metric": {"scorer": PRIMARY_SCORER, "field": PRIMARY_FIELD,
                           "definition": "normalized exact match of the last \\boxed{} answer after removing "
                                         "\\text{} and \\textbf{} wrappers; missing box counts as wrong"},
        "secondary_metrics": [f"{s}:{f}" for s in SCORERS for f in ("normalized_correct", "strict_correct")]
                             + ["token F1", "boxed rate", "truncations", "seconds per question"],
        "uncertainty": "percentile bootstrap 95% CI (10,000 resamples, seed 7); paired bootstrap and exact "
                       "McNemar test for systems evaluated on the same questions",
        "test_selection": {"seed": SELECTION_SEED, "salt": SELECTION_SALT, "per_dataset": per_dataset,
                           "method": "proportional allocation over open/closed answer types",
                           "counts": {d: {k: sum(r["answer_type"] == k for r in rows) for k in ("open", "closed")}
                                      for d, rows in subsets.items()}},
        "test_ids": [r["id"] for r in ordered],
        "rules": ["Test results are never used for prompt, budget, scorer, collection, admission or "
                  "training choices.", "All systems are evaluated on exactly these IDs with this prompt and budget."],
        "dataset_file_sha256": data_hashes(), "manifest_sha256": metadata["manifest_sha256"],
        "code_sha256": code_hashes(),
    }
    PROTOCOL_DIR.mkdir(parents=True)
    write_json(PROTOCOL_DIR / "protocol.json", protocol)
    write_json(PROTOCOL_DIR / "test_selection.json", ordered)
    return {"status": "frozen", "folder": str(PROTOCOL_DIR), "output_token_budget": budget,
            "decision": decision["source"], "test_questions": len(ordered),
            "counts": protocol["test_selection"]["counts"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-dataset", type=int, default=200)
    parser.add_argument("--budget", type=int, choices=(512, 1024), help="Only if the feedback rule cannot decide")
    parser.add_argument("--reason", help="Required with --budget")
    args = parser.parse_args()
    try:
        print(json.dumps(freeze(args.per_dataset, args.budget, args.reason), indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "blocked", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
