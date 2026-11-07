from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .benchmarks import file_sha256, load_bundle, write_jsonl
from .evaluation import (indexed_predictions, prediction_journal, recorded_source_hashes,
                         select_records, source_fingerprints)
from .runtime import write_json
from .scoring import SCORING_ID, score_answer, summarize


PLAIN_SCORING_ID = "plain-whole-answer-em-v1"
NORMALIZATION = "lowercase; collapse whitespace; remove terminal .!? and outer whitespace"


def normalize_plain(text):
    return " ".join(text.lower().split()).rstrip(".!?").rstrip()


def score_predictions(selected, predictions):
    indexed = indexed_predictions(predictions, [r["id"] for r in selected])
    scored = []
    for reference in selected:
        prediction = indexed.get(reference["id"], {"status": "missing"})
        allowed = ("id", "status", "raw_response", "generated_tokens", "generation_seconds", "hit_token_limit", "error")
        row = {"id": reference["id"], "dataset": reference["dataset"],
               "question": reference["question"], "answer_type": reference["answer_type"],
               "reference": reference["answer"],
               **{key: prediction[key] for key in allowed if key in prediction}}
        row["plain_whole_answer_correct"] = None
        if row["status"] == "generated":
            row.update(score_answer(row["raw_response"], reference["answer"]))
            row["normalized_whole_response"] = normalize_plain(row["raw_response"])
            row["normalized_plain_reference"] = normalize_plain(reference["answer"])
            row["plain_whole_answer_correct"] = bool(row["normalized_whole_response"]) and (
                row["normalized_whole_response"] == row["normalized_plain_reference"])
        scored.append(row)
    return scored


def diagnostic_summary(scored):
    boxed = summarize(scored)
    count = boxed["generated_count"]
    correct = sum(row["plain_whole_answer_correct"] is True for row in scored)
    return {"status": boxed["status"], "selected_count": len(scored),
            "generated_count": count, "missing_count": boxed["missing_count"],
            "error_count": boxed["error_count"], "coverage_pct": boxed["coverage_pct"],
            "boxed_strict_em_accuracy_pct": boxed["strict_em_accuracy_pct"],
            "boxed_parse_failure_count": boxed["parse_failure_count"],
            "plain_whole_answer_em_accuracy_pct": 100 * correct / len(scored)
                if boxed["status"] == "complete" else None,
            "plain_generated_only_em_accuracy_pct": 100 * correct / count if count else None,
            "token_limit_count": boxed["token_limit_count"], "generation_seconds": boxed["generation_seconds"]}


def read_saved_run(root, data_dir, directory, dataset, branch, comparison, metadata, records):
    directory = Path(directory).resolve()
    if (directory / "run.lock").exists():
        raise ValueError("A source run is locked; wait until its worker has stopped")
    names = ("run.json", "predictions.jsonl", "metrics.json", "scored.jsonl", "resources.json")
    hashes = {name: file_sha256(directory / name) for name in names}
    if branch == "original":
        expected = comparison["baseline_evidence"][dataset]["file_sha256"]
        if hashes != expected:
            raise ValueError("Original baseline evidence differs from the completed comparison")
    signature = json.loads((directory / "run.json").read_text(encoding="utf-8"))["signature"]
    ids = comparison["datasets"][dataset]["selected_ids"]
    if signature["role"] != "feedback" or signature["benchmark"] != dataset:
        raise ValueError("Answer diagnostics accept feedback runs from the recorded dataset only")
    if signature["limit"] != 5 or signature["scope"] != "subset" or len(ids) != 5:
        raise ValueError("This diagnostic requires the five-question prompt comparison")
    selected = select_records(records, dataset, "feedback", metadata["protocol"]["seed"], 5)
    if ids != signature["selected_ids"] or ids != [r["id"] for r in selected]:
        raise ValueError("Saved IDs do not match the fixed feedback selection")
    expected_protocol = dict(metadata["protocol"])
    if branch == "trial":
        expected_protocol.update(protocol_id=metadata["protocol"]["protocol_id"] + "+" + comparison["experiment_id"],
                                 prompt=comparison["trial_prompt"])
    if signature["protocol"] != expected_protocol or signature["settings"] != comparison["settings"]:
        raise ValueError("Saved prompt, generation budget or model settings disagree with the comparison")
    if signature["dataset_metadata_sha256"] != comparison["dataset_metadata_sha256"]:
        raise ValueError("Saved run has a different dataset identity")
    recorded = recorded_source_hashes(signature)
    current = source_fingerprints(root)
    for name in ("src/model.py", "src/scoring.py"):
        if name not in current or recorded.get(name) != current[name]:
            raise ValueError("The original model or boxed scorer implementation changed")
    scored = score_predictions(selected, prediction_journal(directory / "predictions.jsonl"))
    computed = summarize(scored)
    if computed["status"] != "complete":
        raise ValueError("The saved run is incomplete; use a completed five-question comparison")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    recorded_summary = comparison["datasets"][dataset]["baseline" if branch == "original" else "trial"]["summary"]
    if metrics["summary"] != computed or recorded_summary != computed:
        raise ValueError("Saved boxed scores disagree with the raw predictions or completed comparison")
    return scored, {"directory": str(directory), "file_sha256": hashes}


def write_report(directory, report):
    write_json(directory / "diagnostic.json", report)
    lines = ["# Whole-answer scoring diagnostic", "", "Exploratory feedback analysis of saved predictions; no new inference or training.", "",
             "| Dataset | Prompt | Questions | Coverage % | Boxed strict EM % | Whole-answer EM % |",
             "|---|---|---:|---:|---:|---:|"]
    def cell(value):
        return "incomplete" if value is None else f"{value:.1f}"
    for dataset, values in report["datasets"].items():
        for branch, result in values.items():
            lines.append(f"| {dataset} | {branch} | {result['selected_count']} | {cell(result['coverage_pct'])} | "
                         f"{cell(result['boxed_strict_em_accuracy_pct'])} | {cell(result['plain_whole_answer_em_accuracy_pct'])} |")
    lines.extend(["", "Whole-answer normalization: " + NORMALIZATION + ".", "",
        "The entire raw response is compared with the entire reference. There is no extraction, fallback, synonym mapping, substring search or number conversion.",
        "A boxed answer is not unwrapped by the whole-answer method. Additional explanation or qualifiers remain part of the response and may cause a mismatch.", "",
        "Original boxed scores are verified and retained unchanged. Per-question results are in original/trial JSONL files beside this report.", "",
        "Changing the scoring rule does not train a model or demonstrate a MedEvo improvement. These ten reused feedback questions do not estimate final benchmark performance.",
        "The diagnostic does not adopt a new final-evaluation protocol or supply the reasoning trajectories needed for repair training."])
    (directory / "diagnostic.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def diagnose_answers(root, data_dir, comparison_dir):
    root, data_dir, comparison_dir = Path(root).resolve(), Path(data_dir).resolve(), Path(comparison_dir).resolve()
    comparison_path = comparison_dir / "comparison.json"
    comparison_hash = file_sha256(comparison_path)
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison["status"] != "complete" or comparison["role"] != "feedback":
        raise ValueError("Use a completed feedback prompt comparison for this diagnostic")
    metadata_hash = file_sha256(data_dir / "dataset.json")
    if comparison["dataset_metadata_sha256"] != metadata_hash:
        raise ValueError("The comparison and prepared dataset identities differ")
    metadata, records = load_bundle(data_dir)
    if comparison["original_prompt"] != metadata["protocol"]["prompt"]:
        raise ValueError("The recorded original prompt differs from the prepared protocol")
    report = {"purpose": "exploratory whole-answer scoring of saved feedback predictions", "role": "feedback",
              "boxed_scoring_id": SCORING_ID, "plain_scoring_id": PLAIN_SCORING_ID,
              "plain_normalization": NORMALIZATION, "status": "complete", "comparison_folder": str(comparison_dir),
              "comparison_sha256": comparison_hash, "dataset_metadata_sha256": metadata_hash,
              "source_sha256": source_fingerprints(root), "source_runs": {}, "datasets": {},
              "original_prompt": comparison["original_prompt"], "trial_prompt": comparison["trial_prompt"],
              "new_inference_performed": False, "original_artifacts_modified": False,
              "adopted_for_final_evaluation": False}
    results = {}
    for dataset in ("vqa_rad", "slake"):
        results[dataset], report["source_runs"][dataset], report["datasets"][dataset] = {}, {}, {}
        for branch in ("original", "trial"):
            source = comparison_dir / "baseline" / dataset if branch == "original" else Path(comparison["datasets"][dataset]["trial_run"])
            scored, evidence = read_saved_run(root, data_dir, source, dataset, branch, comparison, metadata, records)
            results[dataset][branch] = scored
            report["source_runs"][dataset][branch] = evidence
            report["datasets"][dataset][branch] = {**diagnostic_summary(scored),
                "by_answer_type": {kind: diagnostic_summary([r for r in scored if r["answer_type"] == kind])
                                   for kind in ("open", "closed")}}
    if file_sha256(comparison_path) != comparison_hash or file_sha256(data_dir / "dataset.json") != metadata_hash:
        raise ValueError("Input comparison or data metadata changed during diagnosis")
    for sources in report["source_runs"].values():
        for evidence in sources.values():
            for name, expected in evidence["file_sha256"].items():
                if file_sha256(Path(evidence["directory"]) / name) != expected:
                    raise ValueError("A saved run changed during diagnosis")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "outputs" / f"{stamp}_answer_diagnostic_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    report["diagnostic_folder"] = str(directory)
    for dataset, branches in results.items():
        for branch, scored in branches.items():
            write_jsonl(directory / f"{dataset}_{branch}.jsonl", scored)
    write_report(directory, report)
    return {"status": report["status"], "diagnostic_folder": str(directory),
            "plain_scoring_id": PLAIN_SCORING_ID, "new_inference_performed": False,
            "datasets": {dataset: {branch: {
                key: result[key] for key in ("selected_count", "coverage_pct", "boxed_strict_em_accuracy_pct",
                                              "boxed_parse_failure_count", "plain_whole_answer_em_accuracy_pct")}
                for branch, result in values.items()} for dataset, values in report["datasets"].items()}}
