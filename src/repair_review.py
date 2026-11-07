from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from .benchmarks import file_sha256
from .evaluation import source_fingerprints
from .repair_audit import AUDIT_VALIDATION_POLICY as VALIDATION_POLICY, validate_medical_audit
from .repair_model import reasoning_steps
from .runtime import safe_error, write_json


def inspect_repair(root, pilot_dir):
    root, pilot_dir = Path(root).resolve(), Path(pilot_dir).resolve()
    if (pilot_dir / "run.lock").exists():
        raise ValueError("The pilot is locked; wait until its worker has stopped")
    inputs = {}
    def read(path):
        inputs[str(path)] = file_sha256(path)
        return json.loads(path.read_text(encoding="utf-8"))
    run = read(pilot_dir / "run.json")
    if run.get("pilot_id") != "training-repair-pilot-v1" or run.get("role") != "train":
        raise ValueError("Use a training-only repair-pilot folder")
    ids = run.get("selected_ids", [])
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError("The recorded pilot must contain three unique selected IDs")
    cases = []
    for index, identity in enumerate(ids, 1):
        directory = pilot_dir / "cases" / f"{index:02d}"
        original = read(directory / "case.json")
        if original.get("id") != identity or original.get("role") != "train":
            raise ValueError("Case identity or role differs from the pilot record")
        result = {"id": identity, "recorded_outcome": original["outcome"],
                  "review_outcome": "no_saved_audit_response", "new_continuation_performed": False}
        for name, field, key in (("initial_request.json", "question", "question"),
                                 ("initial_readout.json", "initial_answer_readout", "raw_response"),
                                 ("verification.json", "reference", "reference")):
            if (directory / name).exists():
                result[field] = read(directory / name).get(key)
        response_path = directory / "audit_response.json"
        if not response_path.exists():
            if original["outcome"] != "initial_answer_match":
                result["evidence_unavailable"] = True
            cases.append(result)
            continue
        initial = read(directory / "initial_reasoning.json")
        result["initial_reasoning"] = initial["raw_response"]
        steps = read(directory / "reasoning_steps.json")
        if steps != reasoning_steps(initial["raw_response"]):
            raise ValueError("Saved reasoning steps disagree with the captured response")
        saved_response = read(response_path)
        try:
            envelope = json.loads(saved_response["response_text"])
            choice = envelope["choices"][0]
            raw_audit = json.loads(choice["message"]["content"])
            audit = validate_medical_audit(raw_audit, steps)
            result.update(raw_audit=raw_audit, action=audit["action"],
                          provider_finish_reason=choice.get("finish_reason"), validation_passed=True)
            if audit["action"] == "repair":
                result.update(review_outcome="repair_decision_requires_remaining_checks",
                              ignored_auxiliary_fields=[], hint_used_for_generation=False)
            else:
                result.update(review_outcome="auditor_" + audit["action"],
                              ignored_auxiliary_fields=["first_error_step", "error_family", "prefix_supported", "hint"],
                              hint_used_for_generation=False)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            result.update(review_outcome="invalid_saved_audit", validation_passed=False, error=safe_error(error))
        cases.append(result)
    for name, digest in inputs.items():
        if file_sha256(name) != digest:
            raise ValueError("A source pilot artifact changed during inspection")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = root / "outputs" / f"{stamp}_audit_review_{uuid.uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    report = {"status": "complete", "purpose": "read-only review of saved auditor decisions; not a rerun or medical validation",
              "validation_policy": VALIDATION_POLICY, "source_pilot_folder": str(pilot_dir), "review_folder": str(directory),
              "selected_count": len(ids), "recorded_outcomes": dict(Counter(value["recorded_outcome"] for value in cases)),
              "review_outcomes": dict(Counter(value["review_outcome"] for value in cases)),
              "nonrepair_decision_count": sum(value.get("action") in ("skip", "unresolved") for value in cases),
              "schema_valid_repair_decision_count": sum(value.get("action") == "repair" for value in cases),
              "invalid_saved_audit_count": sum(value["review_outcome"] == "invalid_saved_audit" for value in cases),
              "unavailable_evidence_count": sum(value.get("evidence_unavailable", False) for value in cases),
              "new_inference_performed": False, "new_continuations_performed": 0,
              "training_examples_exported": 0, "original_artifacts_modified": False,
              "input_sha256": inputs, "source_sha256": source_fingerprints(root), "cases": cases}
    write_json(directory / "audit_review.json", report)
    lines = ["# Saved audit review", "", "Read-only revalidation; original outcomes remain unchanged.", "",
             "| Case | Recorded outcome | Reviewed outcome |", "|---|---|---|"]
    lines.extend(f"| {case['id']} | {case['recorded_outcome']} | {case['review_outcome']} |" for case in cases)
    lines.extend(["", "Skip and unresolved decisions retain their action. Their auxiliary fields are recorded but never used for continuation.",
                  "A structurally valid repair decision still needs hint, boundary, continuation, answer and admission checks.",
                  "This review generates no answers or repairs and changes no original pilot files. Auditor medical quality remains unverified."])
    (directory / "audit_review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {key: report[key] for key in ("status", "review_folder", "validation_policy", "recorded_outcomes", "review_outcomes",
            "nonrepair_decision_count", "schema_valid_repair_decision_count", "invalid_saved_audit_count", "unavailable_evidence_count",
            "new_inference_performed", "new_continuations_performed", "training_examples_exported")}
    summary["case_details"] = [{key: case[key] for key in ("id", "recorded_outcome", "review_outcome", "action",
                              "question", "initial_answer_readout", "reference") if key in case} for case in cases]
    return summary
