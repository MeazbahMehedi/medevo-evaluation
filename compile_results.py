"""Compile all completed runs into result tables and a technical-report draft with the numbers filled in.

Example:  python compile_results.py
Writes outputs/results_v1/results.json, results.md and technical_report_draft.md
"""

import json
import sys

from medevo_study.collection import ADMISSION
from medevo_study.core import (BINARY_SCORER, OUTPUTS, PAPER_REFERENCE, PRIMARY_FIELD, PRIMARY_SCORER, PROTOCOL_DIR, paired_bootstrap,
                               read_json, read_jsonl, safe_error, utc_now, write_json)

SYSTEMS = {"baseline": ("Qwen3.5-9B (Ollama Q4_K_M), unadapted", "test_eval_qwen35_9b_v1"),
           "primary": ("Qwen3.5-2B, unadapted", "test_eval_qwen35_2b_base_v1"),
           "adapted": ("Qwen3.5-2B + LoRA (MedEvo-style bank)", "test_eval_qwen35_2b_adapted_v1")}
KEY = f"{PRIMARY_SCORER}:{PRIMARY_FIELD}"
SECONDARY = f"{BINARY_SCORER}:normalized_correct"
OUT = OUTPUTS / "results_v1"


def load_system(folder):
    path = OUTPUTS / folder
    if not (path / "summary.json").exists():
        return None, None
    summary = read_json(path / "summary.json")
    scored = read_jsonl(path / "scored.jsonl")
    done = [r for r in scored if r.get("status") == "generated"]
    correctness = {r["id"]: int(r["scores"][PRIMARY_SCORER][PRIMARY_FIELD]) for r in done}
    if all(BINARY_SCORER in r["scores"] for r in done):
        correctness_binary = {r["id"]: int(r["scores"][BINARY_SCORER]["normalized_correct"]) for r in done}
    else:
        correctness_binary = None
    return summary, (scored, correctness, correctness_binary)


def pct(value, ci=None):
    if value is None:
        return "-"
    return f"{value:.1f}" + (f" [{ci[0]:.1f}, {ci[1]:.1f}]" if ci else "")


def flips(base_scored, other_scored, limit=6):
    base = {r["id"]: r for r in base_scored if r.get("status") == "generated"}
    gained, lost = [], []
    for row in other_scored:
        if row.get("status") != "generated" or row["id"] not in base:
            continue
        before, after = base[row["id"]]["scores"][PRIMARY_SCORER], row["scores"][PRIMARY_SCORER]
        entry = {"id": row["id"], "question": row["question"], "reference": row["reference"],
                 "before": before["parsed_answer"], "after": after["parsed_answer"]}
        if after[PRIMARY_FIELD] and not before[PRIMARY_FIELD]:
            gained.append(entry)
        elif before[PRIMARY_FIELD] and not after[PRIMARY_FIELD]:
            lost.append(entry)
    return gained[:limit], lost[:limit]


def compile_all():
    protocol = read_json(PROTOCOL_DIR / "protocol.json")
    loaded = {name: load_system(folder) for name, (_, folder) in SYSTEMS.items()}
    results = {"compiled_utc": utc_now(), "primary_metric": KEY, "secondary_metric": SECONDARY,
               "protocol_budget": protocol["output_token_budget"], "systems": {}, "paired": {},
               "paired_secondary": {}, "paper_reference": PAPER_REFERENCE}
    for name, (summary, _) in loaded.items():
        if summary:
            results["systems"][name] = {"label": SYSTEMS[name][0], "status": summary["status"],
                                        "metrics": summary["metrics"][KEY],
                                        "secondary_metrics": summary["metrics"].get(SECONDARY)}
    pairs = [("primary", "adapted"), ("baseline", "primary")]
    for base, other in pairs:
        if not (loaded[base][1] and loaded[other][1]):
            continue
        for slot, target in ((1, "paired"), (2, "paired_secondary")):
            left, right = loaded[base][1][slot], loaded[other][1][slot]
            if left is None or right is None:
                continue
            entry = {"overall": paired_bootstrap(left, right)}
            for dataset in ("vqa_rad", "slake"):
                ids = {r["id"] for r in loaded[other][1][0] if r["dataset"] == dataset}
                entry[dataset] = paired_bootstrap({k: v for k, v in left.items() if k in ids},
                                                  {k: v for k, v in right.items() if k in ids})
            results[target][f"{other}_vs_{base}"] = entry
    bank_v1 = OUTPUTS / "training_bank_v1" / "summary.json"
    bank_v2 = OUTPUTS / "training_bank_v2" / "summary.json"
    results["training_bank"] = read_json(bank_v1) if bank_v1.exists() else None
    results["training_bank_v2"] = read_json(bank_v2) if bank_v2.exists() else None
    adapter_path = OUTPUTS / "adapter_v1" / "run.json"
    if adapter_path.exists():
        adapter = read_json(adapter_path)
        losses = [r["loss"] for r in read_jsonl(OUTPUTS / "adapter_v1" / "train_log.jsonl")]
        window = max(1, len(losses) // 10)
        results["adapter"] = {k: adapter.get(k) for k in ("bank_size", "bank_kinds", "epochs", "optimizer_steps",
                                                          "trainable_parameters", "elapsed_seconds", "lora", "optimizer")}
        results["adapter"]["loss_first_10pct"] = sum(losses[:window]) / window if losses else None
        results["adapter"]["loss_last_10pct"] = sum(losses[-window:]) / window if losses else None
    feedback_path = OUTPUTS / "qwen35_feedback_119_1024_v1" / "summary.json"
    results["feedback_budget_decision"] = read_json(feedback_path).get("decision") if feedback_path.exists() else None
    if loaded["primary"][1] and loaded["adapted"][1]:
        gained, lost = flips(loaded["primary"][1][0], loaded["adapted"][1][0])
        results["examples"] = {"gained": gained, "lost": lost}
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / "results.json", results)
    (OUT / "results.md").write_text(results_markdown(results), encoding="utf-8")
    (OUT / "technical_report_draft.md").write_text(report_draft(results, protocol), encoding="utf-8")
    return {"status": "compiled", "folder": str(OUT), "systems": sorted(results["systems"]),
            "paired": sorted(results["paired"])}


def results_markdown(results):
    lines = ["# Results", "", f"Primary metric: `{results['primary_metric']}`. Accuracy % with 95% bootstrap CI.", "",
             "| System | VQA-RAD | SLAKE | Overall | Boxed % | Truncated |", "|---|---|---|---|---:|---:|"]
    for name, data in results["systems"].items():
        m = data["metrics"]
        lines.append(f"| {data['label']} | {pct(m['datasets']['vqa_rad']['all']['accuracy_pct'], m['datasets']['vqa_rad']['all']['accuracy_ci95'])} | "
                     f"{pct(m['datasets']['slake']['all']['accuracy_pct'], m['datasets']['slake']['all']['accuracy_ci95'])} | "
                     f"{pct(m['overall']['accuracy_pct'], m['overall']['accuracy_ci95'])} | {pct(m['overall']['boxed_rate_pct'])} | "
                     f"{m['overall']['truncated']} |")
    ref = results["paper_reference"]
    lines += [f"| Paper: Qwen3.5-9B base | {ref['vqa_rad']['base']:.2f} | {ref['slake']['base']:.2f} | - | - | - |",
              f"| Paper: Qwen3.5-9B + MedEvo R3 | {ref['vqa_rad']['medevo']:.2f} | {ref['slake']['medevo']:.2f} | - | - | - |",
              "", "Paper numbers use full benchmarks, full-precision weights and MedUniEval scoring; ours use frozen "
              "200-question subsets, quantized/CPU execution and the scorer above. They are a reference, not a matched comparison.",
              "", "## Open vs closed questions", "", "| System | Dataset | Open | Closed |", "|---|---|---|---|"]
    for name, data in results["systems"].items():
        for dataset in ("vqa_rad", "slake"):
            block = data["metrics"]["datasets"][dataset]
            lines.append(f"| {data['label']} | {dataset} | {pct(block['open']['accuracy_pct'])} ({block['open']['generated']}) | "
                         f"{pct(block['closed']['accuracy_pct'])} ({block['closed']['generated']}) |")
    lines += ["", f"## Secondary metric: `{results['secondary_metric']}`", "",
              "Amendment 01: for yes/no questions a boxed 1/0 is read as yes/no. Comparing it with the primary "
              "metric separates answer-format effects from answer-content effects.", "",
              "| System | VQA-RAD | SLAKE | Overall |", "|---|---|---|---|"]
    for name, data in results["systems"].items():
        m = data.get("secondary_metrics")
        if m:
            lines.append(f"| {data['label']} | {pct(m['datasets']['vqa_rad']['all']['accuracy_pct'], m['datasets']['vqa_rad']['all']['accuracy_ci95'])} | "
                         f"{pct(m['datasets']['slake']['all']['accuracy_pct'], m['datasets']['slake']['all']['accuracy_ci95'])} | "
                         f"{pct(m['overall']['accuracy_pct'], m['overall']['accuracy_ci95'])} |")
    for title, key in (("Paired comparisons, primary metric (same questions)", "paired"),
                       ("Paired comparisons, secondary metric (same questions)", "paired_secondary")):
        if not results.get(key):
            continue
        lines += ["", f"## {title}", "",
                  "| Comparison | Scope | n | Difference (points) | 95% CI | Only-other correct | Only-base correct | McNemar p |",
                  "|---|---|---:|---:|---|---:|---:|---:|"]
        for label, entry in results[key].items():
            for scope, p in entry.items():
                if p:
                    ci = f"[{p['ci95_points'][0]:.1f}, {p['ci95_points'][1]:.1f}]" if p["ci95_points"] else "-"
                    lines.append(f"| {label} | {scope} | {p['paired_count']} | {p['difference_points']:+.1f} | {ci} | "
                                 f"{p['other_only_correct']} | {p['base_only_correct']} | {p['mcnemar_exact_p']:.3f} |")
    bank = results.get("training_bank")
    if bank:
        lines += ["", "## Phase 3 training bank", "", f"Questions processed: {bank['questions_processed']} of {bank['selected_questions']}",
                  f"Admitted: {bank['admitted_total']} ({bank['admitted_correct_rollouts']} correct rollouts, "
                  f"{bank['admitted_repairs']} repairs). Audits run: {bank['audits_run']}.", "",
                  "| Outcome | Questions |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in bank["outcomes"].items()]
    bank2 = results.get("training_bank_v2")
    if bank2:
        lines += ["", "## Phase 3 re-audit (v2, MedEvo audit prompt)", "",
                  f"Re-audited failures: {bank2['reaudited']} of {bank2['targets']}. Admitted repairs: "
                  f"{bank2['admitted_repairs']}. Final bank: {bank2['bank_size']} examples.", "",
                  "| Outcome | Questions |", "|---|---:|"] + [f"| {k} | {v} |" for k, v in bank2["outcomes"].items()]
    return "\n".join(lines) + "\n"


def report_draft(results, protocol):
    s = results["systems"]
    get = lambda name, dataset: pct(s[name]["metrics"]["datasets"][dataset]["all"]["accuracy_pct"]) if name in s else "[pending]"
    bank = results.get("training_bank") or {}
    bank2 = results.get("training_bank_v2") or {}
    adapter = results.get("adapter") or {}
    paired = results["paired"].get("adapted_vs_primary", {}).get("overall")
    change = (f"{paired['difference_points']:+.1f} points (95% CI {paired['ci95_points'][0]:.1f} to "
              f"{paired['ci95_points'][1]:.1f}; McNemar p = {paired['mcnemar_exact_p']:.3f})") if paired else "[pending]"
    paired_b = results.get("paired_secondary", {}).get("adapted_vs_primary", {}).get("overall")
    change_secondary = (f"{paired_b['difference_points']:+.1f} points (95% CI {paired_b['ci95_points'][0]:.1f} to "
                        f"{paired_b['ci95_points'][1]:.1f}; McNemar p = {paired_b['mcnemar_exact_p']:.3f})") if paired_b else "[pending]"
    decision = results.get("feedback_budget_decision") or {}
    return f"""# Evaluating MedEvo on VQA-RAD and SLAKE: a reduced-scale CPU study

Author: [your name] · Date: [date]

## Abstract

[Write 4-6 sentences after reviewing the results: aim, setup, main numbers, main limitation.]

## 1. Objective

The task was to evaluate the MedEvo framework (Wang et al., arXiv:2610.17690) on two to three benchmark datasets,
recording the experimental process, metrics and analysis. The authors' project page states that the code, training
data and weights are not released, so this study implements a reduced version of the method independently and
evaluates it on VQA-RAD and SLAKE (English).

## 2. Method and implementation

MedEvo trains a model on its own correct rollouts plus repaired failures: an auditor that does not see the reference
locates the first unsupported reasoning step, the model keeps the supported prefix and regenerates the rest after a
short hint, and repairs are admitted only after correctness, integrity and learnability checks.

What this study implements:

| Component | Paper | This study |
|---|---|---|
| Base / primary model | Qwen3.5-9B/35B/397B | Qwen3.5-2B (trainable on CPU); Qwen3.5-9B evaluated unadapted as the paper-base reference |
| Auditor | Strong external model | Qwen3.5-9B (Ollama, quantized), reference-free, JSON output |
| Rollouts per question | 8 | {ADMISSION['rollouts_per_question']} |
| Training questions | 81,293 | {bank.get('selected_questions', '[pending]')} |
| Rounds | 3 (feedback-guided) | 1 (no feedback-guided reallocation) |
| Admission | correctness, boundary, integrity, learnability | correctness, boundary range, integrity, hint leak guard, hint-free NLL and NLL ratio |
| Training | SFT | LoRA SFT, rank {adapter.get('lora', {}).get('rank', '[pending]')}, {adapter.get('epochs', '[pending]')} epochs, fixed schedule |

## 3. Data

VQA-RAD (flaviagiammarino/vqa-rad) and SLAKE English (BoKelvin/SLAKE) at pinned revisions. Training, feedback and
test roles are separated, and training excludes images that appear in feedback or test. The frozen test subsets
contain {protocol['test_selection']['per_dataset']} questions per dataset, sampled with proportional open/closed
allocation (seed {protocol['test_selection']['seed']}): {json.dumps(protocol['test_selection']['counts'])}.

## 4. Experimental protocol

- Prompt (paper, Appendix D.1): "{protocol['prompt']}"
- Output budget: {protocol['output_token_budget']} tokens, chosen on feedback data by a rule fixed in advance
  (missing boxes = {decision.get('missing_boxes', '[pending]')} of 119; outcome {decision.get('outcome', '[pending]')}).
- Primary metric: {protocol['primary_metric']['definition']}.
- Uncertainty: {protocol['uncertainty']}.
- Secondary metric (amendment 01, decided on 20 feedback questions before any test generation): for yes/no
  questions a boxed 1/0 is read as yes/no, because the 2B model often boxed 1/0 while its text said yes/no.
- Test results were not used for any choice of prompt, budget, scorer, collection, admission or training.

## 5. Results

| System | VQA-RAD | SLAKE |
|---|---|---|
| Qwen3.5-9B, unadapted (ours) | {get('baseline', 'vqa_rad')} | {get('baseline', 'slake')} |
| Qwen3.5-2B, unadapted | {get('primary', 'vqa_rad')} | {get('primary', 'slake')} |
| Qwen3.5-2B + MedEvo-style LoRA | {get('adapted', 'vqa_rad')} | {get('adapted', 'slake')} |
| Paper: Qwen3.5-9B base | 68.10 | 76.80 |
| Paper: Qwen3.5-9B + MedEvo | 70.40 | 78.23 |

Effect of adaptation (same questions): {change}.
Under the secondary metric (boxed 1/0 read as yes/no): {change_secondary}.

Training bank: {bank2.get('bank_size', bank.get('admitted_total', '[pending]'))} admitted examples
({bank.get('admitted_correct_rollouts', '-')} correct rollouts from {bank.get('questions_processed', '-')} training
questions; {bank2.get('admitted_repairs', bank.get('admitted_repairs', '-'))} repairs after the v2 re-audit of
{bank2.get('reaudited', '-')} failures). Full tables: results.md.

## 6. Analysis

[Discuss: gap between our 9B baseline and the paper; whether adaptation helped and on which question types;
what the repair yield says about the auditor and the 2B model; representative gained/lost examples
(results.json → examples); format and truncation behaviour.]

## 7. Earlier feasibility work and negative findings

[Summarize: SmolVLM2-500M format failures; 3B/7B local auditors producing invalid repair boundaries; the readout
artifact that labelled a correct answer as a failure; how each finding changed the design.]

## 8. Limitations

- Independent reduced implementation; not the authors' code. One round, fewer rollouts, smaller training set.
- Smaller trainable model (2B) than the paper; the 9B reference is quantized (Q4_K_M) and run on CPU.
- Auditor is a quantized 9B model, much weaker than the paper's auditor; same model family as the primary model.
- 200-question subsets: confidence intervals are wide; the paper's 1-2 point gains are below what they can resolve.
- Lexical scoring; references and model image readings were not clinically adjudicated.
- Exact-image separation only; no patient-level or near-duplicate guarantee.

## 9. Reproducibility

Hardware: AMD Ryzen 7 9800X3D, 32 GB RAM, CPU only. Commands, in order: see docs/running_the_study.md.
All raw predictions, audits, admission decisions, training logs and hashes are under outputs/.

## References

Wang et al. MedEvo: Medical Reasoning Evolution through Knowledge Navigation. arXiv:2610.17690, 2026.
"""


def main():
    try:
        print(json.dumps(compile_all(), indent=2))
        return 0
    except Exception as error:
        print(json.dumps({"status": "blocked", "error": safe_error(error)}, indent=2))
        return 2


if __name__ == "__main__":
    sys.exit(main())
