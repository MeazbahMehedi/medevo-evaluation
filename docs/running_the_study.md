# Running the study

This guide runs the frozen-protocol study end to end: test baselines, Phase 3 training-bank construction,
LoRA adaptation and result compilation. Commands are for Windows PowerShell from the repository root;
see [windows_setup.md](windows_setup.md) for the environment. Use the project interpreter explicitly:

```powershell
$py = '.\.venv312\Scripts\python.exe'
```

Before each long run: disable sleep, keep the PC on mains power, close heavy programs, and run one job at a time.
Every long command is **resumable**: if it stops, run the same command again and it continues where it left off.

## Systems in this study

| Name in commands | Model | Role |
|---|---|---|
| `baseline` | Qwen3.5-9B, Ollama (Q4_K_M) | Unadapted reference for the paper's base row; also the auditor in Phase 3 |
| `primary` | Qwen3.5-2B, Transformers | The model that is adapted with the MedEvo-style training bank |
| `adapted` | Qwen3.5-2B + LoRA adapter | The primary model after Phase 4 |

## Step 0: check the new code (1 minute)

```powershell
& $py -m unittest discover -s tests -p "test_medevo_study.py" -v
```

## Step 1: freeze the protocol (after the 119-question run finishes)

```powershell
& $py .\freeze_protocol.py --per-dataset 200
```

This reads the decision from `outputs\qwen35_feedback_119_1024_v1\summary.json` (512 or 1024 tokens),
picks 200 test questions per dataset (stratified open/closed, seed 7) and writes
`outputs\frozen_protocol_v1\`. It refuses to run twice. Do not change the prompt, budget or scorer after this.

## Step 2: baseline on the test subset (overnight, about 4-6 hours)

```powershell
& $py .\run_test_evaluation.py --system baseline
```

Output: `outputs\test_eval_qwen35_9b_v1\` (`report.md`, `summary.json`, `scored.jsonl`).

## Step 3: prepare the primary model (about 15-30 minutes, downloads about 4.5 GB)

```powershell
& $py -m pip install -U "transformers" "huggingface-hub"   # only if Step 3 reports an unknown model type
& $py .\prepare_primary_model.py --dtype bfloat16 --threads 8
```

Writes `outputs\frozen_protocol_v1\primary_model.json` with the pinned revision and a one-question readiness check
(speed, memory, whether it produced a boxed answer). Then a short feedback check:

```powershell
& $py .\run_test_evaluation.py --system primary --feedback 20
```

If boxed answers are rare or most answers hit the token limit, stop and review before Phase 3.

## Step 4: primary model on the test subset (about 2-3 hours)

```powershell
& $py .\run_test_evaluation.py --system primary
```

## Step 5: Phase 3, build the training bank (about 3-6 hours; Ollama must be running)

```powershell
& $py .\build_training_bank.py --questions 100 --max-hours 8
```

Training-role questions only. Per question: 4 sampled rollouts; the first clean correct rollout is admitted;
if all fail, Qwen3.5-9B audits one failure without the reference, the primary model continues from the supported
prefix with the hint (up to 3 attempts), and the repair is admitted only if it is correct, complete,
non-repetitive, does not leak or copy the hint, and passes the learnability checks
(hint-free NLL ≤ 1.5 nats/token, hint-free/hinted NLL ratio ≤ 2.0). The hint is removed from the training target.

Outputs in `outputs\training_bank_v1\`: `cases.jsonl` (every decision, including rejections),
`training_bank.jsonl` (admitted examples) and `summary.json`.

**Stop rule:** Phase 3 ends on the planned day whatever the yield. If no repair is admitted, continue with the
correct rollouts only and report the result as a self-training variant, with the repair attempts as a finding.

## Step 5b: Phase 3 re-audit with the MedEvo audit prompt (about 1-1.5 hours)

```powershell
& $py .\reaudit_failures.py
```

Re-audits the failed questions from bank v1 using saved rollouts (no new rollouts). The auditor is told the
trajectory failed verification, without the reference, and may skip only formatting-only failures
(MedEvo Appendix D.3). Up to five continuations per repair; admission uses the paper's learnability gates
(D_repair in [0.35, 0.85], Delta_repair in [0.02, 0.30]). Repairs are trained with the recovery cue kept.
Output: `outputs\training_bank_v2\` (v1 correct rollouts + admitted repairs). Bank v1 is unchanged.

## Step 6: Phase 4, train the adapter (about 1-3 hours on CPU)

```powershell
& $py .\train_adapter.py --epochs 2   # uses training_bank_v2 if present
```

Fixed schedule declared in advance (LoRA rank 16, alpha 32, learning rate 1e-4, 2 epochs, gradient
accumulation 8). Only language-model layers are adapted. Output: `outputs\adapter_v1\`.

## Step 7: adapted model on the same test subset (about 2-3 hours)

```powershell
& $py .\run_test_evaluation.py --system adapted --adapter-dir .\outputs\adapter_v1\adapter
```

## Step 8: compile results and the report draft

```powershell
& $py .\compile_results.py
```

Writes `outputs\results_v1\results.md` (all tables, paired comparison, McNemar test) and
`technical_report_draft.md` with the numbers filled in. Sections marked `[...]` need your own analysis.

## If compute time is limited

Reduce scope in this order: test subsets to 100 questions per dataset; repairs (keep correct rollouts);
adaptation (report the baselines and the Phase 3 findings).
