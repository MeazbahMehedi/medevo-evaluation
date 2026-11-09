# Results

Evidence files from completed runs, copied unmodified from the local `outputs/` folders that produced them.
Every prediction records the raw model response; scores can be recomputed with `src/scoring.py`,
`feedback_scoring.py` and `medevo_study/core.py`.

| Folder | Contents |
|---|---|
| `protocol/` | Frozen protocol (prompt, 1024-token budget, metrics, 400 test question IDs), amendment 01 (secondary metric) and the pinned primary-model settings |
| `feedback/qwen35_9b_20q_512/` | Qwen3.5-9B on 20 feedback questions at 512 tokens (output-budget evidence) |
| `feedback/qwen35_9b_119q_1024_partial/` | The 119-question budget check at 1024 tokens, stopped after 17 questions; basis of the budget decision |
| `feedback/qwen35_2b_20q/` | Qwen3.5-2B on 20 feedback questions; basis of amendment 01 |
| `baselines/qwen35_9b/` | Qwen3.5-9B (Ollama, Q4_K_M) on the 400 reserved test questions |
| `baselines/qwen35_2b/` | Qwen3.5-2B (Transformers, bfloat16) on the same 400 questions |
| `training_bank_v1/` | Phase 3 collection on 100 training questions: every case decision (`cases.jsonl`), the admitted examples (`training_bank.jsonl`) and the summary |
| `adaptation_v1/` | LoRA training run on the 43-example bank: settings, per-example loss log and adapter configuration (adapter weights not included) |
| `training_bank_v2/` | Re-audit of the 60 v1 failures with the MedEvo-aligned audit prompt; final bank of 43 examples used for training |

Each evaluation folder contains `run.json` (settings, code hashes, model identity), `scored.jsonl`
(one record per question) and `summary.json` / `report.md` (metrics with 95% bootstrap intervals).

## Test baselines (400 questions, 200 per dataset)

Primary metric: normalized exact match of the last `\boxed{}` answer. Secondary metric (amendment 01):
the same, reading a boxed `1`/`0` as yes/no for yes/no questions.

| Model | Metric | VQA-RAD | SLAKE | Overall |
|---|---|---|---|---|
| Qwen3.5-9B | primary | 56.5 [49.5, 63.5] | 57.5 [50.5, 64.0] | 57.0 [52.2, 61.8] |
| Qwen3.5-9B | secondary | 56.5 [49.5, 63.5] | 58.0 [51.0, 64.5] | 57.2 [52.5, 62.0] |
| Qwen3.5-2B | primary | 15.5 [11.0, 20.5] | 27.5 [21.5, 33.5] | 21.5 [17.5, 25.5] |
| Qwen3.5-2B | secondary | 47.5 [40.5, 54.5] | 46.5 [39.5, 53.5] | 47.0 [42.0, 51.7] |

Qwen3.5-2B boxed `1`/`0` instead of yes/no in 190 of 400 answers, which explains most of the gap between
its two scores.

## Phase 3 training bank

| | v1 | v2 (re-audit of v1 failures) |
|---|---|---|
| Audit instruction | did not state that the trajectory failed | aligned with MedEvo Appendix D.3 |
| Repairs proposed | 0 of 60 (all skipped) | 58 of 60 |
| Continuation attempts | — | 280 (268 incorrect, 10 correct and clean) |
| Repairs admitted | 0 | 3 |
| Bank size | 40 | 43 (40 correct rollouts + 3 repairs) |

v2 outcomes: 3 repairs admitted, 54 with all continuations rejected, 1 hint rejected for leaking the answer,
1 skipped and 1 invalid audit. Rejection reasons for every attempt are in `training_bank_v2/cases.jsonl`.

## Phase 4 adaptation

| Setting | Value |
|---|---|
| Base model | Qwen3.5-2B, revision `15852e8c`, float32 on CPU |
| Training data | 43 examples (40 correct rollouts, 3 repairs) |
| LoRA | rank 16, alpha 32, dropout 0.05, language-model linear layers only |
| Trainable parameters | 16,819,200 of 2,230,060,864 (0.75%) |
| Optimizer | AdamW, learning rate 1e-4, linear warmup and decay, gradient accumulation 8 |
| Schedule | 2 epochs, 86 examples, 11 optimizer steps, 19 minutes |
| Mean training loss | epoch 1: 0.421, epoch 2: 0.259 |

The schedule was fixed before training; no checkpoint was selected on feedback or test data.
