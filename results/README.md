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

## Phase 3 training bank (v1)

| Outcome | Questions |
|---|---:|
| Correct rollout admitted | 40 |
| All rollouts failed, auditor returned skip | 60 |
| Repair admitted | 0 |

The v1 audit instruction did not state that the trajectory had failed verification, unlike the MedEvo
audit prompt (Appendix D.3), and the auditor skipped every case. This is recorded as a finding; see the
main README for the follow-up.
