# Results

Primary metric: `boxed-format-unwrapped-em-v1:normalized_correct`. Accuracy % with 95% bootstrap CI.

| System | VQA-RAD | SLAKE | Overall | Boxed % | Truncated |
|---|---|---|---|---:|---:|
| Qwen3.5-9B (Ollama Q4_K_M), unadapted | 56.5 [49.5, 63.5] | 57.5 [50.5, 64.0] | 57.0 [52.2, 61.8] | 94.8 | 28 |
| Qwen3.5-2B, unadapted | 15.5 [11.0, 20.5] | 27.5 [21.5, 33.5] | 21.5 [17.5, 25.5] | 98.8 | 6 |
| Qwen3.5-2B + LoRA (MedEvo-style bank) | 51.0 [44.0, 58.0] | 49.0 [42.0, 56.0] | 50.0 [45.0, 54.8] | 99.5 | 5 |
| Paper: Qwen3.5-9B base | 68.10 | 76.80 | - | - | - |
| Paper: Qwen3.5-9B + MedEvo R3 | 70.40 | 78.23 | - | - | - |

Paper numbers use full benchmarks, full-precision weights and MedUniEval scoring; ours use frozen 200-question subsets, quantized/CPU execution and the scorer above. They are a reference, not a matched comparison.

## Open vs closed questions

| System | Dataset | Open | Closed |
|---|---|---|---|
| Qwen3.5-9B (Ollama Q4_K_M), unadapted | vqa_rad | 28.1 (89) | 79.3 (111) |
| Qwen3.5-9B (Ollama Q4_K_M), unadapted | slake | 41.0 (122) | 83.3 (78) |
| Qwen3.5-2B, unadapted | vqa_rad | 15.7 (89) | 15.3 (111) |
| Qwen3.5-2B, unadapted | slake | 27.9 (122) | 26.9 (78) |
| Qwen3.5-2B + LoRA (MedEvo-style bank) | vqa_rad | 24.7 (89) | 72.1 (111) |
| Qwen3.5-2B + LoRA (MedEvo-style bank) | slake | 29.5 (122) | 79.5 (78) |

## Secondary metric: `boxed-binary-mapped-em-v1:normalized_correct`

Amendment 01: for yes/no questions a boxed 1/0 is read as yes/no. Comparing it with the primary metric separates answer-format effects from answer-content effects.

| System | VQA-RAD | SLAKE | Overall |
|---|---|---|---|
| Qwen3.5-9B (Ollama Q4_K_M), unadapted | 56.5 [49.5, 63.5] | 58.0 [51.0, 64.5] | 57.2 [52.5, 62.0] |
| Qwen3.5-2B, unadapted | 47.5 [40.5, 54.5] | 46.5 [39.5, 53.5] | 47.0 [42.0, 51.7] |
| Qwen3.5-2B + LoRA (MedEvo-style bank) | 51.0 [44.0, 58.0] | 49.0 [42.0, 56.0] | 50.0 [45.0, 54.8] |

## Paired comparisons, primary metric (same questions)

| Comparison | Scope | n | Difference (points) | 95% CI | Only-other correct | Only-base correct | McNemar p |
|---|---|---:|---:|---|---:|---:|---:|
| adapted_vs_primary | overall | 400 | +28.5 | [23.5, 33.5] | 127 | 13 | 0.000 |
| adapted_vs_primary | vqa_rad | 200 | +35.5 | [28.5, 42.5] | 74 | 3 | 0.000 |
| adapted_vs_primary | slake | 200 | +21.5 | [14.5, 28.5] | 53 | 10 | 0.000 |
| primary_vs_baseline | overall | 400 | -35.5 | [-40.8, -30.2] | 9 | 151 | 0.000 |
| primary_vs_baseline | vqa_rad | 200 | -41.0 | [-48.0, -34.0] | 3 | 85 | 0.000 |
| primary_vs_baseline | slake | 200 | -30.0 | [-37.0, -23.0] | 6 | 66 | 0.000 |

## Paired comparisons, secondary metric (same questions)

| Comparison | Scope | n | Difference (points) | 95% CI | Only-other correct | Only-base correct | McNemar p |
|---|---|---:|---:|---|---:|---:|---:|
| adapted_vs_primary | overall | 400 | +3.0 | [-1.0, 6.8] | 38 | 26 | 0.169 |
| adapted_vs_primary | vqa_rad | 200 | +3.5 | [-2.0, 9.0] | 20 | 13 | 0.296 |
| adapted_vs_primary | slake | 200 | +2.5 | [-3.0, 8.0] | 18 | 13 | 0.473 |
| primary_vs_baseline | overall | 400 | -10.2 | [-14.8, -5.8] | 22 | 63 | 0.000 |
| primary_vs_baseline | vqa_rad | 200 | -9.0 | [-15.5, -2.5] | 13 | 31 | 0.010 |
| primary_vs_baseline | slake | 200 | -11.5 | [-17.5, -5.5] | 9 | 32 | 0.000 |

## Phase 3 training bank

Questions processed: 100 of 100
Admitted: 40 (40 correct rollouts, 0 repairs). Audits run: 60.

| Outcome | Questions |
|---|---:|
| audit_skip | 60 |
| correct_rollout_admitted | 40 |

## Phase 3 re-audit (v2, MedEvo audit prompt)

Re-audited failures: 60 of 60. Admitted repairs: 3. Final bank: 43 examples.

| Outcome | Questions |
|---|---:|
| audit_invalid | 1 |
| audit_skip | 1 |
| hint_rejected_leak | 1 |
| repair_admitted | 3 |
| repair_attempts_rejected | 54 |
