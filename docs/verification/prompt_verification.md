# Answer-format diagnostic verification

This records the earlier 45-test prompt-diagnostic checkpoint. The later scoring-only update uses 56 tests and is documented in `answer_verification.md`.

Prepared 8 November 2026 in the Linux execution environment. These observations are distinct from the Windows experiment records. The full development evidence, raw responses and references are retained in `prompt_verification.json`.

## Software checks

All 45 tests passed: the retained 38 checks plus seven diagnostic tests. The new tests cover unchanged baseline evidence, exact pairing and controls, blocked non-feedback or changed configurations, incomplete generation, corrupted metric claims, Windows source-path compatibility and invalidation when an original baseline changes. Controlled fixture tests use mocked model responses; they are not medical evaluations.

The public model implementation and scorer are byte-identical to the preceding package. The default protocol and dataset bundle were not changed. The new command uses the existing generation and scoring functions and the same supervisor. No dependency was added.

## Real-model execution

The complete command was exercised using the actual pinned pretrained model and real feedback images. Two fresh five-question original-prompt baselines were generated for this Linux check, followed by their paired concise-prompt trials. Every trial selected the same IDs in the same order as its baseline. Original run files were verified unchanged after execution. Both dataset trials completed without infrastructure errors.

| Feedback subset | Original valid boxes | Concise valid boxes | Original strict EM | Concise strict EM |
|---|---:|---:|---:|---:|
| VQA-RAD, 5 questions | 1/5 | 0/5 | 0/5 | 0/5 |
| SLAKE, 5 questions | 0/5 | 0/5 | 0/5 | 0/5 |

The concise prompt produced shorter answers and no token-limit hits in these trials, but did not improve boxed-answer compliance. Some unboxed answers contain reference-aligned content; the declared strict scorer still records a formatting failure. No fallback extraction, retrospective relabelling or alternative score was used to turn these into successes.

This negative result is retained. It does not justify medical accuracy claims, adoption of the concise prompt for final evaluation, or starting MedEvo training. It indicates that a prompt-only instruction change may be insufficient for reliable boxed output from this small model. The next protocol/model decision requires a separate review. The same ten already examined feedback questions are exploratory development evidence, not an independent evaluation sample.

All original and trial metrics, generation durations, worker durations and resource flags are retained in the JSON record. Linux RSS readings remain implausible and are explicitly marked unverified. Generation timings came from separate attempts and are not controlled hardware comparisons. Windows runtime and diagnostic results require their own run.

## Research boundary

No final test question was generated, no auditor was called, no training example was exported, and no adapter was trained by this command. The concise instruction suppresses explanations and therefore does not establish the reasoning trajectories required for MedEvo repair. The original instruction remains the default. The implementation and a negative diagnostic outcome are both documented; neither is presented as a successful reproduction of the paper.
