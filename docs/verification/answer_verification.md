This records the preceding 56-test scoring checkpoint. The current 79-test repair-pilot update is described in `repair_verification.md`.

# Whole-answer diagnostic verification

Prepared 8 November 2026. This record describes Linux package verification. The source Windows comparison was reported separately; the newly added diagnostic has not yet been executed on that PC.

## Software checks

All 56 tests passed, including the retained 45 checks and 11 new checks. New checks cover fixed normalization, no favourable phrase extraction or synonym/number conversion, independent boxed and whole-answer outcomes, correct references despite prediction metadata, incomplete coverage, read-only report generation, Windows path records, corrupted evidence, changed prompts/IDs/source code and CLI isolation from model execution or data preparation. The complete test output is in `test_results.txt`.

## Saved-output integration check

The command was run against the complete Linux prompt comparison from the previous checkpoint. It reused the existing raw responses and checked references. It did not load the primary model, generate another response, call an auditor or train an adapter. All checked source-run artifact hashes were unchanged after diagnosis. The model and original boxed-scorer files remain byte-identical to the previous package.

| Dataset and prompt | Questions | Boxed strict EM | Whole-answer EM |
|---|---:|---:|---:|
| VQA-RAD, original reasoning instruction | 5 | 0/5 | 0/5 |
| VQA-RAD, concise instruction | 5 | 0/5 | 3/5 |
| SLAKE, original reasoning instruction | 5 | 0/5 | 0/5 |
| SLAKE, concise instruction | 5 | 0/5 | 3/5 |

The concise VQA-RAD matches were the three yes/no responses. Its two open answers did not match. The concise SLAKE matches were MRI, X-ray and the numeral 2. `CT scan` did not match reference `CT`, and the disease response did not match its reference. No synonyms, numeral words or response fragments were substituted to produce a favourable score.

These values describe different scoring rules applied to fixed outputs. They are not MedEvo training gains, final benchmark results or a reliable population accuracy estimate. The ten feedback questions were already inspected during development. Long explanatory answers will generally fail complete-string matching even when they contain the reference concept. Both definitions and their limitations must remain visible.

The diagnostic keeps the old boxed scores and records a separately named whole-answer metric. It does not adopt a new final-evaluation protocol or change the normal evaluator. Medical repair/admission, adaptation, held-out evaluation and the technical report remain later work. Raw records, references, source identities and both score types are retained in `answer_verification.json`.
