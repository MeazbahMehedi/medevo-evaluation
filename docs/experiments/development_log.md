# Development log

Detailed record of each development checkpoint. The versioned summary is in the top-level CHANGELOG.md.

## 8 November 2026 — Expanded feedback selection and output budget

- Added an offline 79 + 40 feedback selection audit, including validation of all earlier twenty requests and responses.
- Added a separately invoked 1024-token run with both scorer versions, paired reproduction records, a fixed missing-box threshold and no automatic retries.
- Preserved the original evaluator, scorer, source modules and completed run files. The expanded live run awaits Windows audit review.

## 8 November 2026 — Qwen3.5 feedback baseline

- Added `evaluate_feedback.py` for twenty fixed feedback questions, ten per dataset, using the paper's shared prompt and explicit CPU, thinking and sampling controls.
- Added raw response preservation, unchanged answer scoring, latency/length statistics, bounded requests, identity checks and stop-on-failure behavior.
- Added eight controlled-transport tests; no live Qwen3.5 medical inference was performed during preparation.
- Recorded the successful Windows tiny NF4/LoRA check as synthetic feasibility evidence only. Full-model training and medical admission remain incomplete.

## 8 November 2026 — One-case auditor-model comparison

- Added explicit 7B replacement and baseline-folder options to `check_auditor.py`, preserving its original default behavior.
- Required a matching completed 3B schema-grounded baseline and changed only the new request's model field.
- Preserved original configurations, both baselines, validation rules and the one-call/no-training bound.
- Added five checks, including a real HTTP request to a scripted local server naming the 7B model. The suite contains 141 tests.
- The 7B model download and real medical inference remain target-PC steps; larger-model quality and resource use are unverified.

## 8 November 2026 — One-case schema-grounding check

- Added the standalone `check_auditor.py` entry point for exactly one call on the first saved training case.
- Preserved the original audit instruction and every other request field, appending the existing JSON schema explicitly.
- Reused archived-input validation, local transport and unchanged audit/hint checks, with separate raw evidence and no retries.
- Added no continuation, admission, training, dependencies or modifications to existing source/configuration files.
- Added seven checks, including one actual HTTP request to a scripted loopback server. The suite contains 136 tests; the live medical reply remains pending.

## 8 November 2026 — Auditor-only instruction comparison

- Added `compare-audits` to compare twelve saved audit responses with a shorter system instruction.
- Reconstructed and checked original image-capable requests, changing only `messages[0].content` for each trial.
- Kept model/endpoint, schema, sampling, budgets and existing validation/hint checks unchanged.
- Added paired raw-response reports, bounded supervised HTTP execution, partial-failure records and no automatic retries.
- Kept coherence pending review and all repair decisions outside continuation, training and admission.
- Added fourteen tests, including a supervised CLI worker calling a scripted local HTTP server twelve times; the suite contains 129 tests.

## 8 November 2026 — Saved-reasoning readout comparison

- Added `compare-readouts` for twelve fixed, saved training cases using the same primary checkpoint and 32-token cap.
- Added a single-turn text-only extraction request, with no reference answers or images supplied to the model.
- Retained original reasoning/readouts and matching rules, and reported lexical changes separately from extraction fidelity and medical validity.
- Added guarded evidence copies, bounded supervised execution, paired JSON/Markdown reports and partial-failure records.
- Kept the original collection and training readout method unchanged, with no new auditor/reasoning calls, training exports or admission.
- Added thirteen tests; the full suite contains 115 tests. Real checkpoint readout inference remains a target-PC checkpoint.

## 8 November 2026 — Bounded training collection

- Added `collect-training` for six fixed training questions per dataset, reusing three saved non-repair cases and processing nine new cases.
- Checked archived data, model, prompt, budget, source and request identity before reuse; preserved original files and raw decisions.
- Recorded question echoes separately from answer mismatches without changing matching rules or generation.
- Indexed potential correct/repaired rollouts and retained schema-valid repair attempts, including later rejections.
- Kept all candidates pending admission, with zero training exports, optimizer steps or adapter creation.
- Added sixteen collection tests, including a supervised CLI run with scripted model responses. The suite contains 102 tests; live collection remains a target-PC checkpoint.

## 8 November 2026 — Non-repair audit handling

- Corrected rejection of skip/unresolved decisions with unused auxiliary fields.
- Preserved raw decisions and kept every non-repair hint outside the continuation input.
- Retained strict schema/boundary checks for genuine repair actions.
- Added `inspect-repair` for a separate read-only review of existing pilot records.
- Added seven review regression tests and replaced the overstrict skip-field expectation; the suite contains 86 checks.
- Recorded the first live Windows pilot as zero repair decisions, with no automatic reinterpretation as successful repairs.

## 8 November 2026 — Training-only repair pilot

- Added a fixed three-question training pilot with original reasoning instructions and explicit short-answer verification readouts.
- Added raw-logit token entropy, heuristic step profiles and an audit-window proposal.
- Added local-only gold-blind audit/review requests and conservative hint checks.
- Added same-checkpoint unfinished-assistant continuation with text/token prefill checks and per-case evidence.
- Kept candidates preliminary, with zero training exports and optimizer steps.
- Added 23 repair tests; the suite now contains 79 checks.
- Kept primary-model loading, existing scoring, prepared data and normal evaluation defaults unchanged.

## 8 November 2026 — Whole-answer scoring diagnostic

- Added `evaluate.py diagnose-answers` for saved feedback prompt comparisons.
- Preserved the boxed scorer and original scores, with a separately named whole-answer exact-match metric.
- Added per-question and JSON/Markdown reports, input hash checks, fixed selections and incomplete-input guards.
- Added 11 tests; the suite now contains 56 checks.
- Added no dependencies, model generation, training or final-test evaluation.

## 8 November 2026 — Answer-format diagnostic

- Added `evaluate.py compare-prompts` for two existing five-question feedback baselines.
- Retained original baseline files, copied their evidence, and created separate concise-prompt trials.
- Added control checks, raw-response verification and paired JSON/Markdown reports.
- Shared the existing supervised execution path between normal evaluation and the diagnostic.
- Normalized source-path separators when checking recorded model and scoring code, including earlier Windows runs.
- Added seven diagnostic checks to the existing 38-test suite.
- Kept the model implementation, scorer, default protocol, dataset preparation, dependencies and final-test restrictions unchanged.

The added prompt is an exploratory adaptation. It is not automatically adopted for final evaluation or MedEvo training.
