# Repair-pilot verification

This records the preceding 79-test checkpoint. The later live Windows outputs and validation correction are described in `audit_review_verification.md`.

Prepared 8 November 2026 in the Linux execution environment. The live Qwen auditor installed on the Windows PC was not evaluated here. Code preparation, model execution, scripted transport fixtures and medical-quality claims are separate evidence categories.

## Software checks

All 79 tests passed: 56 retained checks plus 23 repair-pilot checks. Added checks cover sentence-span preservation, entropy/surge arithmetic and constant profiles, integer step boundaries, skip/unresolved behaviour, hint filters, training-only selection, exact readout verification, gold-blind payload fields, prefix integrity, full-regeneration labelling, candidate rejection paths, preliminary versus training admission, infrastructure records, local HTTP credentials/proxy handling, worker locking and CLI separation from evaluation/training. The test output is `test_results.txt`.

The positive candidate-admission path uses controlled mocked generation and review outputs. Its result is only a preliminary candidate. It does not establish model-based medical correctness or a successful live repair.

## Actual primary-model check

The real pinned pretrained SmolVLM2 model was loaded on CPU and used on one of the selected training images. A technical fixture ran an initial reasoning generation, a separate short answer readout and a controlled assistant-prefill continuation. The fixture used a generic inspection prefix and hint; neither was a clinically audited prefix or a dataset reference answer.

- Initial-generation token entropy was measured from actual raw logits.
- The unfinished assistant message rendered and tokenized with the supplied prefill intact.
- Returned input token IDs matched the input prefix.
- The controlled source prefix was retained verbatim in the reconstructed response.

Budgets were reduced for this development check: 48 reasoning tokens, 16 answer-readout tokens and 24 continuation tokens. Exact settings and observed outputs are retained in `repair_verification.json`. This is a model/processor compatibility check, not a demonstration of a medically valid repair.

## Three-case supervised driver

The complete CLI, supervisor, worker, deterministic training selection, original reasoning instruction, entropy calculation, readout verification, local HTTP serialization and result writer were exercised on all three selected training questions. The primary model was real. The local HTTP auditor was a scripted fixture that always returned `unresolved` and explicitly made no medical judgement.

Observed outcomes were one initial readout match and two scripted unresolved audits. The worker completed with exit code zero, no resource limit reached, zero optimizer steps, zero exported training examples and zero adapters. A validated audit count in this fixture means schema validation only. The driver did not take a medical continuation path because the fixture declined repair; the actual model's prefill capability was checked separately above.

The normal 128/32/128 budgets remain the defaults for the Windows pilot. The driver fixture used 48/16/24 token limits to bound the development check. Its generation outputs are not medical benchmark results or forecasts for Windows timing.

Linux process-memory samples were unavailable, so the driver correctly marked the peak unverified. Its resource record measures the Python process tree and excludes any external Ollama service.

## Remaining checks

The next checkpoint is the expanded test suite on Windows, followed by the actual local Qwen audit/continuation pilot. Schema and lexical hint checks cannot establish the earliest medical error, absence of semantic hint leakage or correctness of the retained reasoning. The same-auditor second pass is a model review, not independent clinical validation.

Preliminary candidates remain evidence for the next implementation decision. Full suffix-learnability admission, calibrated thresholds, detailed medical error subtypes, feedback navigation, SFT target export, adaptation and final held-out evaluation remain open. This study does not include external clinical sign-off; it records these limits rather than claiming validation that has not occurred.
