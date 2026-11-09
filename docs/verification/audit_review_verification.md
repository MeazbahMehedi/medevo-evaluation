# Saved-audit validation verification

Prepared 8 November 2026. The current update contains 86 passing tests: seven saved-review checks plus the earlier suite with its overstrict skip-field expectation corrected. Test output is retained in `test_results.txt`.

## Regression evidence

The three auditor objects from the Windows run are preserved as data in `tests/fixtures/nonrepair_audits.json`. They select one unresolved and two skip actions with step 0 and nonempty hints. The corrected validator retains each raw decision and ignores non-repair auxiliary values for continuation. Genuine repair actions still require a valid nonzero step, supported prefix, medical family, hint and evidence; downstream hint and admission filters remain intact.

Tests confirm that the supplied non-repair decisions invoke no continuation, their raw hints remain unchanged in the record, source pilot files are not overwritten, malformed responses are not converted into valid actions, and structurally valid repairs remain subject to further checks. Active locks, changed case identities and inconsistent saved reasoning spans are rejected.

## Read-only CLI check

The actual `inspect-repair` CLI was run with the reported auditor objects and explicitly synthetic surrounding pilot metadata/reasoning. It produced one `auditor_unresolved` and two `auditor_skip` reviewed outcomes, zero schema-valid repair decisions, zero new continuations and zero training exports. All source fixture files retained their hashes. The command did not run the primary model or contact Ollama. Model loading, existing scoring and the continuation-generator source remain byte-identical to the preceding package.

This is a reproducible software regression check, not a local reinspection of the complete Windows archive. Run the provided read-only command against that archive to check it. Its original outcome labels remain preserved beside the separate reviewed interpretation. The new report includes stored initial answer readouts and references where present; those fields are for local analysis and are never sent to a model by the reviewer.

## Interpretation

The fix removes a validation error in the implementation. It does not improve the auditor's medical judgement, create a repair decision, establish a supported prefix, or qualify training data. The first reported pilot still has no repair decisions and has not exercised live continuation. Some unused strings repeat instructions or contain asserted findings; retaining a skip does not establish that those statements are clinically correct.

No new medical generation, training, test-set evaluation or dependency requirement was introduced by this update. Full admission and medical adaptation remain pending. Exact fixture outputs, hashes and policy/source identities are retained in `audit_review_verification.json`.
