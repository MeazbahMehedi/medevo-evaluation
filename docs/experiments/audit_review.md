# Reviewing saved non-repair audits

The first live Windows pilot recorded three `rejected_invalid_audit` outcomes. Its saved responses actually selected one `unresolved` and two `skip` actions. Each included nonempty auxiliary hint text. The previous validator incorrectly rejected those decisions because it required zero/empty auxiliary fields for every non-repair action.

In the paper's Skip branch, error, boundary and hint values are unused. The corrected policy keeps basic schema/type checks and all repair checks, preserves raw fields, and discards auxiliary values from generation when the action is skip or unresolved. It never turns a non-repair action into a repair. This pilot's explicit unresolved extension uses the same no-continuation rule.

## Apply the correction

No new dependency or model download is added.

The current suite contains 86 checks, including seven new saved-audit review checks. The earlier 79-test Windows result remains a valid record of that software checkpoint.

## Review the existing pilot

From the repository root:

```powershell
& '.\.venv312\Scripts\python.exe' '.\evaluate.py' inspect-repair --run-dir '.\outputs\20261108T044904Z_repair_pilot_01186c'
```

This reads saved files only. Ollama does not need to regenerate responses. It creates a separate `outputs/<timestamp>_audit_review_<id>/` folder with `audit_review.json` and `audit_review.md`. Original case files, responses and summary remain unchanged. The console also includes stored question, initial answer-readout and reference fields when present, so those verification failures can be examined without new inference. Those reference fields are report data, not model inputs.

Expected decision interpretation for the supplied live responses:

| Case | Originally recorded | Auditor action preserved | Reviewed outcome |
|---|---|---|---|
| `vqa_rad:train:000201` | `rejected_invalid_audit` | `unresolved` | `auditor_unresolved` |
| `vqa_rad:train:001467` | `rejected_invalid_audit` | `skip` | `auditor_skip` |
| `slake:train:4022` | `rejected_invalid_audit` | `skip` | `auditor_skip` |

There are still **zero repair decisions and zero continuations** in this source pilot. The review does not claim a successful medical repair, qualify training targets or improve the model. `status: complete` refers to creation of the review report.

## Validation policy

Policy ID: `nonrepair-auxiliary-unused-v2`.

- Every response still requires the declared fields, recognized action/family, integer schema-range step, boolean prefix field and bounded text.
- A repair still requires a valid nonzero reasoning-step index, a supported prefix, a medical error family, nonempty hint and boundary evidence. Subsequent hint, continuation, answer and admission checks remain mandatory.
- For skip or unresolved, first-error step, error-family label, prefix flag and hint are unused for continuation. Their raw values remain available for inspection. In particular, a nonempty hint is never forwarded to the primary model in these branches.
- New pilot runs record the policy and ignored-field decision. Existing runs retain their original outcomes; `inspect-repair` provides a separate retrospective interpretation.

The original request still asks the auditor to use a zero step and empty hint when declining repair. The validator now treats extra unused material as irrelevant to that decision rather than a reason to rerun or force repair. This tolerates unused content, not malformed repair instructions.

## Evidence boundaries

The reviewer checks the pilot/case identities, training role, saved reasoning spans and response schema. It hashes the read files and checks them again before saving a new report. Active pilot locks and inconsistent source identities/spans are rejected. Malformed saved responses are reported as invalid, rather than converted into skips. A schema-valid repair decision found in another run is flagged for remaining checks; the review never executes it.

The supplied replies also show reasons for caution about auditor quality: one repeats instructions in its unused hint, and another supplies a specific modality assertion. Their clinical statements have not been independently established. Those strings remain recorded data and are not reused as guidance. Correctly interpreting a skip does not establish that the auditor's medical judgement was correct.

The regression fixture `tests/fixtures/nonrepair_audits.json` preserves the reported auditor objects as data. Test pilot contexts are explicitly synthetic; only the original Windows archive can establish its complete image/reasoning context. No new medical generation or training was performed to prepare this fix. Full admission and medical adaptation remain later work.
