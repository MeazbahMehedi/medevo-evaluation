# Single-case schema-grounding check

This approved check makes one new auditor call on the first ID in the original twelve-case training collection: `vqa_rad:train:000201` in the prepared bundle. It keeps the original medical audit instruction and appends the complete JSON schema as text. The same schema remains in `response_format`.

All other request fields remain unchanged: model, endpoint, image, question, saved reasoning, steps, entropy, temperature, seed and 512-token budget. The existing local-only transport, 180-second request timeout, audit validator and hint checks are reused. The shorter-instruction experiment is not adopted.

## Install and run once

Core `src/` modules and configurations remain unchanged. The standalone `check_auditor.py` also supports the separately approved [7B model comparison](auditor_model_comparison.md).

Keep Ollama running. From the project terminal, run:

```powershell
& '.\.venv312\Scripts\python.exe' '.\check_auditor.py' --collection-dir '.\outputs\20261108T061937Z_training_collection_9120a6'
```

Run this once and review the returned JSON before any further model calls. There is no automatic retry, replacement case or additional sampling. The first case is selected by the existing collection order rather than by a favorable reference answer.

The command validates the prepared bundle and archived collection before making the one call. These are local integrity checks, not new model inference. It does not load the primary model, generate new reasoning/readouts, continue a response, export training targets, admit candidates or train an adapter.

## Evidence and interpretation

A separate `outputs\..._single_audit_...` folder contains:

- `run.json`: fixed selection, model/configuration, source hashes, request change and limitations.
- `environment.json`: current environment metadata.
- `evidence/`: the unchanged original case, question, reasoning, readout, reference, steps and audit request/response.
- `trial/audit_request.json` and `trial/audit_response.json`: the actual new request and raw reply.
- `trial/request_error.json`, if needed: a captured failure with no automatic retry.
- `summary.json`: original and trial assessments, raw audit content and zero-training fields.

References remain in local verification files and never enter the network request. The original collection is hashed before and after the call. Only the copied request's system text is extended; every other request field and original source file is preserved.

`status: complete` means a response was recorded. It does not mean the audit is coherent, correct or usable. A valid skip or unresolved decision remains a non-repair. An invalid repair remains invalid. Even a repair passing structural and hint checks is labelled `repair_requires_clinical_review` and does not trigger continuation.

This is a single integration observation after inspecting earlier failures. It cannot establish clinical accuracy, generalization, reliable error boundaries or the overall feasibility of the MedEvo adaptation. The next decision must follow review of this saved reply; no further prompt variants or medical collection are automatic.

## Development verification

Seven new checks cover the exact appended schema, unchanged request fields, fixed first-case selection, one-call bound, reference-free requests, preserved original files, invalid-audit handling, transport failure and prevention of continuation/training. A scripted local HTTP server received exactly one request in the transport test. No live Qwen medical call was made in the development environment. The complete test-suite result is recorded in [test_results.txt](../verification/test_results.txt).

The schema-grounding recommendation is documented in [Ollama's structured-output guide](https://docs.ollama.com/capabilities/structured-outputs). Its omission from earlier requests is a plausible implementation concern, not a demonstrated cause of their medical-auditing failures.
