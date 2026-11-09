# Auditor-only instruction experiment

This approved diagnostic compares the twelve saved auditor responses with twelve new responses using a shorter system instruction. It reuses the original training questions, images, reasoning, steps and entropy proposals. The only changed request field is `messages[0].content`.

The purpose is to test whether a clearer instruction improves response validity and coherence. It does not require a repair decision: justified skips and unresolved decisions remain acceptable. It does not generate new primary-model reasoning, readouts, continuations, review calls, candidates or adapters.

## Update and test

No dependency or model download is added.

The current suite count is recorded in the [README](../../README.md). Run the test checkpoint from the project terminal:

```powershell
& '.\.venv312\Scripts\python.exe' -m unittest discover -s tests -v
```

## Run the approved experiment

Keep Ollama running with the already installed `qwen2.5vl:3b` model. Do not pull or replace its weights between the paired conditions. From the repository root, run:

```powershell
& '.\.venv312\Scripts\python.exe' '.\evaluate.py' compare-audits --collection-dir '.\outputs\20261108T061937Z_training_collection_9120a6'
```

The source is the original twelve-case collection, not the subsequent readout comparison. The primary model is not loaded. The command makes at most twelve new audit calls, without retries, and preserves every original and trial decision. A transport failure stops the remaining calls and leaves an incomplete report. Invalid saved responses remain invalid and are never rewritten into repairs.

Keep the PC awake. The unchanged configuration allows 180 seconds per request and one hour for the Python worker. Its memory measurement and 10 GiB process-tree limit apply only to the Python HTTP worker; they exclude the external Ollama model's memory and CPU use. The diagnostic does not measure Ollama peak memory.

## Fixed controls and changed instruction

| Item | Rule |
|---|---|
| Selection | Same twelve training IDs, six per dataset |
| Baseline | Existing raw auditor responses; no repeated baseline calls |
| Trial model | Same local `qwen2.5vl:3b` tag and endpoint |
| Input evidence | Same question, image, generated reasoning, heuristic steps and entropy |
| Sampling | Existing temperature 0, seed 7 and 512-token auditor budget |
| Output contract | Identical JSON schema and existing cross-field validator |
| Hint checks | Existing conservative instruction/reference-term checks |
| Changed field | System instruction only: 172 words to 143 words in the uploaded baseline |
| Training | No continuation, admission, optimizer step, target export or adapter creation |

The shorter instruction keeps the four reasoning-error families, first-error indexing, supported-prefix requirement, non-repair branches, hint restrictions and visible-evidence requirement. It presents the action rules separately. It does not provide a reference answer or demand a repair.

Before any network calls, the command checks collection identity, model/configuration controls, source fingerprints, prepared images and the saved requests. Each original request is reconstructed from the pinned inputs and compared with the archive. A deep copy replaces only the system text. The worker checks this single-field difference again before starting.

The same Ollama tag does not independently prove immutable historical weights. Preserve the local model ID with the experiment record. This experiment cannot eliminate model/runtime changes that occurred between the original and trial calls.

## Outputs

The command creates a new `outputs\..._audit_comparison_...` folder.

| File or folder | Contents |
|---|---|
| `run.json`, `inputs.json` | Fixed selection, controls, shorter instruction and request/evidence hashes |
| `original_run.json`, `original_summary.json` | Copied collection metadata |
| `evidence/01/` through `evidence/12/` | Original case, question, reasoning, readout, reference, steps and raw auditor requests/responses |
| `audits/01/` through `audits/12/` | Trial request, raw response and attempted-call status |
| `comparison.json` | Paired raw audit contents, validation results, hints, evidence and errors |
| `comparison.md` | Paired outcomes with trial hints and evidence for review |
| `summary.json` | Baseline/trial validity counts, actions and hint-check results |
| `environment.json`, `resources.json`, `console.log` | Environment and Python HTTP-worker execution records |
| `worker_error.json`, if present | Failure record; partial results are retained |

In the existing archive, current validation yields eleven valid audit structures and one invalid audit. The eleven include eight skips, two unresolved decisions and one repair rejected by the hint check. A trial can finish all twelve calls while still producing invalid or incoherent decisions. “Complete” describes workload coverage, not scientific success.

The reported `schema_valid_count` uses the existing validator, including its cross-field rules. Thus a JSON object requesting a repair with step 0 is invalid even if it matches the basic field types. A higher validity rate does not establish clinical correctness. A repair passing the hint check is labelled `repair_requires_clinical_review`; it does not become a training example or trigger continuation. Non-repair auxiliary fields remain recorded and unused, even if they contain poor-quality text.

Review the paired raw outputs for clear, relevant evidence, correct action semantics, plausible first-error boundaries and non-answer-bearing hints. Clinical coherence is deliberately not assigned an automatic pass. This is an exploratory training-case diagnostic selected after inspecting earlier outputs, not a final benchmark evaluation.

## Decision after this bounded run

Do not automatically run more prompt variants, relax validation or increase the sample until a favorable result appears. If the trial remains incoherent or supplies no defensible repair opportunity, reassess the model/setup before further collection. If a credible repair is identified, its clinical boundary and remaining admission checks still need review and implementation before an approved continuation or training run. The professor's final benchmark evaluation and technical report remain incomplete.
