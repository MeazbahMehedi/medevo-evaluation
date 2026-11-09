# Expanded feedback output-budget experiment

This experiment uses all 79 VQA-RAD feedback questions and exactly 40 English SLAKE feedback questions. It includes the earlier twenty questions, keeps their model digest and generation controls, and changes only the generation payload's `options.num_predict` from 512 to 1024. It does not use test questions, train an adapter or collect repairs.

The workflow has two explicit stages. Run the offline selection audit first and review its result. The larger generation run starts only after that review.

## Stage 1: offline selection audit

No installation or weight download is required. In the PyCharm PowerShell terminal:

```powershell
& '.\.venv312\Scripts\python.exe' '.\compare_feedback_budget.py' inspect
```

Defaults are the existing prepared `data/benchmarks` bundle and completed `outputs/qwen35_feedback_20_v1` baseline. Optional `--data-dir` and `--baseline-dir` arguments can identify different existing locations, without rebuilding data.

The audit makes zero HTTP/model calls. It:

1. Validates the prepared manifests, counts, image identities and cross-role separation using the existing bundle validator.
2. Requires all 79 VQA-RAD feedback records and selects the first 40 SLAKE feedback records under the existing seed-7 SHA-256 ordering.
3. Requires 119 distinct IDs, membership in the feedback manifest and zero ID overlap with records whose current role is train or test.
4. Requires the original twenty IDs, saved selection, prompt, thinking setting, model and options to match the original completed baseline.
5. Reproduces the earlier saved scores and validates all twenty original request images/prompts/options against current inputs. Raw responses must match the saved predictions.
6. Requires sixteen originally nontruncated responses and includes all earlier twenty questions in the expanded selection.
7. Saves exact selected records, hashes, controls, budgets and the decision rule before any generation.

An ID such as `vqa_rad:train:001713` describes its original source partition. Selection eligibility comes from its current manifest role. String matching on `:train:` must not replace the explicit role checks.

The plan is saved in `outputs/qwen35_feedback_119_plan_v1/`, with `plan.json` and `selection.json`. Inspecting an unchanged existing plan is safe and makes no new calls. Changed inputs or a conflicting saved plan are rejected rather than overwritten. Raw selection records contain references for local verification; they are excluded from generation payloads.

The real Windows selection audit has not yet been run. Preparation tests use controlled data; they cannot certify the local bundle. Require the Windows audit result before launching Stage 2.

## Stage 2: separately approved generation

Only after the audit has been reviewed:

```powershell
& '.\.venv312\Scripts\python.exe' '.\compare_feedback_budget.py' run
```

The command requires the saved plan and revalidates it. It writes `outputs/qwen35_feedback_119_1024_v1/`, refusing any previous attempt. It snapshots the plan and original baseline evidence, preserving all earlier files. There are at most 119 generation requests, with no retries or replacement questions. The baseline twenty are rerun within these 119, not added as twenty extra calls.

| Control | Fixed setting |
|---|---|
| Model | Same installed `qwen3.5:9b` digest as the earlier run |
| Ollama | Same recorded version as the earlier run |
| Prompt | `Please reason step by step, and put your final answer within \boxed{}.` |
| Thinking | `false`; server metadata must confirm support |
| Temperature / seed | 0 / 7 |
| Output ceiling | 1024 new tokens |
| Other sampling | Same earlier top-k 20, top-p 0.95, presence penalty 1.5 |
| Context / image | 4096-token context; clean RGB input, longest edge at most 512 before backend preprocessing |
| Device request | CPU, four threads, zero GPU layers |
| Request/client budgets | At most 300 seconds per request; 10,800 seconds for the generation batch |
| RAM guards | At least 10 GiB available before launch; stop below 4 GiB between requests |

No generation request contains references, answer-type labels, prior outputs, dataset IDs or repair hints. The prompt is verified against Appendix D.1 of the professor's PDF, page 29. Quantized CPU execution, disabled thinking and the stated budgets are local study adaptations, not a verified reproduction of every paper setting.

The earlier measured averages were 45.2079 seconds per VQA-RAD question and 28.7072 seconds per SLAKE question. Scaling to 79 + 40 at those rates gives 78.66 minutes. Allow approximately **80–160 minutes** for planning with the larger output ceiling. This is uncertain because the additional questions may differ in difficulty and response length. The 180-minute client budget is a stop policy, not a completion promise. Metadata preparation is outside the generation budget.

HTTP timeouts can leave work running in Ollama. Inspect server state before another attempt. RAM guards are not continuous memory enforcement during a request and do not measure peak model RSS. `/api/ps` is stored as allocation metadata only. Keep the PC awake.

## Scoring change: retain both versions

The historical `src/scoring.py` and its hash are unchanged.

| Scorer | Meaning |
|---|---|
| `boxed-strict-em-v1` | Original last-box parser and existing lexical scoring |
| `boxed-format-unwrapped-em-v1` | Companion parser strips balanced `\text{...}` and `\textbf{...}` wrappers inside the parsed final answer, then uses the same exact-match/normalization/F1 functions |

The companion supports nested balanced wrappers. It does not remove other LaTeX commands, infer synonyms, extract an answer from prose, use substring matches or repair missing boxes. An empty answer after stripping remains a parsing failure. Accuracy under the two scorer versions must be labelled separately.

Each new response is scored under both versions. The saved earlier twenty responses are also rescored under both versions without changing their source files or making additional model calls. This separates output-budget changes from formatting-score changes.

## Frozen decision rule

Use the full denominator of 119 and the formatting-aware valid-box parser:

- If the run is complete and **at most 11 answers** lack a valid box, adopt **paper prompt + 1024-token ceiling** for the subsequent primary baseline configuration.
- If the run is complete and **12 or more** lack a valid box, retain **paper prompt + the existing 512-token baseline configuration**, with the formatting limitation reported.
- If the run is incomplete or identity/integrity checks fail, make **no adoption decision**.

Eleven of 119 is 9.24%; twelve is 10.08%. The decision is about formatting, not clinical competence, and does not depend on observed accuracy. Full denominator and threshold are not changed after results arrive.

The paper prompt is used with both caps. A concise-prompt condition would be separate and would need its own approved specification and run. This script neither runs it nor automatically adopts it. No test-set results are used for configuration selection.

## Reporting and reproducibility

`summary.json` and `report.md` contain both scorer versions, per-dataset and open/closed breakdowns, coverage, accuracy, boxed-answer rate, truncations, latency and response lengths. Timing and length statistics are inherited from the earlier evaluator, including mean, median, minimum and maximum. Raw responses, stop reasons and provider durations are retained per case.

`paired_earlier_twenty.json` contains both old and new responses and both versions of their scores. For the original sixteen nontruncated responses, report raw-text equality separately from parsed-answer equality. Temperature zero does not guarantee exact output reproduction. Full 119-question accuracy is not a paired gain over twenty-question accuracy; only the shared twenty form a paired comparison.

| Artifact | Purpose |
|---|---|
| `selection_audit/` | Snapshot of reviewed IDs, membership checks, settings and decision rule |
| `baseline/` | Whitelisted original baseline records, copied and hash-checked |
| `run.json`, `model_details.json`, `environment.json` | Actual counters, model/backend identity and environment |
| `cases/001/` through `cases/119/` | Reference-free requests and raw replies |
| `predictions.jsonl` | Flushed predictions and failures |
| `scored_original.jsonl`, `scored_unwrapped.jsonl` | Both scoring views, including references for verification |
| `paired_earlier_twenty.json` | Matched old/new records and reproduction checks |
| `summary.json`, `report.md` | Metrics and the predeclared decision outcome |

A completed HTTP response with no box, wrong answer or truncation remains evidence; it does not trigger a retry. Transport failure, unexpected thinking output, changed data/code/baseline or changed model/version stops or invalidates the run. Incomplete-run primary accuracy is unset, and partial generated-only diagnostics must not be presented as final scores.

Raw requests contain encoded benchmark images. Keep run folders private and excluded from public Git unless rights and contents are reviewed. The repository includes neither benchmark images nor new live model outputs.

Preparation verification uses controlled data and a scripted transport. It tests both sides of the 11/12 threshold, role-safe selection, original-record preservation, wrapper handling, paired reproduction, changed plans and partial failures. It does not establish real Qwen3.5 accuracy or certify the Windows manifest.

The complete Linux development test suite passed: 194 tests in 27.777 seconds. No live model generation was performed for this update.
