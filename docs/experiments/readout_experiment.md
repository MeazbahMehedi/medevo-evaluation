# Saved-reasoning readout experiment

This approved diagnostic compares twelve existing answer readouts with twelve new text-only readouts from the same pinned primary model. It reuses the saved questions and reasoning; it does not generate new medical reasoning, call the auditor or train an adapter.

## Why this experiment is needed

The twelve-case Windows collection completed in 202.625 seconds with zero potential training candidates. Its readouts included one question echo, ten answer mismatches and one truncated response. Inspection showed that case 01's reasoning answered “yes” before the readout repeated the question, and case 05's reasoning named “aorta” before its readout produced a long number. Case 09 contained the reference concept “center” within a longer answer that included additional claims. These observations establish a readout reliability concern. They do not establish medically correct reasoning or acceptable training targets.

Other cases contain answer-content mismatches, and several auditor responses repeat instructions or contain unusable evidence. This diagnostic addresses the readout method only. The auditor and remaining training-admission checks will still need attention.

## Install and test

Keep the existing `.venv312`, `data` and `outputs` folders. No additional Python package is needed. Retain the Python 3.12 interpreter in PyCharm.

From the project terminal, run:

```powershell
& '.\.venv312\Scripts\python.exe' -m unittest discover -s tests -v
```

The current suite count is recorded in the [README](../../README.md). Review the test result before starting inference.

## Run the diagnostic

Use the completed collection already present on the PC:

```powershell
& '.\.venv312\Scripts\python.exe' '.\evaluate.py' compare-readouts --collection-dir '.\outputs\20261108T061937Z_training_collection_9120a6'
```

This uses the cached primary checkpoint and makes twelve short generation calls, with no automatic retries or extra questions. Ollama is not used. Keep the PC awake. The current worker has a twenty-minute time limit and a 10 GiB Python-process memory limit. The actual run may finish much sooner; development tests used scripted model responses and do not establish target-PC inference time.

If the command fails, retain the output folder and inspect its summary and error before repeating it. The experiment does not resume or silently replace failed cases. Repeating the main command creates a separate run and makes new calls; do not rerun it merely to obtain a better result.

## What changes

The original readout used an image/question conversation, the generated assistant reasoning and a follow-up instruction. The trial uses one user message containing a fixed extraction instruction, the original question and the saved response. It supplies no image, reference, dataset annotation, answer type or auditor hint.

The instruction asks for only the answer already stated in the saved response, prohibits solving the question again or adding/correcting information, and requests `unresolved` if the response states no answer. For yes/no questions, it asks for a yes/no readout only when one is stated. This is one exploratory replacement method, combining a single-turn prompt and text-only input; it does not isolate the effect of each component.

| Control | Rule |
|---|---|
| Questions | Same twelve training IDs, six from each dataset |
| Reasoning | Existing captured text, unchanged |
| Primary model | Same SmolVLM2-500M checkpoint and revision |
| Generation | CPU float32, four threads, greedy decoding, at most 32 new tokens |
| Model inputs | Fixed instruction, question and saved reasoning only |
| Answer matching | Existing `plain-whole-answer-em-v1`; no new answer extraction or synonym rules in the scorer |
| Original results | Preserved and copied as comparison evidence |
| Auditor and reasoning calls | Zero new calls |
| Training and admission | Zero updates or target exports; no automatic candidate admission |

The diagnostic validates model/source controls, data-bundle identity, fixed training selection and question/reference consistency before generation. Model and scoring code are unchanged. The worker hashes its frozen requests and evidence and rejects repeated execution in a folder containing readout results.

## Outputs

Each run creates an `outputs\..._readout_comparison_...` folder.

| File or folder | Contents |
|---|---|
| `run.json` | Model/settings, IDs, prompt, code and evidence hashes, scope and limitations |
| `inputs.json` | Twelve reference-free, text-only generation requests |
| `original_run.json`, `original_summary.json` | Copied collection metadata |
| `evidence/01/` through `evidence/12/` | Saved original case, question, reasoning, readout, verification and collection record |
| `readouts/01.json` through `readouts/12.json` | New raw readouts, generation duration, token count and cap warning |
| `comparison.json` | Paired original/trial responses, references, matches and diagnostic labels |
| `comparison.md` | Compact paired readout table |
| `summary.json` | Coverage, overall/per-dataset lexical matching, gained/lost matches and zero-training fields |
| `environment.json`, `model.json`, `resources.json`, `console.log` | Runtime and model evidence |
| `worker_error.json`, if present | Recorded failure with partial outputs retained |

A completed run should record twelve new readouts. Missing or failed generation leaves full trial accuracy unset. Empty and truncated readouts remain visible and cannot count as matches.

## Interpreting the result

Original collection outcomes and scores are not rewritten. A higher trial exact-match rate could reflect a more faithful answer readout. It does not show improved medical reasoning: the underlying reasoning text is unchanged. Even a reference-matching extraction could introduce information absent from the saved response. Inspect each new answer against that response before making a fidelity claim.

The method was selected after inspecting these twelve training cases. Its result is an exploratory diagnostic on those cases, not final benchmark performance or evidence of generalization. Cases containing incorrect or unsupported reasoning remain unsuitable for training until the required admission checks pass. No candidate is admitted by this command, and the original boxed final-evaluation protocol stays unchanged.
