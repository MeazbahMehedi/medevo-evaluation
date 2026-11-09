# Bounded training-data collection

This checkpoint collects evidence for twelve fixed training questions: six from VQA-RAD and six from SLAKE. It reuses the three cases from the first Windows repair pilot and processes exactly nine additional questions. The purpose is to find potential training candidates before implementing the remaining admission checks. It is not a benchmark evaluation or an adapter-training run.

## Install the update

Replace matching source, configuration, documentation and test files. Keep the existing `.venv312`, `data` and `outputs` folders. Do not put the new source inside a second project directory.

No additional Python package is required. In PyCharm, retain `.venv312\Scripts\python.exe` as the project interpreter. Use the repository root as the working directory.

Run this test checkpoint first:

```powershell
& '.\.venv312\Scripts\python.exe' -m unittest discover -s tests -v
```

The current suite count is recorded in the [README](../../README.md). Review the result before starting the collection.

## Run the approved collection

Keep the PC awake and Ollama running with the already downloaded `qwen2.5vl:3b` model. The local endpoint, model tag, prompts and token budgets come from the existing pilot configuration. No new model download or pull is requested.

From the repository root, run:

```powershell
& '.\.venv312\Scripts\python.exe' '.\evaluate.py' collect-training --previous-pilot '.\outputs\20261108T044904Z_repair_pilot_01186c'
```

The saved-audit review folder is not an additional input. The collector reads the original raw requests and responses, checks them against the current non-repair validation policy, and records the interpretation separately. All original pilot files remain unchanged.

Nine new cases may take roughly 5–15 minutes if the PC behaves like the earlier three-case pilot. Repairs and slow auditor calls can extend that estimate. The worker has a one-hour limit and a 10 GiB Python-process memory limit under the current settings. The external Ollama service is excluded from that memory measurement. A sleep interruption or service failure leaves an incomplete record; do not repeatedly rerun the command to obtain a more favorable result.

## Fixed controls

| Item | Rule |
|---|---|
| Selection | The first six training IDs per dataset under the existing SHA-256 ordering and seed 7 |
| Reused cases | Original two VQA-RAD and one SLAKE IDs; their saved decisions remain one unresolved and two skips |
| New workload | Nine questions, one initial rollout each, at most one supported repair attempt per failed rollout |
| Primary model | Same pinned SmolVLM2-500M checkpoint, CPU float32, four threads, 512-pixel single-view input |
| Auditor | Existing local `qwen2.5vl:3b` configuration; unchanged request instructions, temperature 0 and seed 7 |
| Token caps | Original pilot configuration: 128 reasoning, 128 continuation, 32 answer-readout, 512 auditor tokens |
| Verification | Existing `plain-whole-answer-em-v1` training-only readout matching; no synonym or numeric-equivalence expansion |
| Evaluation | Original boxed scorer and final test protocol remain unchanged |
| Training | Zero optimizer steps, zero exported SFT targets, no adapter creation |

The collector checks the original model settings, data identity, prompts, budgets, source hashes and saved requests. It accepts the documented validator correction for unused skip/unresolved fields, without converting those actions into repairs. Changed or incomplete evidence stops reuse before new inference.

An Ollama model tag alone does not establish immutable weight identity. The earlier reported local model ID was `fb90415cde1e`; retain `ollama ps` or `ollama list` output with the experiment record and report any model replacement. The collector does not establish historical weight identity from the tag.

## How the cases are recorded

For each new question, the unchanged pilot code generates reasoning and a separate short answer readout. A literal readout match becomes a potential correct-rollout candidate. Otherwise, the auditor may request a repair, skip the case or leave it unresolved. A repair must pass the existing hint and prefix checks before continuation. Its answer and auditor review must pass before it becomes a preliminary repaired candidate.

Question-echo labels are diagnostic only. They do not alter the generated response, route the case differently, change a score or turn a mismatch into a match.

| Initial readout label | Meaning |
|---|---|
| `answer_match` | Complete readout matches the reference under the existing lexical rule |
| `question_echo` | Complete readout repeats the question under that same normalization |
| `answer_mismatch` | Complete non-echo readout differs from the reference |
| `truncated_readout` | Readout reached its token cap |
| `empty_readout` | No non-whitespace answer |
| `not_recorded` | The case stopped before producing a readout |

These labels are mutually exclusive; empty and truncated readouts take precedence. They detect literal echoes, not every paraphrased repetition. A phrase such as “T2-weighted MRI scan” is not automatically equated with “mr flair.” References remain confined to verification and saved research evidence, outside the model and auditor requests.

## Output files

Each invocation creates a new `outputs\..._training_collection_...` folder.

| Path | Contents |
|---|---|
| `run.json` | Fixed twelve IDs, three reused IDs, controls, hashes and limitations |
| `previous_pilot/` | Byte-preserved copy of the source pilot |
| `cases/01/` through `cases/12/` | Original or new reasoning, readouts, requests, decisions and case outcomes |
| `collection_record.json` in each case | Separate interpretation, reuse provenance and readout labels; original `case.json` is preserved for reused cases |
| `candidate_bank.json` | Index of potential correct and repaired rollouts; not a training dataset |
| `repair_attempts.json` | Index of schema-valid repair decisions, including attempts rejected by later checks |
| `summary.json` | Coverage, outcome counts, echo/mismatch counts, candidates and new versus historical auditor calls |
| `environment.json`, `model.json` | Current environment and loaded primary-model identity |
| `resources.json`, `console.log` | Worker status, limits, process-tree RAM and console record |

Read `summary.json` after the command finishes. A completed collection should show twelve recorded cases, three reused cases and nine new cases. “Complete” means the fixed collection finished; it does not mean there are successful repairs or medically valid training examples.

Correct readouts can accompany unsupported or truncated reasoning. All potential candidates still need the remaining quality, boundary, leakage and learnability checks before admission. The candidate bank explicitly records pending admission, including initial-reasoning truncation. All failed, skipped and unresolved cases remain available for analysis.

If the complete collection has zero potential candidates, the next action is to reassess the primary model/auditor setup before training. If candidates exist, inspect their evidence and implement the remaining admission checks. Neither outcome triggers automatic training or additional sampling. Final baseline/adapted benchmark results and the technical report remain later work.

## Verification limits

Development tests cover fixed selection, archived evidence reuse, unchanged request controls, question echoes, candidate indexing, repair rejection, simulated repair success, infrastructure failures, worker locks and the supervised CLI path. Model and auditor responses in those new integration tests are scripted fixtures. They do not establish live repair quality. The nine-question collection must still run on the Windows PC with the real local auditor.
