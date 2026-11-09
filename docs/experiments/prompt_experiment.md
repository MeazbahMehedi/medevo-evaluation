# Answer-format diagnostic

This is a feedback-only comparison of the original reasoning instruction and a concise answering instruction. The new instruction was selected after development responses showed frequent missing answer boxes. This is an exploratory prompt adaptation, not an independent test-set result or evidence that MedEvo training works.

## Update an existing project

Keep the existing `.venv312`, `data`, `outputs`, `.git` and private environment files. Keep the existing `settings.json` and `configs/benchmark_protocol.json` as well; the diagnostic does not require changes to either file.

No new dependency is needed. The dataset sources, image export, default protocol, primary model and scorer are unchanged. Do not rerun data preparation. The existing 38 tests remain, with seven added diagnostic checks:

```powershell
& '.\.venv312\Scripts\python.exe' -m unittest discover -s tests -v
```

Run tests from the project folder so Python can import `src`. The expected suite contains 45 tests.

## Run the approved comparison

From the repository root in PowerShell:

```powershell
& '.\.venv312\Scripts\python.exe' '.\evaluate.py' compare-prompts --vqa-rad-run '.\outputs\20261108T023429Z_vqa_rad_feedback_119aa2' --slake-run '.\outputs\20261108T023954Z_slake_feedback_a5e572'
```

These paths identify the two completed Windows development runs. The command reads their original predictions and scores; it does not regenerate or overwrite them. It then runs five new answers on each dataset, sequentially, using the same selected questions.

The reference instruction remains:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

The trial instruction in `configs/answer_format.json` is:

```text
Answer the question directly using a word, number, or short phrase.
Put your final answer inside \boxed{}.
Do not include an explanation.
```

There is one shared trial instruction for both datasets and every answer type. It includes no answers, reference-derived clues or examples copied from the benchmarks.

## Controls checked before generation

The command requires exactly five completed feedback predictions per dataset, matching the deterministic selection made with seed 7. It checks the dataset metadata hash, selected IDs, model settings and revision, primary-model implementation, scoring implementation, Python version and direct package versions. It recomputes the original summary from raw responses and checked references and rejects inconsistent stored metrics. Baselines with active locks or unfinished workers are rejected.

The comparison uses the original model settings, single 512-pixel image view, 128-token budget, greedy decoding and strict boxed-answer scorer. The execution and scoring functions are shared with ordinary evaluation runs. The only experimental input changed is the appended answering instruction; the protocol and experiment IDs also change to identify the trial.

Source-path checks accept both Windows backslashes in earlier run records and POSIX separators in new records. The default `run` command still requires its prepared protocol to match exactly. The diagnostic creates explicitly labelled trial metadata without altering the prepared bundle or the default protocol.

## Saved evidence

A new `outputs/<timestamp>_prompt_comparison_<id>/` folder contains:

| Path | Purpose |
|---|---|
| `comparison.json` | Original and trial prompts, controls, selected IDs, source hashes, scores, format rates, timing, resource records and changes in percentage points |
| `comparison.md` | A short table and interpretation limits |
| `baseline/vqa_rad/` | Copies of the original VQA-RAD run, predictions, scores, metrics and resources |
| `baseline/slake/` | Copies of the original SLAKE evidence |

Two fresh trial folders under `outputs/` retain each raw prediction, per-question score, environment, model identity and worker resource record. Their paths appear in the console summary and comparison JSON. Original baseline files are hashed before and after the comparison; a change invalidates the report.

Format success is the percentage of the five selected questions with a valid final answer box. Strict EM is the existing case-insensitive exact-match score. Both require complete generation coverage before being reported as full trial metrics. An infrastructure failure stops the next dataset trial, retains available evidence and marks the comparison incomplete. A low score or format failure is a measured outcome and does not by itself stop the experiment.

The command does not automatically choose the better prompt, retry wrong answers, add a fallback parser or change a reference answer. Repeating the command creates a separate experiment. There is no automatic resume for a paired comparison; retain a stopped comparison and review its error before starting another attempt. The original baseline runs remain available for inspection and scoring. Old runs cannot be resumed through a changed code version, as required by their existing identity checks.

## Interpretation

The selected ten feedback questions were already examined during development. They cannot estimate final benchmark performance or support a statistically reliable improvement claim. Report format compliance separately from exact match. A response can contain the reference concept but fail formatting, or satisfy formatting while giving a wrong answer. Lexical scores still do not establish clinical correctness.

Timing is observed across separate attempts on the same machine, with possible cache and background-load differences. Do not present the observed difference as a controlled speedup. RAM is reportable only when its measurement is marked plausible. Resource measurements concern the Python worker, not the separate Ollama service.

The concise instruction suppresses explanations, so its outputs do not establish the reasoning trajectories needed for MedEvo repair. This diagnostic does not export training targets, run an auditor, train a model or use either final test set. A later reviewed decision is needed before adopting any prompt for training or final evaluation. If an adapted final-evaluation prompt is adopted, the baseline and adapted model must use the same frozen evaluation settings and the deviation from the paper must be disclosed.

## Preparation outcome

The paired command completed in the Linux development environment. The concise prompt produced no valid boxes in either five-question trial, so it did not improve format compliance. The full negative outcome and raw responses are retained in `prompt_verification.json`; the interpretation is in `prompt_verification.md`. The Windows command above records the same diagnostic independently on the target PC. Successful execution should not be confused with a successful prompt adaptation.
