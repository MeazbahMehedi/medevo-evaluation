# Whole-answer scoring diagnostic

This diagnostic compares two fixed scoring rules on the saved prompt-comparison outputs. It reads the same feedback answers already generated; it does not change a prompt, run a model, call an auditor or train an adapter. The original boxed-answer scores remain intact.

## Update and run

No dependency or dataset download is added. The principal additions are `src/answer_diagnostic.py`, its tests and documentation, and one command in `evaluate.py`.

From the repository root in PowerShell:

```powershell
& '.\.venv312\Scripts\python.exe' '.\evaluate.py' diagnose-answers --comparison-dir '.\outputs\20261108T030115Z_prompt_comparison_2f6e16'
```

The argument identifies the completed Windows prompt comparison, not one individual trial folder. Do not rerun `compare-prompts`: the saved comparison already contains the necessary selections, original evidence and trial paths.

The expanded test suite contains 56 checks:

```powershell
& '.\.venv312\Scripts\python.exe' -m unittest discover -s tests -v
```

## Fixed rules

The existing `boxed-strict-em-v1` scorer requires a valid last answer box and compares its content to the reference. This diagnostic imports that scorer unchanged and verifies its stored summary against raw predictions.

The separate `plain-whole-answer-em-v1` rule:

1. Converts the entire raw response and reference to lowercase.
2. Collapses whitespace and trims outer whitespace.
3. Removes contiguous terminal `.`, `!` and `?`, then trims any remaining outer whitespace.
4. Compares the resulting complete strings for equality. Empty responses are wrong.

No sentence selection, substring matching, box removal, prefix stripping, synonym expansion, Unicode compatibility conversion or number-word conversion is performed. The reference is read from the checked feedback manifest, never from fields added to a prediction. The rule is identical for all questions and both datasets.

| Raw response | Reference | Whole-answer match |
|---|---|---|
| `Yes.` | `yes` | Yes |
| `X-ray.` | `X-Ray` | Yes |
| `MRI` | `MRI` | Yes |
| `CT scan` | `CT` | No |
| `The image is a CT scan.` | `CT` | No |
| `two` | `2` | No |
| `\boxed{CT}` | `CT` | No |
| `no mass` | `mass` | No |

A response with a correct answer box can pass boxed scoring and fail whole-answer scoring. A correct short unboxed answer can do the reverse. Extra explanation and qualifiers remain part of the whole response. These are lexical measures, not clinical adjudication.

## Integrity checks

The command accepts a completed five-question-per-dataset feedback comparison. It verifies the prepared dataset identity and manifests, deterministic IDs, dataset/role, saved prompts, token budget, model settings, and original model and boxed-scorer source hashes. Windows source-path separators are supported. Baseline copies must match the hashes recorded by the earlier comparison.

It reads the immutable baseline copies inside the comparison folder and the trial folders recorded by `comparison.json`. It recomputes boxed summaries and checks them against both saved run metrics and the comparison. Duplicate or foreign predictions, unfinished runs, inconsistent scores and active locks cause a failure before a diagnostic report is written.

The saved-run files, input comparison and dataset metadata are checked again before writing the new result folder; the data manifests are verified when loaded. The diagnostic does not overwrite source predictions, metrics, comparison reports, dataset manifests or references. Scoring functions retain missing/error counts and null full-coverage accuracy for incomplete inputs; the paired command requires complete saved runs.

## Output

A unique folder `outputs/<timestamp>_answer_diagnostic_<id>/` contains:

| File | Contents |
|---|---|
| `diagnostic.json` | Both metric definitions, source hashes, input identities, selections, counts, coverage, scores and answer-type breakdowns |
| `diagnostic.md` | Side-by-side table and interpretation limits |
| `vqa_rad_original.jsonl`, `vqa_rad_trial.jsonl` | Questions, raw responses, references, original boxed results and separate whole-answer comparisons |
| `slake_original.jsonl`, `slake_trial.jsonl` | The corresponding SLAKE records |

The console prints a short summary and the diagnostic folder. `new_inference_performed` is false. A new diagnostic folder records a scoring operation, not another model experiment. Repeating the command creates a separate report from the same saved evidence.

## Interpretation and research boundary

The new rule was proposed after feedback inspection. This is exploratory development, and the same ten feedback questions have already informed the investigation. Their scores cannot be reported as final benchmark accuracy or an independent estimate of performance.

Changing the scorer does not improve a model. The original strict scores and the new diagnostic scores must be named separately. Comparisons between the long original responses and short trial responses also reflect the expected answer format, not just answer content. Do not present their difference as a medical-learning gain.

The new scorer is not adopted by the normal evaluator and does not change its default parser, prompt or metrics. Adoption of any final protocol requires a separate reviewed decision and the same frozen rule for baseline and adapted models. A complete MedEvo evaluation still requires the repair/admission method, medical adaptation, held-out test runs, analysis and a report. Short answers alone do not provide the reasoning trajectories needed for repair.

The Linux development run produced 3/5 whole-answer matches for each concise trial, while the boxed scores remained 0/5. These are local exploratory observations, not results from the Windows diagnostic or final test sets. The exact records are in `answer_verification.json`.
