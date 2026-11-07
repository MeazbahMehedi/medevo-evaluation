# Evaluation protocol

Protocol: `medical-vqa-cpu-v1`. Scoring: `boxed-strict-em-v1`. The executable configuration is `configs/benchmark_protocol.json`; model and hardware settings are in `settings.json`.

## Purpose and comparison

The intended final comparison is pretrained baseline versus medical adaptation on two public VQA benchmarks. Phase 2 implements the baseline measurement path. It does not implement the complete MedEvo training loop or accept an adapted checkpoint yet.

The supplied MedEvo report uses Qwen3.5 models, a much larger training pool, a 5,000-question fixed feedback set and seven evaluation benchmarks. This project uses a smaller general vision-language model, two datasets and a restricted CPU budget. Report these differences alongside the results; the study cannot claim exact reproduction of the paper's tables.

## Data identity and roles

Every download is pinned to a commit and verified against both its byte size and SHA-256 digest. VQA-RAD uses the processed Parquet distribution used by the inspected MedUniEval adapter. SLAKE uses the authors' linked distribution, with `q_lang == en`. The exported PNGs are RGB images; the decoded identity hash includes dimensions, mode and RGB pixel bytes. A separate digest checks the exported file bytes.

Canonical IDs are `vqa_rad:<source split>:<row index>` and `slake:<source split>:<qid>`. VQA-RAD's row-based IDs are stable only within the pinned distribution; original patient and question identifiers are not supplied by that mirror.

Preparation assigns roles in this order:

1. Preserve all source test questions selected by the language policy. Their decoded image hashes form the reserved test image set across both datasets.
2. Exclude every non-test question using any reserved test image, regardless of its source dataset.
3. Reserve eligible SLAKE validation images for feedback. Rank eligible VQA-RAD training images by SHA-256 of `7:<image hash>` and reserve the nearest integer to 10%, at least one when any are available. All questions on a reserved feedback image receive the feedback role.
4. Deduplicate remaining adaptation questions by exact decoded image and case-insensitive whitespace-collapsed question. Retain the first canonical ID, record later duplicates, and stop on conflicting reference labels. The final test denominator is not deduplicated.
5. Verify that the three roles have no exact decoded-image overlap. Verify every manifest digest and image digest on preparation, inspection and generation.

Missing non-test questions or references are excluded with a reason. A missing final-test reference stops preparation; it is not silently removed. The observed exclusions and source counts are recorded in [data_sources.md](data_sources.md).

These checks establish exact-image separation. They do not establish patient/case separation, absence of image near-duplicates, or lack of pretrained-model exposure to these public benchmarks. The adaptation pool therefore differs substantially from the paper's pool.

## Model input

The model receives the decoded image and the original question followed by this instruction from Appendix D.1 of the supplied paper:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

This instruction is identical for both datasets and all answer types. Reference answers, answer-derived types, source paths, IDs and hidden dataset annotations do not enter the generation request. Image content is serialized by the processor, without its source filename. The answer type is retained only for scorer breakdowns. The primary model is the unmodified pinned pretrained checkpoint; a synthetic Phase 1 adapter is never used in a medical baseline.

Inference uses CPU float32, four threads, batch size one, greedy decoding (`do_sample=False`), seed 7, eager attention and a single image view. Longest edge is 512 pixels and image splitting is disabled. The generation budget is 128 new tokens. These settings must be the same for the final baseline and adapted comparison unless a separately labelled experiment is declared before test results are examined.

## Answer parsing and metrics

The parser selects the last literal `\boxed{` marker and reads balanced braces through its closing brace. Surrounding whitespace inside the box is stripped. An empty box, absent marker or incomplete last box is a parse failure. There is no fallback to an unboxed sentence and no model-judged extraction.

| Measure | Definition |
|---|---|
| Strict EM accuracy, primary | Percentage of generated questions with a parsed answer equal to the reference after lowercase conversion and outer whitespace removal |
| Normalized EM, secondary | Equality after Unicode NFKC, case folding, whitespace collapse and removal of terminal `.`, `!` or `?` |
| Token F1, secondary | Whitespace-token multiset overlap after the same normalization; harmonic mean of precision and recall |
| Coverage | Successfully recorded generations divided by selected questions |
| Parse failures | Generated responses without a valid final box; scored wrong, not removed |
| Token-limit count | Generations reaching the token cap without an EOS token |
| Open/closed results | The same scores and coverage on the source or reference-derived answer-type groups |

References may contain several words but are treated as one answer string, not an automatically inferred list of acceptable alternatives. Synonyms, spelling variants, medical abbreviations and unit conversions are not matched semantically. Token F1 is a lexical diagnostic, not clinical validation. Repeated tokens are counted rather than collapsed to a set. No fuzzy matching, answer correction or external judge is used.

A generated empty string counts as a format failure. A completed but truncated response is scored under the same parser and also flagged. A generation exception or absent prediction is an infrastructure failure. If coverage is below 100%, primary aggregate accuracies are null. The partial generated-only scores remain explicitly labelled; they must not be presented as a complete benchmark result. A completed subset is still labelled `subset`, even if its requested limit happened to cover the available questions. Use `--full` to request full coverage.

The inspected MedUniEval code includes yes/no substring heuristics, other lexical metrics and optional external judging. This implementation deliberately declares its own strict parser and scorer. Its values must not be placed beside the paper's accuracy as an equivalent reproduction. If an exact MedUniEval comparison is needed later, add and validate a separately versioned scorer before looking at final results.

## Selection, logging and recovery

Question subsets are selected by sorting SHA-256 of `<seed>:<canonical ID>`. Selection is independent of reference answers and input row order. `run.json` records selected IDs, full/subset scope, data metadata hash, direct dependency versions, Python version, all source-module hashes, model identity and the complete settings. The environment and complete installed package list are saved separately.

Generation writes only ID, status, raw response, token count, duration and token-limit flag to `predictions.jsonl`. The scorer joins IDs against the checked manifest. It rejects duplicate or foreign predictions and checks the dataset identity and scoring implementation. References appear only in `scored.jsonl`.

Resume requires the original selection, code, settings, model revision, direct dependency versions and prepared data identity. It skips all existing valid journal records, including recorded question errors. It does not opportunistically retry wrong answers, format failures or errors. An incomplete last append is backed up and removed before continuing; interior corruption causes a failure. A complete last JSON record without a newline is retained. An exclusive `run.lock` prevents simultaneous workers; a stale lock requires confirming process termination before manual removal.

`scored.jsonl` and `metrics.json` can be regenerated from the journal without inference. Incomplete reports can be inspected but must retain their coverage status. Keep raw journals and the original code checkout with any exported report.

## Resource accounting

The parent process samples the worker process tree every 0.2 seconds. A worker is stopped above 10 GiB sampled RSS or after an eight-hour attempt. At least 6 GiB must be available before a model run. These checks cannot guarantee that a brief allocation peak is captured. A sleeping PC or interrupted process makes elapsed timings unsuitable for comparison.

`generation_seconds` measures the model generation/decode operation per question; it excludes checkpoint loading and most preprocessing. Worker elapsed time includes loading, image checks, generation and scoring. Resource records are per attempt. Resume does not convert the latest attempt into a cumulative peak; `resource_history.jsonl` retains prior attempts. Ollama is a separate service and is outside this worker's measurement scope.

A measured RSS below 95% of the loaded parameter storage is flagged as implausible. Generation coverage and resource validity are separate results. Such a run may verify inference but cannot support a validated peak-RAM claim. The Linux development environment showed this limitation; the target Windows PC requires its own measurements.

## Remaining research stages

- Repair and admission: semantic reasoning boundaries, answer-blind hints, continuation, correctness and integrity checks.
- Adaptation: eligible training questions only, independent feedback, saved configuration and checkpoint provenance.
- Final review: freeze prompts, limits, scorer and comparison scope; run both final test sets once the implementation is ready.
- Report: baseline/adapted accuracy and changes in percentage points, per-type analysis, format/error rates, measured costs, illustrative errors and explicit deviations from the paper.

Feedback results may guide development. Final test results must not guide prompts, repairs, training selection or thresholds. This phase reserves the test sets; it does not run the final experiments.

## Exploratory answer-format diagnostic

The separately labelled `compare-prompts` command reads two complete five-question feedback runs, retains their evidence and generates the same questions with a concise instruction. The default protocol, data manifests, primary model and scorer remain unchanged. Only the instruction and identifying trial labels change. Controls, original file hashes and both outcomes are recorded. The new instruction was proposed after feedback inspection; this is exploratory development and is not a test-set result. See `prompt_experiment.md` for the exact prompts and command. The concise instruction is not automatically adopted for final evaluation or reasoning-trajectory training.

## Exploratory whole-answer scoring

A separate `diagnose-answers` command reads a completed feedback prompt comparison and scores the same raw responses with `plain-whole-answer-em-v1`. It lowercases complete strings, collapses whitespace and removes terminal `.?!`; it performs no extraction, synonym matching or numeric conversion. Original boxed scores remain unchanged. No new inference, training or final-test evaluation occurs. The diagnostic is exploratory and is not automatically adopted by the normal evaluator. See `answer_diagnostic.md` for exact ordering, integrity checks and interpretation.

## Training-only repair feasibility

`repair-pilot` selects three eligible training questions and records original-instruction rollouts, raw-logit entropy proposals, a gold-blind local audit and same-checkpoint assistant-prefill continuation where justified. A separate short answer readout is used consistently for initial/candidate training verification. That readout is an explicit adaptation, not a change to the normal evaluator. The pilot performs no medical SFT or target export; full learnability admission and feedback navigation remain open. See `repair_pilot.md` for the exact information flow, budgets, semantic limitations and preliminary outcomes.
