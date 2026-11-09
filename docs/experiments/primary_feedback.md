# Fixed 7B feedback evaluation

This experiment evaluates the frozen concise-prompt 7B configuration on ten feedback questions: the same five VQA-RAD and five SLAKE questions used in the earlier small-model feedback runs. It is an exploratory baseline measurement, not a final test evaluation or a MedEvo improvement experiment.

## Windows command

Keep Ollama running. From the project root, run:

```powershell
& '.\.venv312\Scripts\python.exe' '.\evaluate_primary.py' `
  --profile-dir '.\outputs\larger_primary_concise_v1' `
  --vqa-rad-run '.\outputs\20261108T023429Z_vqa_rad_feedback_119aa2' `
  --slake-run '.\outputs\20261108T023954Z_slake_feedback_a5e572'
```

The default prepared-data directory is `data/benchmarks`. Use `--data-dir` only if the existing bundle is stored elsewhere. The command initiates no package or model download. It requires the already installed `qwen2.5vl:7b` model and the completed concise-prompt training-case run as its configuration record.

## Frozen controls

The concise instruction is:

> Give at most three brief, image-grounded reasoning steps. Then provide a short final answer inside `\boxed{}`.

Each native Ollama request contains just one selected image, its question and this instruction. It receives no reference answer, answer-type label, failed reasoning, repair hint or prior response. Temperature is zero, seed seven, context size 4096, CPU thread count four, GPU layers zero and the output budget 512 new tokens. The image is converted to a clean RGB PNG with a maximum edge of 512 pixels before Ollama's own internal image processing.

The script validates the completed concise profile, its prompt/options, model digest and Ollama version. The prepared bundle is checked for its manifest hashes, image identities, data roles and existing exact-image separation rules. Each old baseline must have the expected five feedback IDs under the original deterministic selection. The original question order is retained. Train/test questions are not substituted, and the held-out test set receives no generation request.

The original source distributions, revisions, bundle hashes and selection records are retained in `run.json` and copied baseline evidence. Existing limitations, including lack of a patient-ID or near-duplicate separation guarantee, still apply.

## Workload and failures

There are at most ten generation requests, one per selected question. Each has a client timeout of at most ten minutes; a total thirty-minute client budget bounds the batch. A service/transport failure stops further generation. There is no automatic retry, replacement question, prompt adjustment or larger token budget. A client timeout may leave work running inside Ollama; review partial output before making another attempt.

Empty, malformed-format and token-limited replies remain recorded. A valid completed server reply can still have no usable answer. Truncation does not trigger a retry or an alternative parser.

Results are written to `outputs/primary_feedback_10_v1/`. Existing output is not overwritten. On interruption or failure, any recorded predictions remain available and the summary marks the run incomplete. Complete-subset accuracy fields are not reported for an incomplete or invalid batch; separately labelled generated-only values must not be substituted for them.

## Metrics

The existing `boxed-strict-em-v1` scorer is reused without modification:

- Strict exact match for the parsed boxed answer.
- Normalized exact match and mean token F1 as supplementary metrics.
- Coverage, missing/error counts, box parse failures and token-limit counts.
- Format success: the percentage of selected questions with a parseable boxed answer, when the batch is complete.
- Per-dataset and open/closed-answer breakdowns, with null values for empty groups.
- Per-request client time, aggregate request time and mean time for generated replies.

Question-level correctness is based on the existing reference string; no model or human rewrites answers to improve their scores. The scorer's hash is checked against the saved baselines and recorded. A matching string is not independent medical validation. A truncated response is scored by the existing parser and separately flagged as truncated; the scoring rule is not changed after generation.

The script records available system RAM before/after, loaded-model metadata when available and client/provider timings. It does not measure peak external Ollama process RSS: `peak_ollama_ram_gib` stays null. Server allocation is not a peak RAM measurement.

## Files to review

- `summary.json`: overall, per-dataset and answer-type metrics, execution status and preservation checks.
- `report.md`: compact result table and interpretation limits.
- `scored.jsonl`: questions, raw responses, references, parsed answers and question-level scores.
- `predictions.jsonl`: flushed generation/error records, preserving partial work.
- `cases/01` through `cases/10`: exact requests, native responses and attempt records.
- `run.json` and `environment.json`: model/configuration identity, versions, data/source hashes, selected IDs and timing.
- `profile/`, `selection_vqa_rad/`, `selection_slake/`: copied source records; originals remain unchanged.

## Interpretation

Five reused feedback questions per dataset cannot establish full benchmark performance. They were previously inspected during development, and the concise instruction was chosen after other training-case trials. This is not a blind held-out test or a controlled comparison of model size alone: the model, backend, quantization, prompt and output budget differ from the original small-model runs. Do not pool these results with earlier baseline scores or label their differences as MedEvo gains.

No auditor call, candidate admission, adapter training or final-test inference is performed. MedEvo adaptation and training feasibility remain unfinished. The command's software checks use small controlled images and a scripted service; actual ten-question model outputs remain to be collected on the Windows PC.
