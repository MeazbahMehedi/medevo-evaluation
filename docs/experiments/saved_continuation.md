# Saved-audit continuation test

This command runs the approved single-case continuation test for `vqa_rad:train:000188`. It uses the existing twelve-case training collection and the sealed independent audit shipped in `configs/continuation_case/`. There is no new auditor call.

## Run on Windows

None of those directories is included in this update.

In the PyCharm terminal, with the repository root as the working directory, run:

```powershell
& '.\.venv312\Scripts\python.exe' '.\continue_case.py' --collection-dir '.\outputs\20261108T061937Z_training_collection_9120a6'
```

The explicit interpreter path avoids the old Python 3.14 environment. If running through a PyCharm configuration instead, select `.venv312\Scripts\python.exe`, choose `continue_case.py`, set the same `--collection-dir` argument and use the project root as the working directory.

The test does not require Ollama to be running. Keep the PC awake. It reuses the cached primary model; a missing cache can still trigger a Hugging Face download through the existing model loader.

## What is fixed

The selected question is “why is the left hemithorax more lucent?” The exact preserved prefix is:

> The left hemithorax is more lucent because it is less dense than the right hemithorax.

The stored prefix also retains its original leading space. The approved hint is:

> Check the relationship between air content and X-ray attenuation before attributing the side-to-side lucency to lung aeration.

The command verifies the sealed audit, its original input, the saved question and reasoning, the image and training role, the source implementations, model settings and token limits before generation. It preserves the original records. The primary model receives the original image and question, original reasoning instruction, exact prefix and the recorded recovery cue/hint. It never receives the reference answer, auditor rationale or parent assessment.

The unchanged `continue_reasoning` function uses the same pinned primary checkpoint, single 512-pixel view, CPU float32 and greedy decoding. Its existing assistant-prefill checks verify that the template and tokenized input retain the supplied prefix. The continuation budget is 128 new tokens. The original multimodal answer-readout function and its 32-token budget remain unchanged; the exploratory text-only readout is not substituted.

## Output and interpretation

Results are written to `outputs/continuation_case_v1/`. The command refuses to overwrite that directory or repeat an attempt in it. Do not delete it to force a retry; review failures first.

Key files are:

- `summary.json`: outcome, call counts and the supervisor's resource measurements.
- `continuation_request.json`: the reference-free generation inputs.
- `candidate_reasoning.json`: unchanged continuation, prefix verification and complete response, if generation returns.
- `candidate_readout.json` and `verification.json`: readout and training-only lexical comparison, if the continuation passes the integrity checks.
- `run.json`, `evidence/` and `approved/`: settings, hashes and the original records.
- `console.log`, `resources.json` and any error file: execution evidence.

Changed prefixes, empty or truncated suffixes, an unchanged erroneous suffix and repetition are rejected without a longer retry. A qualifying suffix receives at most one answer readout. Reference answers are used for offline hint validation and answer checking; they are never included in generation messages.

`status: complete` means the diagnostic finished, including an ordinary rejection or answer mismatch. It does not mean the repair worked. `answer_match_pending_review` means only that the unchanged training-only whole-answer verifier matched the reference. Readout fidelity, reasoning quality, independent candidate review and full learnability admission would still be required. No benchmark accuracy is produced.

This command exports no training targets, admits no examples, performs no optimizer steps and creates no adapter. Do not infer model improvement from this one training case.

## Validation available before delivery

The development checks use controlled callbacks to exercise prefix integrity, truncation, answer mismatch, no retry, reference exclusion from callback arguments, sealed-input tampering and the separation between an answer match and training admission. The full project suite is also checked after the update. These software checks do not run the pretrained model or establish medical validity. The actual same-model continuation remains to be executed on the Windows PC.
