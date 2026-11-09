# Training-only repair pilot

The pilot exercises reasoning generation, entropy-guided audit, supported-prefix continuation and preliminary candidate checks on three eligible training questions. It prepares evidence for the next implementation decision. Adapter training and final benchmark evaluation are separate stages.

## Update and first checkpoint

No dependency or dataset download is added.

From the repository root, run the expanded software suite:

```powershell
& '.\.venv312\Scripts\python.exe' -m unittest discover -s tests -v
```

Expected: 86 tests pass. Review the test result before the live pilot.

## Run the three cases

Keep the Ollama desktop service running and the PC awake. The previously downloaded `qwen2.5vl:3b` model is used at `http://127.0.0.1:11434/v1`. The pilot uses the explicit local configuration in `configs/repair_pilot.json`, so it does not require resetting endpoint/model environment variables in each terminal. If `AUDITOR_API_KEY` is present, it is used only as a request header; it is never copied into run files. The standard local Ollama endpoint normally needs no secret.

```powershell
& '.\.venv312\Scripts\python.exe' '.\evaluate.py' repair-pilot
```

The command runs two VQA-RAD and one SLAKE training questions in deterministic seed-7 order. With the checked data bundle, the IDs are:

- `vqa_rad:train:000201`
- `vqa_rad:train:001467`
- `slake:train:4022`

Selection uses the eligible `train` manifest. Its exact-image exclusion policy separates it from feedback and test images. The pilot stops on an infrastructure failure and retains the available evidence. A skipped, unresolved or rejected repair is a recorded research outcome, not an instruction to force a successful answer.

## Method

1. **Initial reasoning.** The pinned pretrained primary model receives the image, question and the original shared reasoning instruction from the prepared protocol. Generation is greedy and limited to 128 new tokens.
2. **Entropy proposal.** Raw generation logits produce per-token entropy in nats. Visible tokens are assigned to heuristic sentence spans by their last decoded character. The step score combines mean entropy and the linearly interpolated 90th percentile with equal weights. A surge is measured against up to three preceding steps. Min–max normalized score plus surge selects the earliest maximum and a window of one neighbouring step on each side. Constant profiles normalize to zero. An unalignable or invalid profile is recorded as unavailable with its reason.
3. **Training-only answer readout.** The same primary model receives its reasoning in conversation context and generates a short final answer, capped at 32 tokens. The readout sees the image and question, but never the reference. Its complete response is checked with the existing whole-answer normalization. This extra readout is a CPU-study adaptation; it is distinct from the paper's final-answer parser and does not change normal evaluation.
4. **Gold-blind audit.** If the initial readout fails verification, the local auditor receives the image, original question, reasoning spans and entropy proposal. It receives no reference, reference-derived answer type, source filename, hidden labels or verification text. It can repair, skip a format-only/non-error case, or return unresolved. A repair identifies the earliest consequential step, a four-family diagnosis, boundary evidence and a short hint. Entropy is a proposal, not proof of an error.
5. **Prefix continuation.** The first-error step maps to a complete-span boundary. The preceding source text is retained verbatim. Only that prefix and a natural recovery cue containing the hint prefill an unfinished assistant message; structured audit fields remain metadata. The same checkpoint generates one suffix, capped at 128 tokens. Rendered text, tokenized prefill and returned input-token prefix must agree. A first-step error preserves an empty prefix and is labelled `full_regeneration`.
6. **Preliminary checks.** The pilot rejects a direct reference-term/number-word hint leak, malformed boundary, changed prefill, empty/truncated suffix, unchanged erroneous suffix or repeated sentences. A new short answer readout must match the reference under the same rule. Answer-matching candidates receive a second review by the same auditor for unsupported content, repetition and suspected hint leakage. A candidate passing these checks is labelled preliminary.

The reference is confined to the verifier and a conservative hint filter after auditing. Primary requests and auditor requests are constructed from explicit allowlists. The hint filter can overreject legitimate wording and cannot establish absence of semantic answer leakage. A model claim that the earlier prefix is supported is distinct from the mechanically verified fact that its text was preserved.

Heuristic sentence spans can split abbreviations or combine several medical operations. They are not independently validated semantic annotations. The local Qwen model replaces the paper's stronger auditor. Its clinical judgement, earliest-error decision and second-pass review remain model judgements. The second pass is not an independent clinical review.

## Limits and resources

| Setting | Default |
|---|---:|
| Training cases | 3 |
| Reasoning tokens | 128 per case |
| Short readout tokens | 32 per readout |
| Suffix attempts | 1 per failed case |
| Suffix tokens | 128 |
| Auditor response budget | 512 tokens |
| Auditor HTTP timeout | 180 seconds |
| Worker timeout | 3,600 seconds |
| Primary worker sampled-RSS ceiling | 10 GiB |
| Minimum available RAM before launch | 6 GiB |

Only loopback auditor endpoints are accepted. Redirects and environment HTTP proxies are disabled for those requests. The pilot does not configure the auditor's device placement; the service controls it. `ollama ps` reported CPU execution.

The supervisor measures the Python worker process tree. **It excludes the separate Ollama service.** Peak RSS must be marked plausible before it is used as a resource result. A cold auditor load and image processing can add time; use recorded durations rather than treating the timeout as a forecast.

## Records

Each command creates `outputs/<timestamp>_repair_pilot_<id>/`:

| Path | Contents |
|---|---|
| `run.json` | Exact selected IDs, data/model/source identities, configuration, both primary instructions and declared limitations |
| `environment.json`, `model.json` | Runtime inventory and original primary-checkpoint identity |
| `summary.json` | Recorded outcomes, call counts, preliminary candidates, resource status and zero optimizer/export counts |
| `resources.json`, `console.log` | Worker timing, sampled memory, exit/limit outcome and console trace |
| `cases/01/`, `02/`, `03/` | Per-case evidence |

A case folder records initial reasoning, token entropies, span boundaries, short answer readouts, auditor requests/responses, audit decisions, continuation inputs and outputs, verification and any candidate review. `verification.json` contains the reference and match results; it is separate from request files. `case.json` is updated as the case proceeds, including failure or rejection reasons.

Raw auditor requests include image content so they can be inspected locally. Keep these run folders out of Git and public artifacts unless data rights and contents have been reviewed. The repository contains code and sanitized preparation records, not dataset images or raw request image bodies.

`status: complete` means all selected cases reached a recorded outcome. Check `validated_audit_count`, `continuation_path_exercised` and the outcome counts to see which branches actually ran. No failed question is silently substituted, no successful repair is guaranteed, and an unresolved audit does not count as a repair. If the service fails, start the local service and review the error before a new run; a new command creates a separate pilot. No automatic resume or retries are included.

## Scope after this pilot

Preliminary candidates are retained as evidence. The pilot exports zero SFT targets, creates no adapter and performs zero optimizer steps. Full admission, including hint-free suffix learnability ratios/gaps and feedback-based calibration, is deferred. Twenty error subtypes, feedback-guided allocation, historical-target replay and medical SFT are later work. A short answer match does not prove the reasoning is medically supported.

This is an independent CPU adaptation, not the authors' released implementation or a completed reproduction. Final held-out results remain pending. The normal evaluator and its default boxed scorer are unchanged; the training-only readout does not adopt a new final protocol.

## What to send back

Review the software test summary first. After the live pilot, review the final console JSON or `summary.json`, and any error. Keep the full output folder. We will review auditor boundaries, hints, continuation evidence and rejection reasons before adding medical training.

## Saved non-repair outputs

The first Windows pilot returned one unresolved and two skip decisions with nonempty unused hints. These exposed an overstrict validator condition, corrected under policy `nonrepair-auxiliary-unused-v2`. Skip/unresolved auxiliary fields are retained for inspection and never used for continuation. Repair validation is unchanged. Use `inspect-repair` to review the existing run without modifying it; see `audit_review.md`. Zero repair decisions remain zero after this correction.
