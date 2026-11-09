# Qwen3.5 fixed feedback baseline

This approved experiment runs ten fixed VQA-RAD feedback questions and ten fixed English SLAKE feedback questions through the already installed local `qwen3.5:9b` model. It performs no repair, training, final-test generation, package installation or model download.

## Run on Windows

Use the existing Python 3.12 interpreter. Keep Ollama running and the PC awake. If another Ollama model is occupying memory, close that workload first.

```powershell
& '.\.venv312\Scripts\python.exe' '.\evaluate_feedback.py'
```

The default data folder is `data/benchmarks`. Use `--data-dir` only if the prepared bundle is stored elsewhere. The program writes `outputs/qwen35_feedback_20_v1/` and refuses to overwrite any previous attempt. Return its printed summary before planning another experiment. A failure is evidence to inspect, not an instruction to delete the folder or retry.

## Fixed protocol

| Control | Setting |
|---|---|
| Model | Installed `qwen3.5:9b`; record digest and full `/api/show` metadata |
| Data role | Feedback only; ten questions per dataset |
| Selection | Existing SHA-256 ID ordering with seed 7; never substitute train/test items |
| Instruction | Appendix D.1 of the supplied `MedEvo-Ant2026.pdf`, page 29 |
| Thinking | Explicit `think: false`; metadata must confirm support |
| Sampling | Temperature 0, seed 7, top-k 20, top-p 0.95, presence penalty 1.5 |
| Output | At most 512 generated tokens per question |
| Context | 4096 tokens |
| Device request | CPU: `num_gpu: 0`; four threads |
| Image | Clean RGB PNG, longest edge at most 512 before Ollama preprocessing |
| Endpoint | Existing loopback-only native Ollama HTTP transport |
| Workload | At most twenty generation requests, no retries |
| Time budget | 300 seconds per request, within a total 1800-second client budget |

The exact instruction appended to each question is:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

Reference answers, answer-type labels, dataset identifiers and earlier responses are not included in the generation payload. References appear only in local selection/scoring records. Internal Ollama image processing and rendering are backend behavior, not a reproduction of the authors' execution backend.

The prepared bundle must match the existing protocol, source manifests, record counts and image-separation policy. Dataset files, selected image bytes and source fingerprints are checked for changes. Model digest and Ollama version are recorded and checked again before reporting. An unexpected thinking field or visible `<think>` tag stops the run after preserving the raw reply.

These are deterministic feedback subsets, not random samples newly drawn for statistical inference. Their first five questions per dataset overlap previously inspected feedback examples under the existing selection policy. Do not treat results as a blind final benchmark, compare their differences directly to the old five-question scores, or claim MedEvo improvement.

Temperature zero, thinking disabled, quantized CPU execution, context/image limits and the 512-token ceiling are declared study settings. Only the shared instruction is verified against Appendix D.1. These settings must not be described as an exact reproduction of the paper's full evaluation setup.

## Resource and failure behavior

The preflight requires at least 10 GiB available system RAM; a between-request check stops if available RAM falls below 4 GiB. These guards are neither a fit guarantee nor continuous memory enforcement during a request. No peak Ollama RSS is measured. `/api/ps` is saved separately as allocation metadata; allocation is not peak process memory.

The client uses a thirty-minute batch budget and at most five minutes for an individual HTTP request. This is a bound, not a runtime forecast. An HTTP timeout may leave server work running; inspect Ollama before any further attempt. Data-integrity preparation and metadata preflight are outside the generation budget.

On transport/service failure, invalid response, unexpected thinking or changed identity, stop without another generation. Partial records remain available and complete-run accuracy is unset. Generated-only metrics remain explicitly partial diagnostics. Empty answers and unboxed replies are recorded and scored wrong by the unchanged parser. Token-limited replies are retained with flags, including any answer box that can still be parsed; no retry or larger budget is applied automatically.

## Outputs and metrics

| Artifact | Contents |
|---|---|
| `run.json` | Prompt, options, budgets, selected IDs, data/code hashes, digest/version and system available-RAM observations |
| `environment.json`, `model_details.json` | Installed environment and server model metadata |
| `selection.json` | Fixed local evaluation records, including references for scoring |
| `cases/01/` through `cases/20/` | Exact reference-free requests and raw responses |
| `predictions.jsonl` | Flushed per-question outputs, stop reasons, token counts and request timings |
| `scored.jsonl` | Parsed answers, references, exact-match and token-F1 scores |
| `summary.json`, `report.md` | Dataset/overall metrics, lengths, latency, failures, coverage and limitations |
| `server_allocation.json` | Final server allocation metadata, when available |

Report per-dataset and overall strict/normalized exact match, mean token F1, boxed-answer rate, coverage, parsing failures and truncations. Also record mean/median/min/max client seconds per generated question, generated tokens, answer characters and whitespace-separated word counts. The client latency includes model loading, image processing and generation; provider load/generation timing is separately retained per prediction. Failed calls are timed in their records; they are not folded into mean successful-question latency.

Raw requests include encoded benchmark images. Keep these output folders private and excluded from Git unless data rights and contents have been reviewed. The repository contains no newly generated medical outputs or model weights.

## Preparation verification

The complete Linux development test suite passed: 186 tests in 23.958 seconds. This is software verification, not a Windows Qwen3.5 inference result.

Eight controlled-transport tests exercise all twenty requests, reference exclusion, fixed feedback-only selection, earlier-file preservation, incomplete coverage, no retries, thinking rejection, identity changes, empty/unboxed/truncated answers, deadline handling and preflight guards. They do not call a real Ollama service or establish Qwen3.5 medical accuracy. Actual twenty-question outputs remain pending on the Windows PC.

The earlier tiny Windows NF4/LoRA result is feasibility evidence only. This experiment performs zero optimizer steps and does not complete medical repair admission or the final professor report.
