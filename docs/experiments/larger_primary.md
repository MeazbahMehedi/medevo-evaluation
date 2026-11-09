# Larger primary-model runtime check

This test uses the already installed `qwen2.5vl:7b` through the local Ollama service as an answering model. Earlier checks used it as an auditor. A valid auditor schema is not required for this experiment.

## Approved concise-prompt experiment

The reported 512-token Windows run completed in 67.771 seconds but again stopped at the output limit. It produced general descriptions and possible causes without a finished final answer. The 128-token baseline was preserved. This is a negative output-budget result; it does not establish medical accuracy or a successful repair.

The approved next request changes only the instruction text, retaining the same question, image, model digest, server version, CPU options and 512-token budget. Its instruction is:

> Give at most three brief, image-grounded reasoning steps. Then provide a short final answer inside `\boxed{}`.

With the updated project files, run:

```powershell
& '.\.venv312\Scripts\python.exe' '.\check_primary.py' --collection-dir '.\outputs\20261108T061937Z_training_collection_9120a6' --max-new-tokens 512 --baseline-dir '.\outputs\larger_primary_512_v1' --concise-prompt
```

The baseline must be the completed original-prompt 512-token run. A different image, prompt, generation setting, model digest or server version is rejected before any new generation. The new request retains the original question and replaces only the appended instruction. It includes no reference answer, auditor hint or previous model response.

The output folder is `outputs/larger_primary_concise_v1/`. Five source baseline records are copied into `baseline_512/`, and the original baseline tree is checked before and after the call. The earlier output folders are not overwritten. Exactly one new generation request is permitted in the new output folder; no longer-budget or alternate-prompt retry is automatic.

The raw response and existing boxed single-case verification are saved for review. Finishing, following the requested format and matching the reference are separate questions; none by itself validates the reasoning. This is an exploratory change from the paper's answering instruction, not an adopted benchmark protocol. It provides no benchmark-wide accuracy, training example admission, optimizer update or evidence that the full MedEvo method works.

## Approved 512-token extension

The reported 128-token Windows run completed on CPU in 32.478 seconds, returned exactly 128 tokens and stopped with `done_reason: length`. It did not produce a finished answer. The reported 5.278 GiB server allocation is not a measured peak RAM value.

The approved follow-up changes only the saved request's `options.num_predict` from 128 to 512. With the updated project files, run:

```powershell
& '.\.venv312\Scripts\python.exe' '.\check_primary.py' --collection-dir '.\outputs\20261108T061937Z_training_collection_9120a6' --max-new-tokens 512 --baseline-dir '.\outputs\larger_primary_v1'
```

The command validates the original request, truncated baseline reply, training-case identity, collection hashes, model digest and Ollama version. It retains the same image, question, original instruction, context size, CPU settings, seed, temperature and client timeout. It sends exactly one new generation request. No new model is downloaded or substituted.

Results go into `outputs/larger_primary_512_v1/`. Five original baseline records are copied into `baseline_128/`; the source baseline and training collection are checked for changes before and after the call. Neither output folder is overwritten. The 512-token result remains a separate exploratory training-case diagnostic and is not adopted as a new benchmark protocol. A further truncation is recorded without a longer automatic retry.

Compare the raw responses, finish reasons and existing boxed single-case verification. An answer that finishes can still be wrong. Timings include model loading, image processing and generation; warm caches can affect them, so total elapsed differences cannot be attributed solely to the token budget. Adapter training and the rest of MedEvo remain untested by this extension.

## Original 128-token command

```powershell
& '.\.venv312\Scripts\python.exe' '.\check_primary.py' --collection-dir '.\outputs\20261108T061937Z_training_collection_9120a6'
```

No new package or model download is initiated by this command. It confirms the installed model tag and image capability, records its digest and server version, and sends one generation request. If the model is missing, it stops rather than downloading it automatically.

The selected training question and original reasoning instruction match the saved small-model case. The input PNG is the exact saved auditor view, checked by its hash. The larger model receives only that image, the question and the original instruction; it receives no failed reasoning, hint, auditor evidence or reference answer. The reference is read after generation for a separate single-case check under the unchanged boxed scorer.

The request explicitly sets CPU offloading to zero GPU layers, four threads, a 4096-token context, temperature zero, seed seven and a maximum of 128 new tokens. It uses the native `/api/chat` vision interface. The client timeout is ten minutes. There is one generation request and no retry, readout, continuation, adapter training or automatic adoption of this model. A client timeout does not guarantee immediate cancellation inside the external server; inspect a failure before another request.

The chosen preflight guard requires eight GiB of available RAM. This is a conservative project guard, not a guarantee of model fit or an official model memory requirement. The command does not impose a hard memory cap on the external Ollama service.

Results are written to `outputs/larger_primary_v1/`. The directory cannot be overwritten by rerunning the command. `request.json`, `response.json`, model metadata, `single_case_verification.json` and `summary.json` preserve the evidence. Original collection files are not modified.

`model_inference_completed` means a completed server reply was received. `finished_answer_available` additionally requires nonempty text without a detected token-limit stop. Neither field establishes medical correctness. A one-case exact match is not benchmark accuracy. This inference-only test does not establish that adapter training or the full MedEvo method fits the same hardware.

The summary reports client elapsed time, provider timings, system available RAM before/after and any server-reported model allocation/VRAM. These are distinct measurements. Server allocation is not a measured peak process RSS; `peak_ollama_ram_gib` remains null. The earlier Python-worker peak cannot be compared directly with this external server allocation.

This is an exploratory change of both model and backend, including quantization, template and internal image processing. It is not a controlled same-backend model-size comparison and cannot be pooled with existing baseline scores.

Official interface references checked for preparation:

- [Qwen2.5-VL 7B model entry](https://ollama.com/library/qwen2.5vl:7b)
- [Vision requests](https://docs.ollama.com/capabilities/vision)
- [Chat endpoint](https://docs.ollama.com/api/chat)
- [Loaded-model metadata](https://docs.ollama.com/api/ps)
- [Runtime option definitions](https://github.com/ollama/ollama/blob/main/api/types.go)

Software tests use a scripted local transport. The 128- and 512-token outcomes above are the Windows results. Live inference with the concise instruction must still be measured on that PC; no completed answer or medical improvement is claimed in this version.
