# One-case auditor-model comparison

Next experiment: downloading `qwen2.5vl:7b` and comparing it with the saved 3B schema-grounded audit on the same first training case. The larger model is a hypothesis to test, not a medically validated auditor or a guaranteed remedy.

The official [Ollama 7B listing](https://ollama.com/library/qwen2.5vl:7b) identifies an approximately 6.0 GB Q4_K_M package with 8.29B parameters. Download size is not working RAM. The existing 32 GB PC is a plausible test target, but actual memory demand and CPU latency need target-PC observation.

## Download and install

The approved download command is:

```powershell
ollama pull qwen2.5vl:7b
```

Wait for `success`. Keep the 3B model installed. No Python package or core configuration update is required.

Before inference, unload the 3B model from RAM without deleting its weights:

```powershell
ollama stop qwen2.5vl:3b
```

Retain the local model IDs from `ollama list` with the experimental record. Tags alone do not prove immutable weight identity.

## Run once

From the repository root, run this approved one-call comparison:

```powershell
& '.\.venv312\Scripts\python.exe' '.\check_auditor.py' --collection-dir '.\outputs\20261108T061937Z_training_collection_9120a6' --auditor-model 'qwen2.5vl:7b' --baseline-check-dir '.\outputs\20261108T073540Z_single_audit_00a8a9'
```

Both explicit flags are required. No other replacement model is accepted. The command validates the completed 3B schema-grounded baseline and compares its captured request with the frozen request reconstructed from the original collection. The new network body changes **only `model`**. The original audit instruction, appended schema, question, image, reasoning, steps, entropy, JSON schema, temperature, seed, token cap, endpoint, timeout and validation rules stay fixed.

The existing `configs/repair_pilot.json` stays on the original 3B model for historical consistency; the explicit replacement applies only to this call. Do not edit that configuration to bypass the source checks.

The same first training ID is used, with no new question selection. The command makes at most one new audit call, performs no primary-model generation or continuation, and exports no training examples. It does not automatically retry, fall back to 3B, pull missing weights or extend a timeout. Keep the PC awake.

The unchanged 180-second HTTP timeout includes any cold model loading and generation. A timeout establishes failure within that budget, not poor medical reasoning. The historical 3B call and new 7B call may differ in model-cache state, so their elapsed times are not a controlled speed benchmark. No total Ollama peak-RAM claim is produced.

## Output and interpretation

The comparison creates a separate `outputs\..._auditor_model_comparison_...` folder. It contains the original collection evidence, a byte-preserved `baseline_3b/` copy of the previous check metadata/request/response, and a new `trial/` request/response. `run.json` records the source and effective configurations, baseline hashes and the changed `model` field. No original collection or baseline file is modified.

In `summary.json`, `original_condition` is `schema-grounded 3B`, `original` is that captured response, and `trial` is the new 7B response. This is distinct from the older pre-grounding audit, which remains in the copied collection evidence.

A complete response can still be invalid or incoherent. A valid skip/unresolved decision is preserved and does not create a repair opportunity. A repair passing structural/hint checks still needs clinical and boundary review before any separately approved continuation or admission work. Nothing is admitted automatically.

Review the single response before further calls. If it remains unusable, reassess the serving/runtime or compute approach instead of adding local prompt variants. Even a favorable result on this one previously inspected training case does not establish generalization, benchmark performance, training feasibility or completion of the professor's assignment.

## Development checks

Five additional tests cover paired-option enforcement, permitted model identity, exact model-only request difference, preservation of both baselines, rejection of changed baseline controls and failure without fallback. A real local HTTP request to a scripted server verifies that the replacement route sends exactly one request naming the 7B model. Scripted replies are not live Qwen medical results. Current full-suite results are in [test_results.txt](../verification/test_results.txt).
