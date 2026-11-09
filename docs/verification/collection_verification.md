# Training collection verification

Verified on 8 November 2026 using Python 3.12 in the development environment. The full suite passed **102 tests** in **7.335 seconds**; the complete log is [test_results.txt](test_results.txt).

Sixteen new checks cover deterministic six-per-dataset training selection, nine new IDs, literal question-echo classification, legacy non-repair evidence validation, unchanged model/prompt/budget/scorer controls, preserved source records, gold-blind requests, potential-candidate indexing, failed repair attempts, supported repair fixtures, infrastructure failure, process limits, lock protection and CLI routing.

An actual supervised child process exercised the hidden collection-worker CLI with a scripted primary model. It processed nine new fixture cases and reused three archived fixture cases. Other integration checks simulated valid repairs and conservative rejections through the existing pilot orchestration. Scripted positive results are test fixtures; they are not clinical results or successful live Qwen repairs.

The primary model loader, reasoning/readout/continuation implementation, local auditor prompts, boxed scorer, plain diagnostic scorer, dataset preparation and existing model/configuration files remain byte-identical to the previous release. This update adds a collection module, command routing, tests and documentation. It adds no dependency.

The collector checks archived source controls, including the documented old non-repair validator implementation, and compares each reused auditor request with a reconstruction from the saved reasoning and prepared image. Original source-pilot files are copied and hashed; reuse never changes an action into a repair. Reused records retain the original rejection in their original `case.json`, with the separate current interpretation in `collection_record.json`.

Live generation on the nine new medical questions remains a target-PC checkpoint. The current Linux verification does not measure real Qwen auditor quality, establish immutable Ollama model-tag identity, validate clinical evidence or admit examples for training. Python process-tree memory also excludes the external Ollama service. Medical adaptation, full admission checks and final baseline/adapted benchmark results remain pending.
