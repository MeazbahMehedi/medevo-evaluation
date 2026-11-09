# Changelog

All notable changes to this project are documented in this file.

## [0.5.0] - 2026-11-09

Phase 5: adapted-model evaluation.

### Added
- Qwen3.5-2B with the LoRA adapter evaluated on the 400 reserved test questions: primary metric 21.5% to
  50.0% (+28.5 points, McNemar p < 0.001); format-independent secondary metric 47.0% to 50.0%
  (+3.0 points, 95% CI [-1.0, 6.8], p = 0.17). Boxed `1`/`0` answers fell from 190 to 10.
- Compiled results for all systems with paired bootstrap intervals and McNemar tests (`results/compiled/`).

## [0.4.0] - 2026-11-09

Phase 4: LoRA adaptation.

### Added
- LoRA adapter for Qwen3.5-2B trained on the 43-example Phase 3 bank (rank 16, 2 epochs, 11 optimizer
  steps, 0.75% of parameters trainable); mean loss fell from 0.421 to 0.259.
- Training settings, per-example loss log and adapter configuration in `results/adaptation_v1/`.

## [0.3.0] - 2026-11-09

Phase 3: frozen protocol, test baselines and training-bank construction.

### Added
- `medevo_study` package and entry points: `freeze_protocol.py`, `prepare_primary_model.py`,
  `run_test_evaluation.py`, `build_training_bank.py`, `reaudit_failures.py`, `train_adapter.py` and
  `compile_results.py`.
- Frozen protocol fixed before any test generation: paper prompt, 1024-token output budget, primary metric
  and 200 stratified test questions per dataset; amendment 01 adds a secondary metric that reads boxed
  `1`/`0` as yes/no.
- Test baselines on the 400 reserved questions: Qwen3.5-9B 57.0%, Qwen3.5-2B 21.5% (47.0% on the
  secondary metric).
- Training bank v1 from 100 training questions: 40 verified correct rollouts, 0 repairs.
- Re-audit (v2) aligned with the MedEvo audit prompt, continuation budget and learnability gates:
  repairs proposed for 58 of 60 failures, 3 admitted; final training bank of 43 examples.
- Feedback and auditor experiments from the earlier part of this phase, with notes in `docs/experiments`
  and verification records in `docs/verification`.

## [0.2.0] - 2026-11-08

Phase 2: benchmark preparation and evaluation infrastructure.

### Added
- `evaluate.py prepare` downloads pinned VQA-RAD and SLAKE sources, verifies file hashes, exports images
  and writes training, feedback and test manifests with exact-image separation and a recorded exclusion list.
- `evaluate.py inspect`, `run` and `score`: manifest validation, resumable baseline inference with a
  per-question prediction journal, and boxed-answer scoring (strict and normalized exact match, token F1).
- Split manifests in `data/benchmarks/` with source attribution and license notes.
- Exploratory modules for prompt, answer-readout and audit diagnostics and a training-only repair pilot.

## [0.1.0] - 2026-11-07

Phase 1: environment and CPU feasibility.

### Added
- `run_pilot.py` with four separately invoked checks: environment inventory, synthetic image inference,
  one synthetic LoRA update with adapter save and reload, and auditor connectivity.
- Process supervision with RAM guards, time limits and per-run output records (`src/runtime.py`).
- Primary-model wrapper for SmolVLM2-500M on CPU (`src/model.py`) and an OpenAI-compatible auditor
  client with a JSON response schema (`src/auditor.py`).
- Windows and PyCharm setup guide and feasibility notes.
