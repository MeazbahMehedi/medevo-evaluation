# Changelog

All notable changes to this project are documented in this file.

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
