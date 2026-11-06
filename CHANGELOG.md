# Changelog

All notable changes to this project are documented in this file.

## [0.1.0] - 2026-11-07

Phase 1: environment and CPU feasibility.

### Added
- `run_pilot.py` with four separately invoked checks: environment inventory, synthetic image inference,
  one synthetic LoRA update with adapter save and reload, and auditor connectivity.
- Process supervision with RAM guards, time limits and per-run output records (`src/runtime.py`).
- Primary-model wrapper for SmolVLM2-500M on CPU (`src/model.py`) and an OpenAI-compatible auditor
  client with a JSON response schema (`src/auditor.py`).
- Windows and PyCharm setup guide and feasibility notes.
