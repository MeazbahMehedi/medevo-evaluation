# MedEvo Evaluation Study

![MedEvo Evaluation Study](docs/assets/banner.jpg)

An independent, reduced-scale evaluation of
[MedEvo: Medical Reasoning Evolution through Knowledge Navigation](https://arxiv.org/abs/2610.17690)
on the VQA-RAD and SLAKE medical visual question answering benchmarks, run on a single CPU workstation.

**Status:** Phase 1 (environment and CPU feasibility) complete.

## Phase 1: environment and CPU feasibility

Before building the evaluation pipeline, this phase checks that the required model operations work on a
CPU-only Windows workstation:

| Check | Command | What it verifies |
|---|---|---|
| Environment | `python run_pilot.py` | Interpreter, packages, RAM, disk and CPU inventory |
| Inference | `python run_pilot.py inference` | The pinned primary model answers a synthetic image question |
| Training | `python run_pilot.py train` | One LoRA update changes only adapter weights; the adapter saves and reloads with matching outputs |
| Auditor | `python run_pilot.py auditor` | A local image-capable auditor returns a schema-valid JSON audit |

Each check writes its own folder under `outputs/` with settings, environment, result and resource records.
These are software checks only; they say nothing about medical accuracy.

## Setup

Python 3.12 on Windows, CPU only:

```powershell
py -V:3.12 -m venv .venv312
$py = '.\.venv312\Scripts\python.exe'
& $py -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt
& $py -m unittest discover -s tests -v
```

See [docs/windows_setup.md](docs/windows_setup.md) and [docs/experiments/feasibility.md](docs/experiments/feasibility.md).

## License

[MIT](LICENSE).
