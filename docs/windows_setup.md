# Windows and PyCharm setup

The study runs on Windows 11 with Python 3.12, CPU only. These steps set up a fresh environment from a
clone of this repository.

## 1. Interpreter and dependencies

```powershell
py -V:3.12 -m venv .venv312
$py = '.\.venv312\Scripts\python.exe'
& $py -m pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
& $py -m pip install -r requirements.txt
& $py -m pip check
& $py -m unittest discover -s tests -v
```

Install the CPU builds of PyTorch first, then the pinned requirements. If another Python version is the
system default, call the virtual environment's interpreter explicitly as above.

In PyCharm, open the repository folder, then **Settings > Python > Interpreter > Add interpreter >
Existing** and select `.venv312\Scripts\python.exe`. Use the repository root as the working directory
for terminal commands and run configurations.

## 2. Benchmark data

Either download the prepared images (see [data/README.md](../data/README.md)) or rebuild the data from
the pinned sources:

```powershell
& $py evaluate.py prepare
& $py evaluate.py inspect
```

Preparation downloads six pinned files (251,079,630 bytes), verifies their hashes, then exports images
and manifests. A Hugging Face symlink warning on Windows does not indicate a failure. No token is needed.

| Dataset | Train | Feedback | Test |
|---|---:|---:|---:|
| VQA-RAD | 655 | 79 | 451 |
| SLAKE, English | 4,911 | 1,053 | 1,061 |

Preparation never overwrites an existing data folder. To build a second copy, pass a different
destination before the command, and use the same option for later commands:

```powershell
& $py evaluate.py --data-dir data\second_copy prepare
```

## 3. Small feedback runs

```powershell
& $py evaluate.py run --benchmark vqa_rad --limit 5
& $py evaluate.py run --benchmark slake --limit 5
```

Each run writes a folder under `outputs/` with `metrics.json`, `predictions.jsonl` and `resources.json`.
Feedback runs are development checks, not final benchmark results.

Keep the PC awake during runs: sleep interrupts generation and invalidates timing records. The evaluation
worker enforces a sampled memory ceiling and an attempt time limit as safeguards.

## 4. Ollama

Qwen3.5-9B (baseline and auditor) is served by [Ollama](https://ollama.com):

```powershell
ollama pull qwen3.5:9b
```

Keep the Ollama service running for baseline evaluation and Phase 3 collection. The frozen protocol records
the model digest, and runs stop if the installed model differs.

## Recovering a stopped run

Re-run the same command. The frozen-protocol scripts resume automatically. For `evaluate.py`, pass the
run folder:

```powershell
& $py evaluate.py run --benchmark vqa_rad --limit 5 --resume 'outputs\RUN_FOLDER'
```

Completed records are kept. A changed configuration, changed code or a corrupt record stops the run. If a
force-stopped worker leaves `run.lock`, confirm in Task Manager that the process has ended before removing
the lock.

Reference: [PyCharm interpreter configuration](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html).
