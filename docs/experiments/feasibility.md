# CPU feasibility checks

`run_pilot.py` retains the Phase 1 checks. They are separate from medical benchmark evaluation.

```powershell
python run_pilot.py
python run_pilot.py inference
python run_pilot.py train
python run_pilot.py auditor
```

Run one stage at a time with the selected Python 3.12 interpreter. The default is environment inventory only. Inference loads the pinned pretrained model and answers a synthetic red-circle question. Training performs one rank-4 LoRA update on the same synthetic fixture, verifies changed adapter parameters and one unchanged frozen parameter, then checks that the saved adapter reloads with matching selected logits. It is not medical adaptation and is never loaded by `evaluate.py`.

Each command writes a separate directory under `outputs/`, containing `run.json`, `settings.json`, `environment.json` and `result.json`. Model stages also record `resources.json` and `console.log`. Training saves the adapter, processor and original base-model identity. Use an adapter only when its result confirms successful reload.

The pilot requires 6 GiB available RAM, caps sampled worker RSS at 10 GiB and stops an attempt after 1,200 seconds. Its short synthetic generation cap is 16 tokens. The Phase 2 evaluation budget is separate and documented in `protocol.md`.

## Local auditor configuration

The working local configuration uses a separate image-capable model served by Ollama. Set the variables in the same PowerShell session as the command:

```powershell
$env:AUDITOR_BASE_URL = 'http://127.0.0.1:11434/v1'
$env:AUDITOR_MODEL = 'qwen2.5vl:3b'
$env:AUDITOR_API_KEY = 'ollama'
python run_pilot.py auditor
```

The `ollama` value is a dummy local key, not a paid API credential. The model must already be available to the local service; this command does not install Ollama or pull weights. In a PyCharm Run configuration, set the same variables under **Environment variables**. Variables set in one terminal do not persist in a newly opened terminal or a separate Run configuration.

An unconfigured endpoint returns `blocked`. A configured endpoint receives one synthetic image, question and deliberately wrong response, without the reference answer. The response must use the declared JSON schema and an integer character offset. Redacted request and response records are retained for inspection.

A schema pass is not medical auditor validation. In the reported Windows run, `prefix_end: 1` was structurally valid but did not represent a supported reasoning-step boundary. Repair training must wait for semantic boundary checks and medical-quality assessment in the next phase. Do not weaken the validator merely to make an auditor response pass.

A remote auditor would need an explicitly selected image-capable service, its actual model identity and appropriate endpoint configuration. No remote service is required by this project's data preparation or baseline evaluator.
