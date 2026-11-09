# CPU quantized-training backend check

This checkpoint checks whether the installed CPU stack can update a small LoRA adapter over a real frozen NF4 projection. It uses synthetic random tensors, not questions, images or medical labels. It does not load or download pretrained model weights.

The Windows inventory reported Python 3.12.10, PyTorch 2.15.0+cpu, Transformers 5.22.2, PEFT 0.21.2, Accelerate 1.15.0, AVX512, 19.77 GiB available RAM and 255.13 GiB free disk. That inventory alone does not establish quantized-training support.

## Run on Windows

In the PyCharm PowerShell terminal, from that project folder:

```powershell
& '.\.venv312\Scripts\python.exe' -m pip install --only-binary=:all: --no-deps 'bitsandbytes==0.50.2'
```

This installs only the pinned binary package. It does not ask pip to change existing dependencies or compile a local backend. If installation fails, retain the full error and stop; do not upgrade PyTorch or add other packages to force a pass.

After installation succeeds:

```powershell
& '.\.venv312\Scripts\python.exe' '.\check_training_backend.py'
```

Keep the PC awake. The worker has a 180-second timeout and a sampled 4 GiB process-memory guard. These are supervisor limits, not a hard operating-system memory reservation. Peak RSS is sampled every 0.2 seconds and may miss short peaks. Unavailable or implausibly low readings are flagged and cannot support a RAM claim.

The result goes into `outputs/cpu_4bit_backend_v1/`. Existing attempts are never overwritten. Return the printed result and, if needed, `console.log`; inspect a completed or failed attempt before considering another run.

## What a pass requires

1. A 64 × 64 projection must actually be packed as CPU NF4 weights with compressed quantization statistics. A floating-point substitute does not pass.
2. PEFT must attach its bitsandbytes LoRA implementation. Only the 512 adapter parameters may be trainable.
3. Forward and backward calculations must produce a finite loss, finite adapter gradients and a finite, nonzero input gradient through the quantized base.
4. One AdamW update must change adapter parameters and the output while preserving the frozen packed base and its quantization state.
5. Saving and restoring the packed base and adapter must preserve the base and reproduce the updated output within the stated tolerance.

Outputs are compared in training mode with zero dropout. Optimized CPU inference repacking is a separate path and is not tested. A missing optional `kernels` warning can appear even when the basic CPU operator succeeds. The final checks determine the result; optimized-kernel availability and performance are not established. No additional package is installed automatically.

The fresh worker enables Hugging Face and Transformers offline modes before importing their dependent packages. Neither model identifiers nor remote checkpoints are supplied. Outputs include package versions, script fingerprint, attempted operations, the tiny synthetic checkpoint and resource records.

## Development verification and limits

The real synthetic test passed in the Linux development environment with the same reported core package versions and bitsandbytes 0.50.2. It performed one update, kept the packed base fixed and reproduced the output exactly after reload. The complete 178-test software suite also passed, including this real operator check. Its sampled RSS was implausibly low and is explicitly marked invalid. This is functional development evidence, not a Windows result or a reliable memory measurement.

The Windows run passed: 5.141 seconds total, sampled worker peak 0.377594 GiB, one synthetic update and zero reload output difference. This does not prove that a complete Qwen multimodal checkpoint can load or train within 32 GiB RAM. It does not test attention, image processing, activation memory, full-model serialization, runtime per medical example or medical reasoning quality. The current Ollama GGUF inference model is a different backend; these results cannot be used as its training implementation or as a controlled before/after comparison.

This checkpoint admits no medical examples and performs no medical optimizer steps. Repair admission, full-model feasibility, medical adaptation and final held-out evaluation remain open.

## References

- [bitsandbytes 0.50.2 installation and supported CPU platforms](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation)
- [PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization)
- [PEFT custom-model adapter guide](https://huggingface.co/docs/peft/v0.21.0/en/developer_guides/custom_models)

These documents describe backend features. Compatibility with the complete medical experiment requires separate measurement.
