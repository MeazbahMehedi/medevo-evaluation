# References

## Method

- Wang, X., Chen, Z., Zheng, C., Wang, Q., Lao, J., Liu, Y., Liu, J., Cao, J., Lin, H., Lu, Z., Jin, H., Wang, J.,
  and Wu, L. *MedEvo: Medical Reasoning Evolution through Knowledge Navigation.* arXiv:2610.17690, 2026.
  [Paper](https://arxiv.org/abs/2610.17690) ·
  [Project page](https://pork-pay.github.io/MedEvo/) ·
  [Repository](https://github.com/pork-pay/MedEvo)

The answering prompt used throughout this study is the paper's (Appendix D.1):
`Please reason step by step, and put your final answer within \boxed{}.`
The v2 audit instruction follows the paper's audit prompt (Appendix D.3).

## Datasets

- Lau, J. J., Gayen, S., Ben Abacha, A., and Demner-Fushman, D. A dataset of clinically generated visual
  questions and answers about radiology images. *Scientific Data* 5, 180251 (2018).
  Distribution used: [flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) (CC0 1.0).
- Liu, B., Zhan, L.-M., Xu, L., Ma, L., Yang, Y., and Wu, X.-M. SLAKE: A semantically-labeled
  knowledge-enhanced dataset for medical visual question answering. *IEEE ISBI* (2021).
  Distribution used: [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) (CC BY 4.0).

## Models

- [Qwen3.5-9B](https://ollama.com/library/qwen3.5:9b) via Ollama (Q4_K_M): baseline and auditor.
- [Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B): primary (adapted) model.
- [SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct):
  Phase 1-2 feasibility model.
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) and `qwen2.5vl:3b`:
  earlier auditor experiments.

## Software

- [Hugging Face Transformers](https://huggingface.co/docs/transformers) and
  [PEFT](https://huggingface.co/docs/peft) (LoRA).
- [bitsandbytes 0.50.2](https://huggingface.co/docs/bitsandbytes/v0.50.2/en/installation): CPU 4-bit
  feasibility check.
- [Ollama](https://ollama.com) and its [thinking controls](https://docs.ollama.com/capabilities/thinking).
