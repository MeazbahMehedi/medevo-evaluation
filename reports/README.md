# MedEvo report and findings

[Download the revised technical report](MedEvo_Technical_Report.pdf). The study uses 400 held-out questions, with 200 each from VQA-RAD and SLAKE.

| System | Primary accuracy | Secondary accuracy |
|---|---:|---:|
| Qwen3.5-9B reference | 57.0% | 57.3% |
| Qwen3.5-2B base | 21.5% | 47.0% |
| Qwen3.5-2B with LoRA | 50.0% | 50.0% |

LoRA improved primary exact match by 28.5 percentage points. When numeric yes/no answers were accepted, the improvement was 3.0 points (95% CI −1.0 to 6.8, McNemar p = 0.17). The training bank contained 40 correct rollouts and three admitted repairs.

- [Compiled findings](../results/compiled/results.md) and [machine-readable metrics](../results/compiled/results.json)
- [Prediction and experiment records](../results/README.md)
- [Accuracy by question type](accuracy.png) and [answer-format effects](format_content.png)
- [Data preprocessing](preprocessing_workflow.png) and [test subset composition](test_subset_composition.png)
- [Validation workflow](validation_design.png) and [paired scoring outcomes](validation_transitions.png)
- [Repair screening](repair_admission_flow.png), [learnability](learnability.png), and [training loss](training_loss.png)
- [CPU runtime](cpu_runtime_by_stage.png)
