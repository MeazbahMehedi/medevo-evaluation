# Experiment log

Use one entry per operation. Keep planned budgets separate from measured outcomes. Preserve the complete local output folder, including failed runs.

## Record template

- Date, time and timezone:
- Run ID and output folder:
- Stage and purpose:
- Exact command and interpreter:
- Code version and model revision:
- Dataset revision, role, manifest hash and selected count:
- Settings, prompt and scoring protocol:
- Outcome and generation coverage:
- Scores, when applicable:
- Generation time, worker time and resource measurement validity:
- Error, exclusion or limitation:
- Decision and next action:

## Reported Windows feasibility observations

These observations come from the Windows terminal during Phase 1. They have not been remeasured by the Phase 2 development environment. The timestamps below identify the UTC-style run folders printed by the program.

| Run | Observation |
|---|---|
| `20261108T010508Z_inference_e6bdfb` | Synthetic red-circle response correct; 12 tokens; generation 1.79 s; worker elapsed 322.734 s; sampled peak 2.388 GiB |
| `20261108T011621Z_train_597e0b` | Synthetic LoRA update 1.165 s; 409,600 trainable parameters; 128 parameter tensors changed; sampled frozen parameter unchanged; reload passed; worker elapsed 10.703 s; sampled peak 2.53817 GiB |
| `20261108T015312Z_auditor_f92468` | Ollama `qwen2.5vl:3b` synthetic request/schema check passed; response used `prefix_end: 1` |

The auditor's offset 1 is only a valid integer character offset. It does not demonstrate a medically supported reasoning-step boundary. Semantic boundary validation and medical quality remain open for the repair phase. The synthetic adapter is a feasibility artifact, not a medical model.

`ollama ps` reported `qwen2.5vl:3b`, model ID `fb90415cde1e`, 2.8 GB loaded allocation, 100% CPU and context 4096. That is a service report, not a measured peak-RAM result. The Python worker measurements do not include this separate service.

An earlier overnight inference timed out after the PC slept. Retain that run as a failed attempt; do not use its elapsed duration for performance comparisons.

## Phase 2 preparation

Pinned data preparation and local software checks are described in `verification.md`. Any development feedback inference is separate from final test evaluation. No held-out medical test score is available at this checkpoint.

## Remaining decisions

- Run the expanded 45-check diagnostic suite on Windows; the earlier 38-check suite and data preparation already passed.
- Compare the approved concise instruction with the retained five-question feedback runs before deciding the next model/prompt step.
- Validate semantic repair boundaries and answer-blind medical hints.
- Implement and audit adaptation, then freeze the final comparison.
- Run final baseline and adapted evaluations and write the technical report.

GitHub publication is not required to run the experiments.

## Reported Windows Phase 2 observations

These results come from the Windows installation. They are separate from Linux package checks. The 38-test suite passed in 3.460 seconds. Data preparation and inspection returned 5,566 training, 1,132 feedback and 1,512 final test questions, with the documented source revisions and exclusions.

| Run | Questions | Format failures | Token-limit hits | Strict EM | Generation time | Worker elapsed | Sampled peak RAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| `20261108T023429Z_vqa_rad_feedback_119aa2` | 5 | 4 | 2 | 0/5 | 31.299 s | 44.969 s | 2.42057 GiB |
| `20261108T023954Z_slake_feedback_a5e572` | 5 | 5 | 1 | 0/5 | 20.596 s | 33.828 s | 2.432701 GiB |

Both runs recorded all selected generations, no generation errors, clean worker exits and plausible memory samples. These were feedback subsets, not final test evaluation. The SLAKE subset happened to contain only open questions. Raw responses showed both reference mismatches and unboxed responses naming the reference concept; strict scores remain unchanged.

Decision: retain these runs and test a separately labelled concise instruction on the same ten questions. Keep the original model, data, token budget and scorer fixed. The decision covers this diagnostic, not medical training or adoption of the trial instruction for final evaluation.

## Reported Windows prompt comparison

The completed comparison `20261108T030115Z_prompt_comparison_2f6e16` reused the original five feedback questions from each dataset and retained the original evidence.

| Dataset | Trial folder | Original valid boxes | Trial valid boxes | Original strict EM | Trial strict EM | Trial generation |
|---|---|---:|---:|---:|---:|---:|
| VQA-RAD | `20261108T030121Z_vqa_rad_feedback_71a714` | 1/5 | 0/5 | 0/5 | 0/5 | 5.065 s |
| SLAKE | `20261108T030147Z_slake_feedback_b8def3` | 0/5 | 0/5 | 0/5 | 0/5 | 5.091 s |

The concise instruction did not improve boxed-answer compliance. Decision: preserve that negative result and add a separately labelled whole-answer scoring diagnostic on the saved predictions. The approved rule uses complete-string equality after fixed case, whitespace and terminal-punctuation cleanup; it does not guess acceptable synonyms or extract a favourable phrase. No new inference or training is authorized by that scoring step.

## Reported Windows whole-answer diagnostic

`20261108T032319Z_answer_diagnostic_42e70b` completed without new inference. On the five concise responses per dataset, whole-answer exact match was 3/5 for VQA-RAD and 3/5 for SLAKE. Boxed scores remained 0/5. The original reasoning-response whole-answer scores were also 0/5. These are exploratory re-scorings of fixed feedback outputs, not medical learning gains or final test results.

Decision: proceed to an approved three-question training-only repair pilot, record entropy and audit evidence, and inspect candidate checks before medical SFT. The normal final-evaluation protocol remains undecided.

## First reported Windows repair pilot

The Windows run passed all 79 software checks in 8.585 seconds, then ran `20261108T044904Z_repair_pilot_01186c`. All three selected cases reached Ollama, but the original validator recorded `rejected_invalid_audit` for each because non-repair actions contained nonempty hints. Runtime was 90.25 seconds; the sampled Python-worker peak was 2.433205 GiB and was marked plausible. That measurement excludes the Ollama service. No continuation, target export or optimizer step occurred.

Inspection of the supplied raw responses showed one unresolved action and two skip actions, all with step 0. Their unused hints caused the cross-field rejection. Decision: correct handling of unused non-repair fields, preserve all raw decisions and original recorded outcomes, and review the existing archive separately. The correction does not produce a repair or validate the auditor's medical statements. Phase 4 remains pending qualified training evidence and the remaining admission checks.

## Reported Windows audit review and next collection

The read-only review `20261108T051139Z_audit_review_593326` completed under `nonrepair-auxiliary-unused-v2`. It retained all three original rejection records and interpreted their raw actions as one unresolved and two skips. There were zero repair decisions and no new continuations or training exports.

The grey–white differentiation readout repeated the question. The modality readout differed literally from the reference, with clinical equivalence unestablished. The spinal-cord readout contradicted its reference and received no repair. These observations do not establish a working live repair mechanism.

Next experiment: a bounded twelve-question training collection: six fixed questions per dataset, the existing three cases reused and nine new cases processed. Models, prompts, budgets and verification rules remain unchanged. Question echoes receive a separate diagnostic label. Potential correct and repaired rollouts will be indexed for later admission, while every skip, unresolved decision and rejected attempt is preserved. If the complete collection produces no potential candidates, reassess the model/auditor setup before training.

Development verification passed 102 tests, including sixteen collection checks and a supervised CLI worker with scripted model responses. This is software evidence, not a new medical result. The live twelve-case collection has not yet been reported. No medical adapter training or final test-set evaluation has been performed at this checkpoint.

## Reported Windows twelve-case collection and archive review

The Windows run passed the 102-test suite in 13.301 seconds and completed `20261108T061937Z_training_collection_9120a6`. The run reused three cases and processed nine new questions. It recorded eight skips, two unresolved decisions, one rejected hint and one invalid audit. Initial readouts comprised ten mismatches, one question echo and one truncated response. No continuation was generated, and there were zero potential candidates, targets, optimizer steps or adapters. Worker elapsed time was 202.625 seconds and plausible sampled peak Python-process-tree RAM was 2.437794 GiB, excluding Ollama.

Review of the uploaded archive verified twelve training identities, summary outcome counts, unchanged reused-case bytes and reference-free auditor request fields. Source fingerprints matched the development copy. Two auditor responses used the raw action `repair`: one passed structural validation but failed hint checking; the other used the invalid repair step 0. Neither reached continuation.

Case 01's reasoning stated “yes” before the readout echoed the question. Case 05 named “aorta” before a truncated number readout. Case 09 contained “center” within a longer response with additional claims. Other cases included polarity, side, specificity and completeness mismatches. Several auditor outputs repeated instructions, supplied unusable evidence or repeated the primary response. These are output/protocol observations, not independent clinical image interpretation.

Next experiment: a bounded readout-only experiment on the same twelve saved questions and reasoning responses. It uses the same primary model and 32-token cap with a single text-only extraction request. It keeps original matching and results intact, calls neither medical reasoning nor the auditor, and admits no training examples. Development verification passed 115 tests in 8.394 seconds, including a supervised worker with scripted readouts. The live readout comparison remains pending. Medical adaptation and the final benchmark comparison are still incomplete.

## Reported Windows readout result and auditor-only diagnostic

The Windows run passed all 115 tests in 23.273 seconds and completed `20261108T064457Z_readout_comparison_efb00c`. All twelve trial readouts were generated. Whole-answer reference matches increased from 0/12 to 2/12: VQA-RAD 2/6 and SLAKE 0/6. Worker time was 12.781 seconds, with plausible sampled peak Python RAM of 2.299995 GiB. There were no new medical reasoning/auditor calls, training exports or updates.

The supplied comparison table showed that VQA-RAD case 000508 faithfully extracted “aorta” from its saved reasoning. Case 000501 returned “No” although its reasoning expressed uncertainty; that match is not faithful extraction and must not be admitted for training. Case 000201 preserved a yes answer within a sentence but failed unchanged whole-answer matching. SLAKE case 2526 lost the location answer and returned “heart.” The reported 2/12 lexical result remains unchanged; it does not establish two correct training examples, clinical accuracy or improved medical reasoning.

Next experiment: one auditor-only experiment on the original twelve cases. It changes only the system instruction, preserving images, questions, reasoning, steps, entropy, auditor model/endpoint, schema, sampling, budgets and validation checks. There are no new primary reasoning/readout calls, continuations or training updates. Invalid, skipped and unresolved responses remain recorded. Coherence requires review; a structurally valid repair is not a clinically validated candidate. If the trial remains unreliable, reassess the setup before additional prompt trials. Final benchmark evaluation, medical adaptation and the technical report are still incomplete.

## Reported Windows auditor result and serving controls

The Windows run passed 129 tests in 36.892 seconds and completed `20261108T070535Z_audit_comparison_06b8a7`. The shorter instruction reduced valid audit structures from 11/12 to 2/12. The trial recorded ten invalid audits, one unresolved decision and one rejected hint, with zero repairs passing hint checks. Worker elapsed time was 224.609 seconds. Sampled peak Python HTTP-worker RAM was 0.035954 GiB; this excludes external Ollama model memory and must not be reported as total inference RAM.

The supplied comparison showed eight repairs missing required support/hint/evidence, one invalid repair step/family, one malformed JSON response, one unresolved response with repetitive irrelevant evidence and one declarative finding used as a hint. No continuation or training occurred. The shorter instruction is not adopted.

The local server reported version 0.40.15 and a 3.8B `qwen25vl` GGUF model with Q4_K_M quantization, vision/completion capabilities and an active Go template. These metadata do not establish that quantization or templating caused the failures. Two simple images were then tested in unconstrained and schema-constrained modes, with the schema explicitly present in both prompts. All four replies named the correct color and shape; only the two constrained replies used the required flat JSON fields. These controls establish neither clinical reasoning ability nor medical audit quality.

Next experiment: one integration call using the first original training case, the original medical instruction and an explicitly appended copy of the same schema. All other controls and validation rules remain unchanged. No new primary reasoning/readout, continuation, admission or training is included. The result must be reviewed before any further call or setup change; its live outcome is pending.

## Reported single-case schema result and approved model comparison

The Windows check `20261108T073540Z_single_audit_00a8a9` completed one call in 16.265 seconds. Its raw response requested repair at step 1 with `prefix_supported: false`, an inspection hint and evidence consisting of “f.” The existing validator rejected it. Explicit schema grounding did not produce a usable audit in this observed case. No continuation, training export or update occurred, and no validity field was manually repaired.

Next experiment: downloading `qwen2.5vl:7b` and testing the same case with the same schema-grounded request, changing only model identity. The standalone comparison requires the captured 3B check as its baseline, retains its files and keeps all request controls and local validation fixed. The larger-model result is pending. Model size is a hypothesis; the experiment provides no automatic medical-quality or training-admission guarantee. Further local prompt variants are not planned.
