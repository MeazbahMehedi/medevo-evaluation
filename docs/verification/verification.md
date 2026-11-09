# Verification

Prepared on 8 November 2026. The development checks ran on Linux with Python 3.12.14, not on the target Windows PC. The initial Phase 2 development record is `verification.json`. The added diagnostic has its own current record in `prompt_verification.json` and explanation in `prompt_verification.md`. The current scoring-only diagnostic is documented in `answer_verification.json` and `answer_verification.md`. The current repair-pilot checkpoint is documented in `repair_verification.json` and `repair_verification.md`. The current non-repair validation correction is documented in `audit_review_verification.json` and `audit_review_verification.md`. The current software test output is `test_results.txt`.

## Checks completed

| Check | Outcome |
|---|---|
| Current software suite | 86 checks passed; skip-field handling corrected and seven review checks added |
| Required packages | No broken requirements in the development environment |
| Windows dependency availability | PyArrow 26.0.1 has a CPython 3.12 Windows x64 wheel |
| Source identity | All six pinned dataset files matched their byte sizes and SHA-256 fingerprints |
| Real data preparation | Both benchmark distributions decoded and produced the documented manifests |
| Data roles | All exported questions and referenced images validated; no exact decoded-image overlap across train, feedback and test |
| Test preservation | 451 VQA-RAD and 1,061 English SLAKE test questions retained |
| Small model runs | One real feedback example generated and scored from each dataset |
| Resume | An already completed feedback run resumed without adding or regenerating a prediction |
| Answer-format diagnostic | Full paired command exercised on five real feedback questions per dataset; concise prompt did not produce valid answer boxes |
| Whole-answer diagnostic | Read-only scoring exercised on saved outputs; concise trials matched 3/5 per dataset under the separate whole-answer rule, with boxed scores unchanged |
| Repair model compatibility | Real primary-model entropy, answer readout and exact unfinished-assistant prefill checked on a training image |
| Three-case repair driver | Linux scripted-driver check retained; Windows live auditor returned three non-repair decisions, with medical quality unresolved |
| Saved non-repair review | Reported objects preserve one unresolved/two skip actions under corrected validation; no repair is created |
| Final test generation | Not performed |

The software suite checks image and manifest changes, exact-image separation, duplicate and foreign predictions, prompt boundaries, partial coverage, conservative parsing, token multiplicity, interrupted journals, resume identity and explicit final-test selection. Existing tests also check synthetic supervision masks, adapter updates, auditor request contracts and process-limit termination. Mocked tests are identified by their source fixtures; they are not presented as model or medical evaluations.

## Feedback observations

Each development run selected one feedback question using the declared seed. Both used the actual pinned pretrained model and actual benchmark images. They exercised checkpoint loading, preprocessing, generation, journaling and scoring.

| Dataset | Recorded generation | Scoring observation |
|---|---|---|
| VQA-RAD | Completed with a final box | Box contained a disease phrase instead of the reference yes/no answer; strict EM marked it wrong |
| SLAKE | Completed without a final box | Parser recorded a format failure and scored it wrong |

These are pipeline observations from one question per dataset, not estimates of benchmark accuracy. The responses also show why medical competence and output-format compliance need separate analysis. The parser was not relaxed to accept the observed unboxed response. Final test predictions and adapted-model results remain unavailable.

The first exploratory run downloaded the missing model weights. Later checks used the local cache in offline mode. Development timings in the JSON record are observations from this development environment. Some runs overlapped, so the timings are not controlled speed comparisons or forecasts for the Ryzen desktop.

## Resource limitation

The Linux process-memory interface returned implausible RSS values below the loaded model's parameter storage. The evaluator flagged these readings as invalid. They must not be reported as validated peak RAM or used to scale the experiment. The resource-limit branches are tested with controlled processes/samples; this does not establish physical high-memory behaviour on Windows.

The Windows feasibility readings reported earlier are kept separately in `experiment_log.md`. Windows runs later confirmed the 38-test suite, data preparation/inspection and five-question feedback runs passing on Windows, with plausible sampled memory. Those observations are recorded in `experiment_log.md`. The paired prompt diagnostic was also completed on Windows. The scoring-only diagnostic was then completed on Windows, confirming 3/5 whole-answer matches per concise subset with the boxed scores unchanged. The Windows run passed all 79 checks on Windows and completed the first live pilot. Its one unresolved/two skip responses exposed the unused-hint validation error. The corrected 86-test update and read-only archive review await Windows confirmation. A resumed attempt's resource history is retained separately; the latest measurement is not treated as a cumulative peak.

## Scope still open

The repair pilot records entropy proposals, structural boundaries, exact continuation and preliminary checks. Medical auditor competence, full repair admission, feedback-guided training selection, medical LoRA adaptation, full baseline/adapted test results and the final technical report remain open. Patient-level separation, image near-duplicate detection and possible benchmark exposure during pretraining are not established.
