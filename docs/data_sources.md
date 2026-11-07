# Data sources and development record

Verified on 8 November 2026. Downloaded file fingerprints are included in `configs/data_sources.json`; the same values are copied into each prepared bundle and evaluation run.

## VQA-RAD

- Original dataset paper: Lau et al., [A dataset of clinically generated visual questions and answers about radiology images](https://www.nature.com/articles/sdata2018251), Scientific Data (2018).
- Distribution used: [flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad).
- Revision: `bcf91e7654fb9d51c8ab6a5b82cacf3fafd2fae9`.
- Verified source rows: 1,793 training and 451 test questions.

This is a processed mirror, not a claim of an identical export of every original paper record. The original publication describes 2,248 question-answer pairs; this distribution contains 2,244. It supplies image bytes, question and answer but not the original patient identifiers or open/closed annotation. This project marks yes/no references as closed and other references as open for score breakdowns only. That derived field is never used to choose the prompt.

The inspected [MedUniEval VQA-RAD adapter](https://github.com/ZJUI-AI4H/Hulu-Med/blob/6d4d47aa014342b1beda70695ab7c799edc7ade6/MedUniEval/utils/VQA_RAD/VQA_RAD.py) selects this distribution's test split. The dataset identity is pinned independently of the evaluation toolkit.

## SLAKE

- Author dataset site: [SLAKE](https://www.med-vqa.com/slake/).
- Author-linked distribution: [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE).
- Revision: `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`.
- Files: `train.json`, `validation.json`, `test.json`, `imgs.zip`. The knowledge-graph archive is not required by this evaluator.

The authors note that the cleaned distribution can differ from their original paper. The prepared data therefore identifies this revision rather than assuming paper-era row counts. Only English questions are evaluated. Source `OPEN`/`CLOSED` annotations are retained for score breakdowns.

| Source partition | All languages | English rows | English rows with a usable question and reference |
|---|---:|---:|---:|
| Training | 9,835 | 4,919 | 4,918 |
| Validation | 2,099 | 1,053 | 1,053 |
| Test | 2,094 | 1,061 | 1,061 |

The English training record `slake:train:1622` has an empty reference. It is excluded, without filling in an assumed answer. No test reference is missing in the checked release.

## Prepared roles

| Dataset | Role | Questions | Unique decoded images | Open | Closed |
|---|---|---:|---:|---:|---:|
| VQA-RAD | Train | 655 | 100 | 305 | 350 |
| VQA-RAD | Feedback | 79 | 11 | 29 | 50 |
| VQA-RAD | Test | 451 | 203 | 200 | 251 |
| SLAKE | Train | 4,911 | 450 | 2,971 | 1,940 |
| SLAKE | Feedback | 1,053 | 96 | 631 | 422 |
| SLAKE | Test | 1,061 | 96 | 645 | 416 |

The preparation audit found 1,059 VQA-RAD source training questions whose decoded images also occur in the source test set. Those questions are excluded from adaptation. Seven exact duplicate SLAKE adaptation questions are also excluded. All 451 VQA-RAD test questions and all 1,061 English SLAKE test questions are retained.

| Exclusion reason | Count |
|---|---:|
| Non-English SLAKE question | 6,995 |
| Missing SLAKE non-test question/reference | 1 |
| Duplicate SLAKE adaptation question | 7 |
| VQA-RAD training question on a test image | 1,059 |

The resulting manifests contain 5,566 training, 1,132 feedback and 1,512 final test questions. The bundle's `source_english_rows` field records the 9,276 usable exported English rows after excluding the empty reference. The train and feedback pools are restricted local-study pools; they do not reconstruct MedEvo's 81,293-question training bank or its 5,000-question feedback set.

## Integrity and rights

All six downloaded files were checked against their pinned byte size and SHA-256. All referenced images decoded successfully. Exported manifests and images have individual digests. Adaptation and evaluation roles have no overlapping decoded image hashes. This is not a patient-level or near-duplicate audit.

Raw source files are not stored in the repository; the downloader retrieves them from the cited distributions. Prepared images are published as a release asset (see data/README.md). Consult each original source's current access and usage terms before redistributing data. A model's code or weight licence does not grant rights to benchmark images. No new licence for the datasets or third-party implementation is implied by this project.

## Evaluation reference

Inspected repository: [ZJUI-AI4H/Hulu-Med](https://github.com/ZJUI-AI4H/Hulu-Med/tree/6d4d47aa014342b1beda70695ab7c799edc7ade6/MedUniEval), commit `6d4d47aa014342b1beda70695ab7c799edc7ade6`. Source inspection covered the VQA-RAD and SLAKE adapters, shared question formats and scoring utilities. This project implements its own small loader and scorer; it does not bundle those modules or claim exact metric equivalence.
