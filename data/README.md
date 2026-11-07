# Benchmark data

`data/benchmarks/` contains the prepared split manifests used by every experiment in this repository.
The images are distributed separately because of their size.

| File | Contents |
|---|---|
| `dataset.json` | Source revisions, file hashes, split policy, counts and manifest hashes |
| `train.jsonl` | Training-role questions (Phase 3 collection only) |
| `feedback.jsonl` | Feedback-role questions (protocol decisions only) |
| `test.jsonl` | Reserved test questions |
| `exclusions.jsonl` | Every excluded or deduplicated source question, with the reason |

| Dataset | Train | Feedback | Test |
|---|---:|---:|---:|
| VQA-RAD | 655 | 79 | 451 |
| SLAKE (English) | 4,911 | 1,053 | 1,061 |

Training excludes any image that appears in the feedback or test roles. Patient-level and near-duplicate
separation are not guaranteed. Details: [docs/data_sources.md](../docs/data_sources.md).

## Images

Each record's `image_path` points to `images/<sha256>.png`, and `image_file_sha256` is checked on load.

**Option A: download the prepared images.** Download `medevo-benchmark-images-v1.zip` from the
repository's Releases page and extract it into `data/benchmarks/`, so that the PNG files are in
`data/benchmarks/images/`. Then verify:

```powershell
python evaluate.py inspect
```

**Option B: rebuild from the original sources.** This downloads the pinned source files (about 251 MB)
into a new folder and regenerates the manifests:

```powershell
python evaluate.py --data-dir data\rebuilt prepare
```

The manifest hashes in `data\rebuilt\dataset.json` should match `data\benchmarks\dataset.json`.

## Sources and licenses

| Dataset | Source | License |
|---|---|---|
| VQA-RAD | [flaviagiammarino/vqa-rad](https://huggingface.co/datasets/flaviagiammarino/vqa-rad), revision `bcf91e7` | CC0 1.0 |
| SLAKE | [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE), revision `a9083ce` | CC BY 4.0 |

The manifests and images here are derived from these datasets and keep their original licenses.
If you use them, cite the original work:

- Lau, J. J., Gayen, S., Ben Abacha, A., and Demner-Fushman, D. A dataset of clinically generated visual
  questions and answers about radiology images. *Scientific Data* 5, 180251 (2018).
- Liu, B., Zhan, L.-M., Xu, L., Ma, L., Yang, Y., and Wu, X.-M. SLAKE: A semantically-labeled
  knowledge-enhanced dataset for medical visual question answering. *IEEE ISBI* (2021).

Changes made here: English SLAKE questions only, image re-encoding to PNG, exact-image split separation
and deduplication of repeated training questions, all recorded in `exclusions.jsonl`.
