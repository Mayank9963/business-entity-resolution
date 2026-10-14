# Business Entity Resolution

An end-to-end record-linkage pipeline for the Amazon ML Challenge 2026. The system links noisy business records from sources S2 and S3 to reference entities in S1, while allowing distractor records to remain unmatched. Evaluation uses macro F0.5, so precision is weighted more heavily than recall.

## Pipeline

1. **Learn transliteration maps** from the training ground truth for Indic names and address segments.
2. **Normalize records** into canonical name, address, number, postcode, state, legal-form, alternate-name, and synthetic-token fields.
3. **Generate candidates** within each country using ten inverted-index blocking keys, IDF weighting, document-frequency caps, and a top-10 limit per query.
4. **Build pair features** from string, token, numeric, address, blocking, and within-query ambiguity signals.
5. **Train two LightGBM models:** a seen-country model and a label-free model for unseen countries.
6. **Score and assign** each S2/S3 query to its highest-probability S1 candidate when it clears the model threshold.
7. **Write and validate** `matching_results.tsv` and `candidate_pairs.tsv`.

The detailed rules, constants, and comparisons are documented in [V5_MODEL_SPEC.md](V5_MODEL_SPEC.md).

## Results

| Model | Validation macro F0.5 | Precision | Recall | Blocking recall ceiling | Threshold |
|---|---:|---:|---:|---:|---:|
| v5 seen-country model | **0.98253** | 0.99586 | 0.95773 | 0.9808 | 0.75 |
| v5 unseen-country proxy (LOCO average) | **0.949** | — | — | — | 0.90 |

LOCO scores for the label-free model are 0.9621 for India→US and 0.9358 for US→India. A v5 leaderboard score was not available when the model specification was written.

## Repository layout

```text
.
├── src/                  # pipeline and modeling code
├── utils/                # submission validator
├── requirements.txt
├── run_linux.sh
├── run_windows.bat
└── V5_MODEL_SPEC.md
```

The dataset, intermediate `work/` artifacts, generated `output/` files, and Python caches are intentionally excluded from version control.

## Data layout

Place the challenge data under `dataset/`:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Alternatively, set `ER_DATA` to the dataset directory.

## Installation

Python 3.11–3.13 is supported. From the repository root:

```bash
python -m pip install -r requirements.txt
```

## Run

Linux or macOS:

```bash
chmod +x run_linux.sh
./run_linux.sh
```

Windows:

```bat
run_windows.bat
```

Direct invocation and useful options:

```bash
cd src
python run_all.py                  # complete pipeline; suitable for a 16 GB machine
python run_all.py --jobs 3         # parallel countries; approximately 32 GB RAM recommended
python run_all.py --from train     # resume at train
python run_all.py --from predict   # reuse the trained models
python run_all.py --smoke          # quick end-to-end test on a 1% sample
```

To reproduce the shipped 16 GB configuration, set `ER_TRAIN_Q=450000`. Other optional environment variables are `ER_WORKERS` and `ER_RUN_DIR`.

Generated submissions are written to `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## Notes

- Matching is restricted to records with the same country value.
- Each S2/S3 record links to at most one S1 entity.
- The pipeline uses CPU-only LightGBM and no external data, APIs, embeddings, or LLMs.
- Randomized stages use seed 42; use the pinned dependency versions for comparable validation splits.
