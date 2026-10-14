"""Paths and global settings for the entity-resolution pipeline (v5, portable).

Folder layout (everything relative to the v5 folder, so it can be copied anywhere):
    v5/src/        code (this file)
    v5/dataset/    challenge data: dataset/train/*.tsv, dataset/test/*.tsv
    v5/work/       intermediates + model artifacts (created automatically)
    v5/output/     matching_results.tsv, candidate_pairs.tsv (created automatically)

Environment overrides (optional):
    ER_DATA      path to the dataset folder (default: v5/dataset, else ../dataset)
    ER_WORKERS   processes for normalisation (default: CPU count - 2, max 16)
    ER_TRAIN_Q   number of training queries (default 650000; ~1M if RAM >= 32 GB)
    ER_RUN_DIR   where work/ and output/ are created (default: the v5 folder; the
                 smoke test uses v5/smoke)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # the v5 folder


def _data_dir() -> Path:
    if os.environ.get("ER_DATA"):
        return Path(os.environ["ER_DATA"]).resolve()
    for cand in (ROOT / "dataset", ROOT.parent / "dataset"):
        if (cand / "test").exists():
            return cand
    return ROOT / "dataset"


DATA = _data_dir()
RUN = Path(os.environ.get("ER_RUN_DIR", ROOT))
WORK = RUN / "work"             # intermediate parquet files, model artifacts
OUTPUT = RUN / "output"
ART = WORK / "artifacts"

for p in (WORK, OUTPUT, ART):
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
N_THREADS = -1
N_WORKERS = int(os.environ.get("ER_WORKERS", max(2, min(16, (os.cpu_count() or 4) - 2))))
N_TRAIN_Q = int(os.environ.get("ER_TRAIN_Q", 650_000))


def src_path(split: str, source: int) -> Path:
    return DATA / split / f"{split}_source{source}.tsv"


def norm_path(split: str, source: int) -> Path:
    return WORK / f"norm_{split}_s{source}.parquet"
