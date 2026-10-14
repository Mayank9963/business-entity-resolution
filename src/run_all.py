"""End-to-end pipeline: data -> normalise -> block -> train -> predict -> validate.

    python run_all.py                       # everything, one country at a time (16 GB RAM)
    python run_all.py --jobs 3              # 3 countries in parallel (needs ~32 GB RAM)
    python run_all.py --from train          # re-use existing normalisation + blocking
    python run_all.py --from predict        # re-use trained model, only score test
    python run_all.py --smoke               # 5-minute test on a 1% sample (checks setup)

Stages: translit, normalize, block, train, predict, validate.
Starting at a stage deletes the outputs of that stage and all later ones, so nothing
stale (e.g. old candidate files or old scores) is ever mixed into a new run.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SRC = Path(__file__).resolve().parent
V5 = SRC.parent
if "--smoke" in sys.argv:  # must be set before config is imported (also inherited by children)
    os.environ["ER_RUN_DIR"] = str(V5 / "smoke")
    os.environ["ER_DATA"] = str(V5 / "smoke" / "dataset")

from config import DATA, OUTPUT, ROOT, WORK  # noqa: E402
STAGES = ["translit", "normalize", "block", "train", "predict", "validate"]


def py(*args):
    t = time.time()
    print(f"\n>>> python {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-u", *args], cwd=SRC, check=True)
    print(f"<<< done in {(time.time() - t) / 60:.1f} min", flush=True)


def parallel(cmds, jobs):
    with ThreadPoolExecutor(max_workers=jobs) as ex:
        for f in [ex.submit(py, *c) for c in cmds]:
            f.result()  # re-raise failures


def countries(split):
    import polars as pl
    return (pl.scan_parquet(WORK / f"norm_{split}_s1.parquet").select("country").unique()
            .collect()["country"].sort().to_list())


def clean(from_stage):
    idx = STAGES.index(from_stage)
    if idx <= STAGES.index("block"):
        for d in ("cand_train", "cand_test"):
            shutil.rmtree(WORK / d, ignore_errors=True)
    if idx <= STAGES.index("predict"):
        for f in WORK.glob("best_*.parquet"):
            f.unlink()


def make_smoke_dataset(full: Path, out: Path, pct: int = 1):
    """~pct% of S1 entities (+ their true matches + a matching share of other records)."""
    import polars as pl
    if (out / "test" / "test_source1.tsv").exists():
        return
    rd = lambda p: pl.read_csv(p, separator="	", quote_char=None, infer_schema=False)
    keep = lambda df: df.filter(pl.col("entity_id").hash(seed=1) % 100 < pct)
    for split in ("train", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)
        s1 = keep(rd(full / split / f"{split}_source1.tsv"))
        s1.write_csv(out / split / f"{split}_source1.tsv", separator="	", quote_style="never")
        linked = set()
        if split == "train":
            gt = rd(full / "train" / "train_ground_truth.tsv").filter(
                pl.col("source1_entity_id").is_in(s1["entity_id"].implode()))
            gt.write_csv(out / "train" / "train_ground_truth.tsv", separator="	",
                         quote_style="never")
            linked = set(",".join(gt["matched_entity_ids"].fill_null("").to_list()).split(","))
        for s in (2, 3):
            q = rd(full / split / f"{split}_source{s}.tsv")
            q = q.filter(pl.col("entity_id").is_in(list(linked))
                         | (pl.col("entity_id").hash(seed=2) % 100 < pct))
            q.write_csv(out / split / f"{split}_source{s}.tsv", separator="	",
                        quote_style="never")
    print(f"[smoke] sample dataset written to {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="start", default="translit", choices=STAGES)
    ap.add_argument("--jobs", type=int, default=1,
                    help="countries processed in parallel for block/predict (RAM ~4 GB each)")
    ap.add_argument("--smoke", action="store_true",
                    help="quick end-to-end check on a 1%% sample, written to v5/smoke/")
    a = ap.parse_args()
    if a.smoke:
        full = next((p for p in (V5 / "dataset", V5.parent / "dataset")
                     if (p / "test").exists()), None)
        if full is None:
            raise SystemExit("dataset not found - see README.md")
        make_smoke_dataset(full, DATA)
    if not (DATA / "test" / "test_source1.tsv").exists():
        raise SystemExit(f"dataset not found at {DATA} - see README.md")
    print(f"v5 folder: {ROOT}\ndataset:   {DATA}\nstart at:  {a.start}, jobs={a.jobs}")
    t0 = time.time()
    clean(a.start)
    todo = STAGES[STAGES.index(a.start):]
    if "translit" in todo:
        py("translit.py")
    if "normalize" in todo:
        py("normalize.py", "train", "test")
    if "block" in todo:
        for split in ("train", "test"):
            parallel([("run_blocking.py", split, c) for c in countries(split)], a.jobs)
    if "train" in todo:
        py("train.py")
    if "predict" in todo:
        parallel([("predict.py", "score", "test", c) for c in countries("test")], a.jobs)
        py("predict.py", "write", "test")
    if "validate" in todo:
        subprocess.run([sys.executable, str(ROOT / "utils" / "validate_submission.py"),
                        "--matching", str(OUTPUT / "matching_results.tsv"),
                        "--candidate", str(OUTPUT / "candidate_pairs.tsv"),
                        "--test-dir", str(DATA / "test")], check=True)
    print(f"\n[run_all] finished in {(time.time() - t0) / 60:.1f} min -> {OUTPUT}")


if __name__ == "__main__":
    main()
