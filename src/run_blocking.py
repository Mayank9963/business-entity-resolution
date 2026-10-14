"""Stage 2: candidate generation for a split -> work/cand_<split>.parquet
Columns: id (S2/S3 query), s1_id, bscore, brank.  Countries are processed one at a
time (open set: whatever labels appear in S1), which keeps memory bounded."""
import gc
import sys
import time

import polars as pl
import pyarrow.parquet as pq

import blocking
from config import WORK, norm_path

K = 10
COLS = ["entity_id", "country", "name_core", "addr_set", "nums", "postcode"]


def cand_dir(split):
    d = WORK / f"cand_{split}"
    d.mkdir(exist_ok=True)
    return d


def cand_path(split):
    """Glob over the per-country candidate files (usable with pl.scan_parquet)."""
    return str(cand_dir(split) / "*.parquet")


def countries(split: str):
    return (pl.scan_parquet(norm_path(split, 1)).select("country").unique().collect()
            ["country"].sort().to_list())


def run(split: str, k: int = K, only: str | None = None):
    t0 = time.time()
    s1_all = pl.scan_parquet(norm_path(split, 1)).select(COLS)
    q_all = pl.concat([pl.scan_parquet(norm_path(split, s)).select(COLS) for s in (2, 3)])
    countries = s1_all.select("country").unique().collect()["country"].sort().to_list()
    total = 0
    for country in countries:
        if only is not None and country != only:
            continue
        path = cand_dir(split) / f"{country}.parquet"
        if path.exists():  # resumable: country already done
            print(f"[block] {split}/{country}: exists, skipping", flush=True)
            continue
        tmp = path.with_suffix(".tmp")
        writer = None
        s1 = s1_all.filter(pl.col("country") == country).collect()
        q = q_all.filter(pl.col("country") == country).collect()
        index = blocking.build_index(s1)
        s1_ids = s1.select(pl.col("entity_id").alias("s1_id")).with_row_index("s1_idx")
        q_ids = q.select(pl.col("entity_id").alias("id")).with_row_index("q_idx")
        print(f"[block] {split}/{country}: S1 {s1.height:,}, queries {q.height:,}, "
              f"index {index.height:,} ({time.time() - t0:.0f}s)", flush=True)
        n = 0
        for part in blocking.query_chunks(index, q, k):
            out = (part.join(q_ids, on="q_idx").join(s1_ids, on="s1_idx")
                   .select("id", "s1_id", "bscore", "brank")).to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(tmp, out.schema)
            writer.write_table(out)
            n += out.num_rows
        if writer is not None:
            writer.close()
            tmp.replace(path)
        total += n
        print(f"[block] {split}/{country}: {n:,} candidate pairs ({time.time() - t0:.0f}s)",
              flush=True)
        del s1, q, index, s1_ids, q_ids
        gc.collect()
    print(f"[block] {split}: {total:,} new pairs -> {cand_dir(split)}", flush=True)


if __name__ == "__main__":
    # python run_blocking.py <split> [country]   (one country per process keeps RAM low)
    run(sys.argv[1], only=sys.argv[2] if len(sys.argv) > 2 else None)
