"""Stage 5: score test candidates and write the two submission files.

    python predict.py score <split> <country>   # one country (several can run in parallel)
    python predict.py write <split>             # assemble output/*.tsv from the scores
    python predict.py all <split>               # both, sequentially

output/candidate_pairs.tsv  - every (S1, candidate) pair the model scores
output/matching_results.tsv - each S2/S3 record assigned to its best S1 candidate
                              when the model probability >= tuned threshold
Memory stays bounded: candidates are streamed by parquet row group (each row group
holds complete queries), and the TSVs are built per country in hash buckets of S1.
"""
import json
import sys
import time

import lightgbm as lgb
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

import token_rates
from config import OUTPUT, WORK
from features import set_token_rates
from featurize import featurize, load_norm, load_vocab
from run_blocking import cand_dir, countries
from train import META_PATH, MODEL_PATH, MODEL_UNSEEN_PATH

ROWS_PER_STEP = 1_500_000
N_BUCKETS = 8


def model_for(country: str, meta: dict):
    """(booster, feature list, best_iteration, threshold) for a country: the seen-country
    model for countries present in the training labels, the label-free model otherwise."""
    if country in meta.get("train_countries", []) or "unseen" not in meta:
        return (lgb.Booster(model_file=str(MODEL_PATH)), meta["features"],
                meta["best_iteration"], meta["threshold"])
    u = meta["unseen"]
    return (lgb.Booster(model_file=str(MODEL_UNSEEN_PATH)), u["features"],
            u["best_iteration"], u["threshold"])


def best_path(split: str, country: str):
    return WORK / f"best_{split}_{country}.parquet"


def score(split: str, country: str):
    """Best candidate per query (id, s1_id, prob) -> work/best_<split>_<country>.parquet"""
    t0 = time.time()
    out = best_path(split, country)
    path = cand_dir(split) / f"{country}.parquet"
    if not path.exists():
        print(f"[predict] {country}: no candidates", flush=True)
        return
    meta = json.loads(META_PATH.read_text())
    booster, feats, n_iter, thr = model_for(country, meta)
    load_vocab(split)
    set_token_rates(token_rates.load())
    q = load_norm(split, sources=(2, 3), country=country)
    s1 = load_norm(split, sources=(1,), country=country)
    pf = pq.ParquetFile(path)
    best, buf, nbuf = [], [], 0

    def flush():
        cand = pl.from_arrow(pa.concat_tables(buf))
        for keys, X in featurize(cand, q, s1):
            prob = booster.predict(X.select(feats).to_numpy(), num_iteration=n_iter)
            d = keys.with_columns(prob=pl.Series(prob, dtype=pl.Float32))
            best.append(d.sort("prob", descending=True).group_by("id").first())

    for i in range(pf.num_row_groups):
        rg = pf.read_row_group(i)
        buf.append(rg)
        nbuf += rg.num_rows
        if nbuf >= ROWS_PER_STEP:
            flush()
            buf, nbuf = [], 0
    if buf:
        flush()
    res = pl.concat(best).with_columns(thr=pl.lit(thr, pl.Float32))
    res.write_parquet(out)
    kind = "seen" if country in meta.get("train_countries", []) else "UNSEEN (label-free)"
    print(f"[predict] {country}: {kind} model, threshold {thr}; scored {res.height:,} queries, "
          f"{(res['prob'] >= thr).sum():,} linked ({time.time() - t0:.0f}s)", flush=True)


def _write_tsv(f, s1_ids: pl.Series, pairs: pl.LazyFrame):
    """pairs: lazy (s1_id, id). Written in S1-hash buckets to bound memory."""
    base = pl.DataFrame({"s1_id": s1_ids})
    for b in range(N_BUCKETS):
        sel = base.filter(pl.col("s1_id").hash(seed=7) % N_BUCKETS == b)
        g = (pairs.filter(pl.col("s1_id").is_in(sel["s1_id"].implode()))
             .group_by("s1_id").agg(pl.col("id").unique().sort().str.join(","))
             .collect())
        rows = sel.join(g, on="s1_id", how="left").with_columns(pl.col("id").fill_null(""))
        f.write("".join(f"{a}\t{b_}\n" for a, b_ in rows.iter_rows()))


def write(split: str = "test"):
    t0 = time.time()
    s1_meta = load_norm(split, sources=(1,)).select("entity_id", "country")
    fm = open(OUTPUT / "matching_results.tsv", "w", encoding="utf-8", newline="")
    fc = open(OUTPUT / "candidate_pairs.tsv", "w", encoding="utf-8", newline="")
    fm.write("source1_entity_id\tmatched_entity_ids\n")
    fc.write("source1_entity_id\tcandidate_entity_ids\n")
    n_match = 0
    for country in s1_meta["country"].unique().sort().to_list():
        s1_ids = s1_meta.filter(pl.col("country") == country)["entity_id"]
        cpath, bpath = cand_dir(split) / f"{country}.parquet", best_path(split, country)
        if cpath.exists() != bpath.exists():
            raise RuntimeError(f"{country}: candidates and scores out of sync - run score")
        if cpath.exists():
            matches = pl.read_parquet(bpath).filter(pl.col("prob") >= pl.col("thr"))
            cand_lazy = pl.scan_parquet(cpath).select("s1_id", "id")
        else:  # a country with no candidates at all
            matches = pl.DataFrame(schema={"s1_id": pl.String, "id": pl.String})
            cand_lazy = matches.lazy()
        _write_tsv(fm, s1_ids, matches.lazy().select("s1_id", "id"))
        _write_tsv(fc, s1_ids, cand_lazy)
        n_match += matches.height
        print(f"[predict] {country}: {s1_ids.len():,} S1 rows, {matches.height:,} matched "
              f"records", flush=True)
    fm.close()
    fc.close()
    print(f"[predict] wrote output/: {n_match:,} matched records ({time.time() - t0:.0f}s)",
          flush=True)


if __name__ == "__main__":
    cmd, split = sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "test"
    if cmd == "score":
        score(split, sys.argv[3])
    elif cmd == "write":
        write(split)
    elif cmd == "all":
        for c in countries(split):
            score(split, c)
        write(split)
    else:
        raise SystemExit(__doc__)
