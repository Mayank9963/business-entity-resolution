"""Shared helper: candidate pairs -> feature matrix, in memory-bounded chunks."""
import polars as pl

import features as F
from config import norm_path

TEXT = ["entity_id", "country"] + F.TEXT_COLS


def load_norm(split: str, ids: pl.Series | None = None, sources=(1,),
              country: str | None = None) -> pl.DataFrame:
    lf = pl.concat([pl.scan_parquet(norm_path(split, s)).select(TEXT) for s in sources])
    if country is not None:
        lf = lf.filter(pl.col("country") == country)
    if ids is not None:
        lf = lf.filter(pl.col("entity_id").is_in(ids.implode()))
    return lf.collect()


def load_vocab(split: str) -> set:
    """Split-level S1 statistics used by features: the core-name token vocabulary
    (q_df0_frac) and core-name / core-name+legal frequencies (name ambiguity)."""
    s1 = pl.scan_parquet(norm_path(split, 1)).select("name_core", "legal").collect()
    vocab = set(s1["name_core"].str.split(" ").explode().unique().to_list())
    F.set_vocab(vocab)
    s1 = s1.with_columns(c=pl.col("name_core").str.replace_all(" ", ""))
    nf = s1.group_by("c").len()
    nlf = s1.group_by(pl.col("c") + "|" + pl.col("legal")).len()
    F.set_name_freq(dict(zip(nf["c"].to_list(), nf["len"].to_list())),
                    dict(zip(nlf["c"].to_list(), nlf["len"].to_list())))
    F.set_lratio(token_lratio(split))
    return vocab


def token_lratio(split: str, min_count: int = 30) -> dict:
    """Per country: log((share of S2/S3 names containing t) / (share of S1 names
    containing t)). Uses only the input records of the split (no labels)."""
    def doc_freq(sources):
        lf = pl.concat([pl.scan_parquet(norm_path(split, s)).select("country", "name_n")
                        for s in sources])
        n = lf.group_by("country").len("N").collect()
        t = (lf.select("country", t=pl.col("name_n").str.split(" ").list.unique())
             .explode("t").filter(pl.col("t") != "").group_by("country", "t").len("k")
             .collect())
        return t.join(n, on="country")
    q, s = doc_freq((2, 3)), doc_freq((1,))
    j = (q.filter(pl.col("k") >= min_count)
         .join(s.rename({"k": "ks", "N": "Ns"}), on=["country", "t"], how="left")
         .with_columns(pl.col("ks").fill_null(0),
                       pl.col("Ns").fill_null(pl.col("Ns").max().over("country")))
         .with_columns(lr=((pl.col("k") / pl.col("N"))
                           / ((pl.col("ks") + 1) / pl.col("Ns"))).log()))
    out = {}
    for c, t, lr in j.select("country", "t", "lr").iter_rows():
        out.setdefault(c, {})[t] = round(lr, 3)
    return out


def featurize(cand: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame,
              chunk: int = 250_000):
    """cand must be grouped by query id (all candidates of a query in one chunk).
    Yields (keys_df[id, s1_id], X) per chunk."""
    cand = cand.sort("id", "brank")
    n = cand.height
    start = 0
    while start < n:
        end = min(start + chunk, n)
        if end < n:  # extend to the end of the current query's group
            last = cand["id"][end - 1]
            while end < n and cand["id"][end] == last:
                end += 1
        part = cand.slice(start, end - start)
        p = F.attach_text(part, q, s1)
        yield p.select("id", "s1_id"), F.pair_features(p)
        start = end
