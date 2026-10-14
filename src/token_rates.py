"""Per-token match rates learned from the training labels.

The S2/S3 generators create decoy records by adding a word to a real business name
("X Holdings", "X Eastgate", "X Public Limited"): records containing such words are
almost never true matches, whereas "X formerly Y" / "X dba Y" records always are.
For every name / address token of the training S2/S3 records we store the smoothed
fraction of records containing it that have a true S1 match; features.py then looks up
the tokens a query ADDS relative to its S1 candidate.
"""
import json

import polars as pl

from config import ART, norm_path
from io_utils import read_ground_truth

PATH = ART / "token_rates.json"
MIN_COUNT = 30
SMOOTH = 20


def _rates(df: pl.DataFrame, col: str, prior: float) -> dict:
    t = (df.select("m", tok=pl.col(col).str.split(" ").list.unique()).explode("tok")
         .filter(pl.col("tok") != "")
         .group_by("tok").agg(n=pl.len(), k=pl.col("m").sum())
         .filter(pl.col("n") >= MIN_COUNT)
         .with_columns(rate=((pl.col("k") + SMOOTH * prior) / (pl.col("n") + SMOOTH))))
    return dict(zip(t["tok"].to_list(), t["rate"].round(4).to_list()))


def build(exclude_ids: pl.Series | None = None, country: str | None = None,
          save: bool = True) -> dict:
    gt = read_ground_truth().select("id").unique().with_columns(m=pl.lit(1, pl.Int8))
    q = pl.concat([pl.read_parquet(norm_path("train", s), columns=["entity_id", "country",
                                                                    "name_n", "addr_set"])
                   for s in (2, 3)]).rename({"entity_id": "id"})
    if country is not None:  # leave-one-country-out experiments
        q = q.filter(pl.col("country") == country)
    if exclude_ids is not None:
        q = q.filter(~pl.col("id").is_in(exclude_ids.implode()))
    q = q.join(gt, on="id", how="left").with_columns(pl.col("m").fill_null(0))
    prior = float(q["m"].mean())
    out = {"prior": prior, "name": _rates(q, "name_n", prior),
           "addr": _rates(q, "addr_set", prior)}
    if save:
        PATH.write_text(json.dumps(out))
    print(f"[token_rates] prior {prior:.3f}; name tokens {len(out['name']):,}, "
          f"address tokens {len(out['addr']):,}", flush=True)
    return out


def load() -> dict:
    return json.loads(PATH.read_text())
