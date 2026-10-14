"""Macro F_0.5 exactly as defined in the challenge README."""
import polars as pl

BETA2 = 0.25


def macro_f05(pred: pl.DataFrame, truth: pl.DataFrame, s1_ids) -> dict:
    """pred / truth: long (s1_id, id) pairs. s1_ids: the S1 entities to average over."""
    base = pl.DataFrame({"s1_id": pl.Series(list(s1_ids), dtype=pl.String)})
    p = pred.select("s1_id", "id").unique().with_columns(pl.lit(1).alias("p"))
    t = truth.select("s1_id", "id").unique().with_columns(pl.lit(1).alias("t"))
    j = p.join(t, on=["s1_id", "id"], how="full", coalesce=True).fill_null(0)
    per = j.group_by("s1_id").agg(tp=(pl.col("p") * pl.col("t")).sum(),
                                  np_=pl.col("p").sum(), nt=pl.col("t").sum())
    per = base.join(per, on="s1_id", how="left").fill_null(0)
    per = per.with_columns(
        prec=pl.when(pl.col("np_") > 0).then(pl.col("tp") / pl.col("np_")).otherwise(None),
        rec=pl.when(pl.col("nt") > 0).then(pl.col("tp") / pl.col("nt")).otherwise(None))
    f = (pl.when((pl.col("np_") == 0) & (pl.col("nt") == 0)).then(1.0)
         .when((pl.col("np_") == 0) | (pl.col("nt") == 0) | (pl.col("tp") == 0)).then(0.0)
         .otherwise((1 + BETA2) * pl.col("prec") * pl.col("rec")
                    / (BETA2 * pl.col("prec") + pl.col("rec"))))
    per = per.with_columns(f=f)
    tot = per.select(pl.col("tp").sum(), pl.col("np_").sum(), pl.col("nt").sum()).row(0)
    return {
        "macro_f05": per["f"].mean(),
        "singleton_f": per.filter(pl.col("nt") == 0)["f"].mean(),
        "nonsingleton_f": per.filter(pl.col("nt") > 0)["f"].mean(),
        "micro_precision": tot[0] / max(tot[1], 1),
        "micro_recall": tot[0] / max(tot[2], 1),
        "n_s1": per.height,
    }
