"""Compare two submission files (e.g. v5 vs another model) without spending submissions.

    python compare.py <a/matching_results.tsv> <b/matching_results.tsv> [--examples 15]

Reports, per country: links in A, links in B, links in both, links only in A / only in B,
and S1 entities whose match lists differ. Prints sample disagreements with the
normalised names/addresses so you can judge which model is right.
Needs the normalised test files (run `python run_all.py --from normalize` or a full run).
"""
import argparse

import polars as pl

from config import norm_path


def load(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (df.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "id"})
            .with_columns(pl.col("id").fill_null("").str.split(","))
            .explode("id").filter(pl.col("id") != "").select("s1_id", "id"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--examples", type=int, default=15)
    args = ap.parse_args()
    a, b = load(args.a).with_columns(in_a=pl.lit(True)), load(args.b).with_columns(in_b=pl.lit(True))
    s1 = pl.read_parquet(norm_path("test", 1), columns=["entity_id", "country"]).rename(
        {"entity_id": "s1_id"})
    j = (a.join(b, on=["s1_id", "id"], how="full", coalesce=True)
         .with_columns(pl.col("in_a").fill_null(False), pl.col("in_b").fill_null(False))
         .join(s1, on="s1_id", how="left"))
    summary = j.group_by("country").agg(
        links_a=pl.col("in_a").sum(), links_b=pl.col("in_b").sum(),
        both=(pl.col("in_a") & pl.col("in_b")).sum(),
        only_a=(pl.col("in_a") & ~pl.col("in_b")).sum(),
        only_b=(~pl.col("in_a") & pl.col("in_b")).sum()).sort("country")
    summary = summary.with_columns(
        agreement=(pl.col("both") / (pl.col("both") + pl.col("only_a") + pl.col("only_b"))).round(4))
    per_s1 = j.group_by("s1_id", "country").agg(diff=(pl.col("in_a") != pl.col("in_b")).any())
    ent = per_s1.group_by("country").agg(s1_lists_differ=pl.col("diff").sum()).sort("country")
    with pl.Config(tbl_cols=-1, tbl_width_chars=200):
        print(summary.join(ent, on="country"))

    cols = ["entity_id", "name_n", "addr_n", "nums"]
    q = pl.concat([pl.scan_parquet(norm_path("test", s)).select(cols) for s in (2, 3)])
    s = pl.scan_parquet(norm_path("test", 1)).select(cols)
    for label, cond in (("ONLY IN A", pl.col("in_a") & ~pl.col("in_b")),
                        ("ONLY IN B", ~pl.col("in_a") & pl.col("in_b"))):
        ex = j.filter(cond).sample(min(args.examples, j.filter(cond).height), seed=1)
        qq = q.filter(pl.col("entity_id").is_in(ex["id"].implode())).collect()
        ss = s.filter(pl.col("entity_id").is_in(ex["s1_id"].implode())).collect()
        ex = (ex.join(qq.rename({c: "q_" + c for c in cols}), left_on="id", right_on="q_entity_id")
              .join(ss.rename({c: "s_" + c for c in cols}), left_on="s1_id", right_on="s_entity_id"))
        print(f"\n== {label} (sample)")
        for r in ex.iter_rows(named=True):
            print(f"  [{r['country']}] {r['q_name_n']} / {r['q_addr_n']} {r['q_nums']}"
                  f"  ||  S1: {r['s_name_n']} / {r['s_addr_n']} {r['s_nums']}")


if __name__ == "__main__":
    main()
