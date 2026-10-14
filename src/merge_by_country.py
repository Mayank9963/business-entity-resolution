"""Combine submission files country by country (e.g. c15_ce for US/India, v5 for France).

    python merge_by_country.py --file v5=../output/matching_results.tsv \
                               --file c15=path/to/c15/matching_results.tsv \
                               --pick US=c15 --pick India=c15 --pick France=v5 \
                               --out ../output/hybrid/matching_results.tsv

--pick COUNTRY=NAME       take that country's rows from file NAME
--pick COUNTRY=intersect  keep only links present in every file
--pick COUNTRY=union      keep links present in any file; a record claimed by two S1
                          entities keeps the one from the first --file listed
Records never cross countries, so "one record -> at most one S1" still holds after a
per-country merge. Rows are written in test_source1 order; run the validator after.
"""
import argparse
from pathlib import Path

import polars as pl

from config import DATA


def load(path: str) -> pl.DataFrame:
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return (df.rename({"source1_entity_id": "s1_id", "matched_entity_ids": "id"})
            .with_columns(pl.col("id").fill_null("").str.split(","))
            .explode("id").filter(pl.col("id") != "").select("s1_id", "id"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", action="append", required=True, help="NAME=path")
    ap.add_argument("--pick", action="append", required=True, help="COUNTRY=NAME|intersect|union")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = dict(x.split("=", 1) for x in a.file)
    picks = dict(x.split("=", 1) for x in a.pick)
    s1 = pl.read_csv(DATA / "test" / "test_source1.tsv", separator="\t", quote_char=None,
                     infer_schema=False, columns=["entity_id", "country"])
    missing = set(s1["country"].unique()) - set(picks)
    if missing:
        raise SystemExit(f"no --pick for countries: {sorted(missing)}")
    pairs = {n: load(p).join(s1.rename({"entity_id": "s1_id"}), on="s1_id", how="left")
             for n, p in files.items()}
    parts = []
    for country, how in picks.items():
        per = {n: d.filter(pl.col("country") == country).select("s1_id", "id")
               for n, d in pairs.items()}
        if how == "intersect":
            names = list(per)
            sel = per[names[0]]
            for n in names[1:]:
                sel = sel.join(per[n], on=["s1_id", "id"], how="semi")
        elif how == "union":
            sel = pl.concat([d.with_columns(pri=pl.lit(i)) for i, d in enumerate(per.values())])
            sel = sel.sort("pri").unique(subset=["id"], keep="first").select("s1_id", "id")
        else:
            sel = per[how]
        parts.append(sel)
        n_s1 = s1.filter(pl.col("country") == country).height
        print(f"{country:8s} <- {how:10s}: {sel.height:,} links ({sel.height / n_s1:.2f} per S1)")
    allp = pl.concat(parts)
    dup = allp.group_by("id").len().filter(pl.col("len") > 1).height
    if dup:
        raise SystemExit(f"{dup} records linked to two S1 entities - check the inputs")
    g = allp.group_by("s1_id").agg(pl.col("id").unique().sort().str.join(","))
    rows = (s1.select(pl.col("entity_id").alias("s1_id")).join(g, on="s1_id", how="left",
                                                               maintain_order="left")
            .with_columns(pl.col("id").fill_null("")))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("".join(f"{x}\t{y}\n" for x, y in rows.iter_rows()))
    print(f"wrote {out} ({rows.height:,} rows, {allp.height:,} links)")


if __name__ == "__main__":
    main()
