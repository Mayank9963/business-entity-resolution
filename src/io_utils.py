import polars as pl

from config import DATA, src_path


def read_source(split: str, source: int) -> pl.DataFrame:
    df = pl.read_csv(src_path(split, source), separator="\t", quote_char=None,
                     infer_schema=False)
    return df.with_columns(
        pl.col("business_name").fill_null(""),
        pl.col("business_address").fill_null("").str.replace(r"^None$", ""),
        pl.col("country").fill_null(""),
    )


def read_ground_truth() -> pl.DataFrame:
    """Long format: one row per (s1_id, other_id) true pair."""
    gt = pl.read_csv(DATA / "train" / "train_ground_truth.tsv", separator="\t",
                     quote_char=None, infer_schema=False)
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "id"}))


def write_id_lists(df: pl.DataFrame, all_s1: pl.Series, path, col_name: str):
    """df: (s1_id, id) pairs -> TSV with one row per S1 entity, comma-joined ids."""
    grouped = df.group_by("s1_id").agg(pl.col("id").unique().sort().str.join(","))
    out = (pl.DataFrame({"s1_id": all_s1}).join(grouped, on="s1_id", how="left")
             .with_columns(pl.col("id").fill_null(""))
             .rename({"s1_id": "source1_entity_id", "id": col_name}))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(f"source1_entity_id\t{col_name}\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
