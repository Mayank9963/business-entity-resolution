"""Candidate generation.

Every S2/S3 record belongs to at most one S1 entity (verified on train), so blocking is
run from the S2/S3 side: each S2/S3 record is a *query* against an inverted index of the
S1 records of the same country. Blocking keys:

  n:<token>        core-name token
  k:<skeleton>     phonetic consonant skeleton of a core-name token (typos, transliteration)
  p:<tok>|<tok>    sorted pair of core-name tokens (selective even when tokens are common)
  c:<concat>       core name without spaces ("beaconbiotechnologies.com")
  a:<num>|<word>   address house-number + address word
  x:<tok>|<word>   core-name token + address word (selective even for generic names)
  z:<postcode>
  b:<word>|<word>  pair of address words (invented names / no house number)
  m:<num>|<num>    pair of house numbers (survives street-name typos)
  f:<prefix7>      first 7 letters of the space-free core name ("rockyadamas")

A pair's blocking score is the sum of IDF weights of the keys it shares; the top-K S1
records per query are kept. Keys whose S1 document frequency exceeds MAX_DF are dropped.
"""
import time

import numpy as np
import polars as pl

from normalize import skeleton

MAX_DF = {"n": 500, "k": 300, "p": 1500, "c": 5000, "a": 150, "z": 300, "x": 300,
          "b": 150, "m": 100, "f": 300}
MAX_NAME_TOKS = 6


def record_keys(name_core: str, addr_set: str, nums: str, postcode: str):
    toks = [t for t in name_core.split() if len(t) >= 2]
    keys = set()
    for t in toks:
        keys.add("n:" + t)
        sk = skeleton(t)
        if len(sk) >= 2:
            keys.add("k:" + sk)
    tt = sorted(set(toks[:MAX_NAME_TOKS]))
    for i in range(len(tt)):
        for j in range(i + 1, len(tt)):
            keys.add("p:" + tt[i] + "|" + tt[j])
    concat = "".join(name_core.split())
    if len(concat) >= 4:
        keys.add("c:" + concat)
    words = [w for w in addr_set.split() if len(w) >= 3][:8]
    for n in nums.split()[:3]:
        if n != "0" and len(n) <= 5:
            for w in words:
                keys.add("a:" + n + "|" + w)
    for t in toks[:4]:
        for w in words[:6]:
            keys.add("x:" + t + "|" + w)
    if postcode:
        keys.add("z:" + postcode)
    if len(concat) >= 7:
        keys.add("f:" + concat[:7])
    ww = words[:6]
    for i in range(len(ww)):
        for j in range(i + 1, len(ww)):
            keys.add("b:" + ww[i] + "|" + ww[j])
    nn = [n for n in nums.split() if len(n) >= 2][:4]
    for i in range(len(nn)):
        for j in range(i + 1, len(nn)):
            keys.add("m:" + nn[i] + "|" + nn[j])
    return list(keys)


def key_table(df: pl.DataFrame, idx_col: str, chunk: int = 100_000) -> pl.DataFrame:
    """(idx, h = u64 key hash, kt = key type). Built in chunks so the Python key strings
    never exist for more than `chunk` records at a time."""
    parts = []
    for start in range(0, df.height, chunk):
        sl = df.slice(start, chunk)
        keys = [record_keys(*r) for r in sl.select("name_core", "addr_set", "nums",
                                                     "postcode").iter_rows()]
        t = pl.DataFrame({idx_col: sl[idx_col], "key": keys}).explode("key").drop_nulls()
        parts.append(t.select(idx_col, pl.col("key").hash().alias("h"),
                              pl.col("key").str.slice(0, 1).cast(pl.Categorical).alias("kt")))
        del keys, t
    return pl.concat(parts)


def build_index(s1: pl.DataFrame) -> pl.DataFrame:
    """Inverted index over one country's S1 rows: (h = key hash, s1_idx, w = idf)."""
    ks = key_table(s1.with_row_index("s1_idx"), "s1_idx").unique()
    df = ks.group_by("h", "kt").len("df")
    cap = pl.col("kt").cast(pl.String).replace_strict(MAX_DF, return_dtype=pl.UInt32)
    df = df.filter(pl.col("df") <= cap).with_columns(
        w=(np.log(s1.height / pl.col("df"))).cast(pl.Float32))
    return ks.join(df.select("h", "w"), on="h").select("h", "s1_idx", "w")


def query_chunks(index: pl.DataFrame, q: pl.DataFrame, k: int, chunk: int = 25_000):
    """Yield (q_idx, s1_idx, bscore, brank) frames, top-k S1 rows per query row."""
    for start in range(0, q.height, chunk):
        part = key_table(q.slice(start, chunk).with_row_index("q_idx", offset=start),
                         "q_idx").drop("kt")
        yield (part.join(index, on="h")
               .group_by("q_idx", "s1_idx").agg(pl.col("w").sum().alias("bscore"))
               .sort(["q_idx", "bscore", "s1_idx"], descending=[False, True, False])
               .with_columns(brank=pl.col("s1_idx").cum_count().over("q_idx").cast(pl.Int16))
               .filter(pl.col("brank") <= k)
               .with_columns(pl.col("q_idx").cast(pl.UInt32), pl.col("s1_idx").cast(pl.UInt32)))


def block_country(s1: pl.DataFrame, q: pl.DataFrame, k: int, verbose: bool = True):
    t0 = time.time()
    index = build_index(s1)
    if verbose:
        print(f"  [block] index: {index.height:,} postings ({time.time() - t0:.0f}s)", flush=True)
    res = pl.concat(list(query_chunks(index, q, k)))
    if verbose:
        print(f"  [block] {q.height:,} queries -> {res.height:,} pairs "
              f"({time.time() - t0:.0f}s)", flush=True)
    return res
