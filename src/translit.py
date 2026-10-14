"""Learn Indic-script -> Latin mappings from the TRAINING ground truth only.

* Name tokens: S2/S3 names written in Devanagari (e.g. "प्राइवेट लिमिटेड") are aligned
  position-by-position with the Latin S1 name of the entity they belong to, and the
  most frequent Latin token per Indic token is kept ("प्राइवेट" -> "private").
* Address segments: comma-separated segments in a native script (mostly state names
  such as "தமிழ்நாடு") are mapped to the S1 address segment they co-occur with most.

No external data is used; unseen tokens fall back to unidecode in normalize.py.
"""
import json
import re
from collections import Counter, defaultdict

import polars as pl

from config import ART
from io_utils import read_ground_truth, read_source

INDIC = re.compile(r"[ऀ-෿]")
PUNCT = re.compile(r"[!-/:-@\[-`{-~।॥]")  # ASCII punctuation + danda

NAME_MAP_PATH = ART / "indic_name_tokens.json"
SEG_MAP_PATH = ART / "indic_addr_segments.json"


def _tok(s: str):
    return PUNCT.sub(" ", s.lower()).split()


def learn():
    gt = read_ground_truth()
    s1 = read_source("train", 1).filter(pl.col("country") == "India").select(
        pl.col("entity_id").alias("s1_id"), pl.col("business_name").alias("n1"),
        pl.col("business_address").alias("a1"))
    others = pl.concat([read_source("train", s) for s in (2, 3)]).filter(
        pl.col("country") == "India").select(
        pl.col("entity_id").alias("id"), pl.col("business_name").alias("n2"),
        pl.col("business_address").alias("a2"))
    others = others.filter(pl.col("n2").str.contains(r"[ऀ-෿]")
                           | pl.col("a2").str.contains(r"[ऀ-෿]"))
    pairs = others.join(gt, on="id").join(s1, on="s1_id")
    print(f"[translit] aligned pairs with Indic text: {pairs.height:,}")

    tok_counts = defaultdict(Counter)
    seg_counts = defaultdict(Counter)
    for n1, a1, n2, a2 in pairs.select("n1", "a1", "n2", "a2").iter_rows():
        if INDIC.search(n2):
            t2, t1 = _tok(n2), _tok(n1)
            if len(t1) == len(t2):
                for x, y in zip(t2, t1):
                    if INDIC.search(x) and not INDIC.search(y):
                        tok_counts[x][y] += 1
        if INDIC.search(a2):
            segs1 = [s.strip().lower() for s in a1.split(",") if s.strip()]
            segs1 = [s for s in segs1 if not any(c.isdigit() for c in s)]
            for seg in a2.split(","):
                seg = seg.strip().strip('"').strip()
                if INDIC.search(seg):
                    for y in segs1:
                        seg_counts[seg][y] += 1

    def best(counts, min_n=2, min_share=0.4):
        out = {}
        for k, c in counts.items():
            (y, n), tot = c.most_common(1)[0], sum(c.values())
            if n >= min_n and n / tot >= min_share:
                out[k] = y
        return out

    name_map, seg_map = best(tok_counts), best(seg_counts, min_share=0.2)
    NAME_MAP_PATH.write_text(json.dumps(name_map, ensure_ascii=False), encoding="utf-8")
    SEG_MAP_PATH.write_text(json.dumps(seg_map, ensure_ascii=False), encoding="utf-8")
    print(f"[translit] name tokens learned: {len(name_map):,}  address segments: {len(seg_map):,}")
    return name_map, seg_map


def load():
    return (json.loads(NAME_MAP_PATH.read_text(encoding="utf-8")),
            json.loads(SEG_MAP_PATH.read_text(encoding="utf-8")))


if __name__ == "__main__":
    learn()
