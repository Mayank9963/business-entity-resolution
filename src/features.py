"""Pair features for (query = S2/S3 record, candidate = S1 record).

All features are country-agnostic similarities, so the model transfers to countries
not seen in training (France). Country itself is deliberately NOT a feature.
"""
import numpy as np
import polars as pl
from rapidfuzz import process
from rapidfuzz.distance import JaroWinkler, Levenshtein
from rapidfuzz.fuzz import partial_ratio, ratio, token_set_ratio, token_sort_ratio

from normalize import skeleton

TEXT_COLS = ["name_n", "name_core", "name_sorted", "legal", "name_alt", "synth", "altflag",
             "addr_n",
             "addr_set", "nums", "postcode", "state"]


FEATURES = [
    "nm_ratio", "nm_tsort", "nm_tset", "nm_partial", "nm_jw", "nm_sorted_lev",
    "nm_concat_ratio", "nm_concat_partial", "nm_full_tset", "nm_skel_tset", "nm_alt_tset",
    "nm_jacc", "nm_inter", "nm_q_ntok", "nm_s_ntok", "nm_len_diff", "nm_first_eq",
    "lg_q_has", "lg_s_has", "lg_jacc", "ad_q_empty", "ad_s_empty", "ad_tset", "ad_tsort",
    "ad_partial", "ad_jacc", "ad_inter", "ad_skel_tset", "num_jacc", "num_inter", "num_q_n",
    "num_s_n", "num_tset", "post_eq", "state_eq", "bscore", "brank", "q_src3",
    "bscore_rel", "nm_tset_gap", "ad_tset_gap", "q_ncand",
    # v2
    "q_synth", "s_synth", "q_core_empty", "nm_sub_q", "nm_sub_s", "nm_alt_sub",
    "nm_initials", "q_df0_frac", "num_near", "ad_tset_rank", "nm_tset_rank",
    # v3: learned decoy-word rates of tokens the query ADDS relative to the candidate
    "q_altflag", "tok_extra_min", "tok_extra_nlow", "tok_all_min", "atok_extra_min",
    # v5: how ambiguous is the name among S1 entities (unique name -> safe to link)
    "s_name_freq", "s_name_legal_freq", "q_name_freq", "q_name_legal_freq", "legal_eq",
    # v5: label-free decoy signal - how over-represented a token is in S2/S3 names vs S1
    # names of the same country (decoys = S1 name + added word); works for unseen France
    "tok_extra_lratio_max", "tok_all_lratio_max",
]

_LRATIO: dict = {}


def set_lratio(table: dict):
    """{country: {token: log over-representation ratio}}."""
    global _LRATIO
    _LRATIO = table


def _extra_lratio(q: str, s: str, table: dict) -> float:
    st = set(s.split())
    rs = [table[t] for t in set(q.split()) if t not in st and t in table]
    return max(rs) if rs else np.nan

_NAME_FREQ: dict = {}
_NAME_LEGAL_FREQ: dict = {}


def set_name_freq(name_freq: dict, name_legal_freq: dict):
    """Counts of S1 records per space-free core name (and per core name + legal form)."""
    global _NAME_FREQ, _NAME_LEGAL_FREQ
    _NAME_FREQ, _NAME_LEGAL_FREQ = name_freq, name_legal_freq

_RATES = {"prior": 0.74, "name": {}, "addr": {}}


def set_token_rates(rates: dict):
    global _RATES
    _RATES = rates


def _extra_rates(q: str, s: str, table: dict):
    """(min rate over query tokens absent from s, #such tokens with rate < 0.3)."""
    st = set(s.split())
    rs = [table[t] for t in set(q.split()) if t not in st and t in table]
    if not rs:
        return np.nan, 0.0
    return min(rs), float(sum(r < 0.3 for r in rs))


def _substr_cov(a: str, b_concat: str) -> float:
    """Fraction of tokens (len >= 3) of `a` that occur inside the space-free string b."""
    toks = [t for t in a.split() if len(t) >= 3]
    if not toks or not b_concat:
        return np.nan
    return sum(t in b_concat for t in toks) / len(toks)


def _initials(a: str, b: str) -> float:
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return np.nan
    ia, ib = "".join(t[0] for t in ta), "".join(t[0] for t in tb)
    ca, cb = "".join(ta), "".join(tb)
    return float((len(ta) >= 2 and ia == cb) or (len(tb) >= 2 and ib == ca))


def _num_near(a: str, b: str) -> float:
    """Number of house-number pairs (len >= 2) that differ by exactly one digit."""
    if not a or not b:
        return np.nan
    na = [x for x in a.split() if len(x) >= 2]
    nb = set(x for x in b.split() if len(x) >= 2)
    n = 0
    for x in na:
        if x in nb:
            continue
        for y in nb:
            if len(x) == len(y) and sum(c1 != c2 for c1, c2 in zip(x, y)) == 1:
                n += 1
                break
    return float(n)


_VOCAB: set = set()


def set_vocab(vocab: set):
    """S1 name-token vocabulary of the split being scored (for q_df0_frac)."""
    global _VOCAB
    _VOCAB = vocab


def _df0(a: str) -> float:
    toks = a.split()
    if not toks:
        return np.nan
    return sum(t not in _VOCAB for t in toks) / len(toks)


def _cp(a, b, scorer, **kw):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32, **kw)


def _skel_str(s: pl.Series) -> list:
    return [" ".join(skeleton(t) for t in x.split()) for x in s.to_list()]


def _jaccard(df: pl.DataFrame, a: str, b: str) -> pl.Series:
    la, lb = pl.col(a).str.split(" "), pl.col(b).str.split(" ")
    inter = la.list.set_intersection(lb).list.len()
    union = la.list.set_union(lb).list.len()
    empty = (pl.col(a) == "") | (pl.col(b) == "")
    return df.select(pl.when(empty).then(None).otherwise(inter / union).cast(pl.Float32)).to_series()


def _inter(df: pl.DataFrame, a: str, b: str) -> pl.Series:
    la, lb = pl.col(a).str.split(" "), pl.col(b).str.split(" ")
    return df.select(pl.when((pl.col(a) == "") | (pl.col(b) == "")).then(None)
                     .otherwise(la.list.set_intersection(lb).list.len())
                     .cast(pl.Float32)).to_series()


def pair_features(p: pl.DataFrame) -> pl.DataFrame:
    """p has columns q_<col> and s_<col> for every TEXT_COLS entry plus bscore, brank,
    q_src. Returns a float32 feature frame (same row order)."""
    f = {}
    qn, sn = p["q_name_core"].to_list(), p["s_name_core"].to_list()
    f["nm_ratio"] = _cp(qn, sn, ratio)
    f["nm_tsort"] = _cp(qn, sn, token_sort_ratio)
    f["nm_tset"] = _cp(qn, sn, token_set_ratio)
    f["nm_partial"] = _cp(qn, sn, partial_ratio)
    f["nm_jw"] = _cp(qn, sn, JaroWinkler.normalized_similarity)
    qc = [x.replace(" ", "") for x in p["q_name_sorted"].to_list()]
    sc = [x.replace(" ", "") for x in p["s_name_sorted"].to_list()]
    f["nm_sorted_lev"] = _cp(qc, sc, Levenshtein.normalized_similarity)
    qcat = [x.replace(" ", "") for x in qn]
    scat = [x.replace(" ", "") for x in sn]
    f["nm_concat_ratio"] = _cp(qcat, scat, ratio)
    f["nm_concat_partial"] = _cp(qcat, scat, partial_ratio)
    f["nm_full_tset"] = _cp(p["q_name_n"].to_list(), p["s_name_n"].to_list(), token_set_ratio)
    f["nm_skel_tset"] = _cp(p["q_name_skel"].to_list(), p["s_name_skel"].to_list(),
                            token_set_ratio)
    alt = p["q_name_alt"].to_list()
    f["nm_alt_tset"] = np.where(np.array([a != "" for a in alt]),
                                _cp(alt, sn, token_set_ratio), np.nan).astype(np.float32)
    f["nm_jacc"] = _jaccard(p, "q_name_core", "s_name_core").to_numpy()
    f["nm_inter"] = _inter(p, "q_name_core", "s_name_core").to_numpy()
    f["nm_q_ntok"] = p["q_name_core"].str.count_matches(" ").cast(pl.Float32).to_numpy() + 1
    f["nm_s_ntok"] = p["s_name_core"].str.count_matches(" ").cast(pl.Float32).to_numpy() + 1
    f["nm_len_diff"] = np.abs(np.array([len(x) for x in qcat], dtype=np.float32)
                              - np.array([len(x) for x in scat], dtype=np.float32))
    f["nm_first_eq"] = np.array([(a.split(" ")[0] == b.split(" ")[0]) if a and b else False
                                 for a, b in zip(qn, sn)], dtype=np.float32)

    ql, sl = p["q_legal"], p["s_legal"]
    f["lg_q_has"] = (ql != "").cast(pl.Float32).to_numpy()
    f["lg_s_has"] = (sl != "").cast(pl.Float32).to_numpy()
    f["lg_jacc"] = _jaccard(p, "q_legal", "s_legal").to_numpy()

    qa, sa = p["q_addr_n"].to_list(), p["s_addr_n"].to_list()
    has_a = np.array([bool(a) and bool(b) for a, b in zip(qa, sa)])
    f["ad_q_empty"] = np.array([not a for a in qa], dtype=np.float32)
    f["ad_s_empty"] = np.array([not b for b in sa], dtype=np.float32)
    for nm, sc_ in [("ad_tset", token_set_ratio), ("ad_tsort", token_sort_ratio),
                    ("ad_partial", partial_ratio)]:
        f[nm] = np.where(has_a, _cp(qa, sa, sc_), np.nan).astype(np.float32)
    f["ad_jacc"] = _jaccard(p, "q_addr_set", "s_addr_set").to_numpy()
    f["ad_inter"] = _inter(p, "q_addr_set", "s_addr_set").to_numpy()
    f["ad_skel_tset"] = np.where(has_a, _cp(p["q_addr_skel"].to_list(),
                                            p["s_addr_skel"].to_list(), token_set_ratio),
                                 np.nan).astype(np.float32)
    f["num_jacc"] = _jaccard(p, "q_nums", "s_nums").to_numpy()
    f["num_inter"] = _inter(p, "q_nums", "s_nums").to_numpy()
    f["num_q_n"] = p["q_nums"].str.split(" ").list.len().cast(pl.Float32).to_numpy() * (p["q_nums"] != "").cast(pl.Float32).to_numpy()
    f["num_s_n"] = p["s_nums"].str.split(" ").list.len().cast(pl.Float32).to_numpy() * (p["s_nums"] != "").cast(pl.Float32).to_numpy()
    f["num_tset"] = _cp(p["q_nums"].to_list(), p["s_nums"].to_list(), token_set_ratio)

    def eq_or_missing(a, b):
        return p.select(pl.when((pl.col(a) == "") | (pl.col(b) == "")).then(None)
                        .otherwise(pl.col(a) == pl.col(b)).cast(pl.Float32)).to_series().to_numpy()
    f["post_eq"] = eq_or_missing("q_postcode", "s_postcode")
    f["state_eq"] = eq_or_missing("q_state", "s_state")

    f["bscore"] = p["bscore"].cast(pl.Float32).to_numpy()
    f["brank"] = p["brank"].cast(pl.Float32).to_numpy()
    f["q_src3"] = (p["q_src"] == 3).cast(pl.Float32).to_numpy()
    out = pl.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})
    # query-level context: how this candidate compares with the query's other candidates
    ctx = pl.DataFrame({"q": p["q_key"], "bscore": f["bscore"], "nm_tset": f["nm_tset"],
                        "ad_tset": np.nan_to_num(f["ad_tset"], nan=0.0)})
    ctx = ctx.with_columns(
        (pl.col("bscore") / pl.col("bscore").max().over("q")).alias("bscore_rel"),
        (pl.col("nm_tset") - pl.col("nm_tset").max().over("q")).alias("nm_tset_gap"),
        (pl.col("ad_tset") - pl.col("ad_tset").max().over("q")).alias("ad_tset_gap"),
        pl.len().over("q").cast(pl.Float32).alias("q_ncand"),
    )
    out = out.with_columns(ctx.select("bscore_rel", "nm_tset_gap", "ad_tset_gap",
                                      "q_ncand").cast(pl.Float32))
    # ---- v2 features
    qalt = p["q_name_alt"].to_list()
    qnums, snums = p["q_nums"].to_list(), p["s_nums"].to_list()
    v2 = {
        "q_synth": p["q_synth"].cast(pl.Float32).to_numpy(),
        "s_synth": p["s_synth"].cast(pl.Float32).to_numpy(),
        "q_core_empty": np.array([not a for a in qn], dtype=np.float32),
        "nm_sub_q": np.array([_substr_cov(b, a) for a, b in zip(qcat, sn)], dtype=np.float32),
        "nm_sub_s": np.array([_substr_cov(a, b) for a, b in zip(qn, scat)], dtype=np.float32),
        "nm_alt_sub": np.array([_substr_cov(b, a.replace(" ", "")) if a else np.nan
                                for a, b in zip(qalt, sn)], dtype=np.float32),
        "nm_initials": np.array([_initials(a, b) for a, b in zip(qn, sn)], dtype=np.float32),
        "q_df0_frac": np.array([_df0(a) for a in qn], dtype=np.float32),
        "num_near": np.array([_num_near(a, b) for a, b in zip(qnums, snums)], dtype=np.float32),
    }
    rk = pl.DataFrame({"q": p["q_key"], "a": np.nan_to_num(f["ad_tset"], nan=-1.0),
                       "n": f["nm_tset"]}).select(
        pl.col("a").rank("min", descending=True).over("q").cast(pl.Float32).alias("ad_tset_rank"),
        pl.col("n").rank("min", descending=True).over("q").cast(pl.Float32).alias("nm_tset_rank"))
    out = out.with_columns(**{k: pl.Series(v) for k, v in v2.items()}).with_columns(rk)
    # ---- v3 features
    nt, at = _RATES["name"], _RATES["addr"]
    qfull, sfull = p["q_name_n"].to_list(), p["s_name_n"].to_list()
    ex = [_extra_rates(a, b, nt) for a, b in zip(qfull, sfull)]
    aex = [_extra_rates(a, b, at)[0] for a, b in
           zip(p["q_addr_set"].to_list(), p["s_addr_set"].to_list())]
    v3 = {
        "q_altflag": p["q_altflag"].cast(pl.Float32).to_numpy(),
        "tok_extra_min": np.array([e[0] for e in ex], dtype=np.float32),
        "tok_extra_nlow": np.array([e[1] for e in ex], dtype=np.float32),
        "tok_all_min": np.array([min((nt[t] for t in a.split() if t in nt), default=np.nan)
                                 for a in qfull], dtype=np.float32),
        "atok_extra_min": np.array(aex, dtype=np.float32),
    }
    out = out.with_columns(**{k: pl.Series(v) for k, v in v3.items()})
    # ---- v5 features
    ql, sl = p["q_legal"].to_list(), p["s_legal"].to_list()
    v5 = {
        "s_name_freq": [_NAME_FREQ.get(c, 0) for c in scat],
        "s_name_legal_freq": [_NAME_LEGAL_FREQ.get(c + "|" + l, 0) for c, l in zip(scat, sl)],
        "q_name_freq": [_NAME_FREQ.get(c, 0) if c else np.nan for c in qcat],
        "q_name_legal_freq": [_NAME_LEGAL_FREQ.get(c + "|" + l, 0) if c else np.nan
                              for c, l in zip(qcat, ql)],
        "legal_eq": [float(a == b) if (a or b) else np.nan for a, b in zip(ql, sl)],
    }
    countries = p["q_country"].to_list()
    v5["tok_extra_lratio_max"] = [_extra_lratio(a, b, _LRATIO.get(c, {}))
                                  for a, b, c in zip(qfull, sfull, countries)]
    v5["tok_all_lratio_max"] = [max((_LRATIO.get(c, {}).get(t, -9.0) for t in a.split()),
                                    default=np.nan) for a, c in zip(qfull, countries)]
    out = out.with_columns(**{k: pl.Series(v, dtype=pl.Float32) for k, v in v5.items()})
    return out.select(FEATURES)


def _with_skeletons(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Phonetic skeletons computed once per record (not once per pair)."""
    return df.with_columns(
        pl.Series(prefix + "name_skel", _skel_str(df[prefix + "name_core"])),
        pl.Series(prefix + "addr_skel", _skel_str(df[prefix + "addr_set"])))


def attach_text(cand: pl.DataFrame, q: pl.DataFrame, s1: pl.DataFrame) -> pl.DataFrame:
    """cand: (id, s1_id, bscore, brank). q / s1: normalised frames with entity_id."""
    qq = q.filter(pl.col("entity_id").is_in(cand["id"].unique().implode())).select(
        pl.col("entity_id").alias("id"), pl.col("country").alias("q_country"),
        *[pl.col(c).alias("q_" + c) for c in TEXT_COLS])
    ss = s1.filter(pl.col("entity_id").is_in(cand["s1_id"].unique().implode())).select(
        pl.col("entity_id").alias("s1_id"), *[pl.col(c).alias("s_" + c) for c in TEXT_COLS])
    p = cand.join(_with_skeletons(qq, "q_"), on="id").join(_with_skeletons(ss, "s_"), on="s1_id")
    return p.with_columns(q_key=pl.col("id"),
                          q_src=pl.col("id").str.slice(1, 1).cast(pl.Int8))
