"""Stage 4: train the pair classifiers (LightGBM) and tune the decision thresholds.

Two models (v5):
  * SEEN model   - all features except the label-free over-representation ones; used for
                   countries present in the training labels (US, India). Threshold tuned
                   on the in-country validation below.
  * UNSEEN model - "label-free": no features derived from training labels (decoy-word
                   match rates), so it cannot rely on knowing a country's decoy words;
                   used for countries absent from training (France). Its threshold comes
                   from leave-one-country-out experiments (loco.py): train on India only ->
                   evaluate on US and vice versa; best average at 0.90 (label-free 0.949
                   vs 0.937 for the seen-model approach at its best threshold).

Validation design (avoids optimistic precision):
  * V = 1% of S1 entities (hash-selected).
  * Validation queries = EVERY S2/S3 query that has some V entity among its candidates,
    so every possible false merge onto a V entity is scored.
  * Training queries = a sample of the remaining queries whose true S1 is not in V.
Decision rule: each query is assigned to its highest-probability candidate if that
probability >= threshold (S2/S3 records belong to at most one S1 entity).
"""
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from config import ART, N_TRAIN_Q, SEED, WORK
from evaluate import macro_f05
import token_rates
from features import FEATURES, set_token_rates
from featurize import featurize, load_norm, load_vocab
from io_utils import read_ground_truth
from run_blocking import cand_path

VAL_PCT = 1
MODEL_PATH = ART / "lgbm_pair_model.txt"            # seen-country model
MODEL_UNSEEN_PATH = ART / "lgbm_pair_model_unseen.txt"
UNSEEN_THRESHOLD = 0.90                             # from loco.py (see docstring)
LRATIO = ["tok_extra_lratio_max", "tok_all_lratio_max"]
LABELED = ["tok_extra_min", "tok_extra_nlow", "tok_all_min", "atok_extra_min"]
SEEN_FEATURES = [f for f in FEATURES if f not in LRATIO]
UNSEEN_FEATURES = [f for f in FEATURES if f not in LABELED]
META_PATH = ART / "model_meta.json"


def hsel(col: str, mod: int = 100):
    return pl.col(col).hash(seed=SEED) % mod


def build_matrix(cand: pl.DataFrame, gt: pl.DataFrame, tag: str, boosters=None):
    """Featurise candidate pairs. Returns (keys, X, y) with X a list of chunks, or
    (keys, {name: prob}, y) when boosters {name: booster} are given (scored chunk by chunk
    so the full matrix is never held in memory)."""
    t = time.time()
    q = load_norm("train", cand["id"].unique(), sources=(2, 3))
    s1 = load_norm("train", cand["s1_id"].unique(), sources=(1,))
    keys, xs = [], []
    probs = {n: [] for n in (boosters or {})}
    for k, X in featurize(cand, q, s1):
        keys.append(k)
        if boosters is None:
            xs.append(X.to_numpy())
        else:
            for n, b in boosters.items():
                probs[n].append(b.predict(X.select(b.feature_name()).to_numpy(),
                                          num_iteration=b.best_iteration))
    keys = pl.concat(keys)
    out = xs if boosters is None else {n: np.concatenate(v) for n, v in probs.items()}
    y = keys.join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["id", "s1_id"],
                  how="left", maintain_order="left")["y"].fill_null(0).to_numpy()
    print(f"[train] {tag}: {len(y):,} pairs, {int(y.sum()):,} positives "
          f"({time.time() - t:.0f}s)", flush=True)
    return keys, out, y


def assign(keys: pl.DataFrame, prob: np.ndarray, thr: float) -> pl.DataFrame:
    """Best candidate per query, kept if prob >= thr -> (s1_id, id) matches."""
    d = keys.with_columns(prob=pl.Series(prob))
    best = d.sort("prob", descending=True).group_by("id").first()
    return best.filter(pl.col("prob") >= thr).select("s1_id", "id", "prob")


def main():
    t0 = time.time()
    load_vocab("train")
    gt = read_ground_truth()
    cand = pl.scan_parquet(cand_path("train"))
    s1_all = pl.scan_parquet(WORK / "norm_train_s1.parquet").select("entity_id").collect()
    val_s1 = s1_all.filter(hsel("entity_id") < VAL_PCT)["entity_id"]
    val_q = (cand.filter(pl.col("s1_id").is_in(val_s1.implode())).select("id").unique()
             .collect()["id"])
    # decoy-word rates learned on training records only (validation queries excluded)
    set_token_rates(token_rates.build(exclude_ids=val_q))

    # training queries: not touching V, and whose true entity is not in V
    bad = gt.filter(pl.col("s1_id").is_in(val_s1.implode()))["id"]
    tr_pool = (cand.select("id").unique()
               .filter(~pl.col("id").is_in(val_q.implode()) & ~pl.col("id").is_in(bad.implode()))
               .collect())
    tr_q = tr_pool.sample(min(N_TRAIN_Q, tr_pool.height), seed=SEED)["id"]
    tr_cand = cand.filter(pl.col("id").is_in(tr_q.implode())).collect()
    print(f"[train] V entities {val_s1.len():,}, val queries {val_q.len():,}, "
          f"train queries {tr_q.len():,} ({time.time() - t0:.0f}s)", flush=True)

    # early-stopping split inside the training queries (by query)
    es_mask = hsel("id", 10) == 0
    _, X_tr, y_tr = build_matrix(tr_cand.filter(~es_mask), gt, "train")
    _, X_es, y_es = build_matrix(tr_cand.filter(es_mask), gt, "early-stop")
    del tr_cand
    params = dict(objective="binary", learning_rate=0.05, num_leaves=127,
                  min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, max_bin=255, verbose=-1,
                  num_threads=0, seed=SEED)
    col = {f: i for i, f in enumerate(FEATURES)}
    models = (("seen", SEEN_FEATURES, MODEL_PATH), ("unseen", UNSEEN_FEATURES, MODEL_UNSEEN_PATH))
    # memory: convert to LightGBM's binned format (~1 byte/value) one model at a time,
    # then drop the float32 matrices before any training starts
    data = {}
    for name, feats, _ in models:
        idx = [col[f] for f in feats]
        dtr = lgb.Dataset(np.vstack([x[:, idx] for x in X_tr]), y_tr, feature_name=feats,
                          params={"max_bin": 255, "verbose": -1}).construct()
        dva = lgb.Dataset(np.vstack([x[:, idx] for x in X_es]), y_es, reference=dtr).construct()
        data[name] = (dtr, dva)
        gc.collect()
    del X_tr, X_es
    gc.collect()
    boosters = {}
    for name, feats, path in models:
        dtr, dva = data.pop(name)
        b = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
        b.save_model(str(path))
        boosters[name] = b
        del dtr, dva
        gc.collect()
        print(f"[train] {name} model: {b.best_iteration} trees, {len(feats)} features "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- validation: full decision pipeline + macro F0.5 on V entities
    val_cand = cand.filter(pl.col("id").is_in(val_q.implode())).collect()
    k_va, probs, y_va = build_matrix(val_cand, gt, "validation", boosters=boosters)
    k_va.with_columns(prob=pl.Series(probs["seen"]), prob_unseen=pl.Series(probs["unseen"]),
                      y=pl.Series(y_va)).write_parquet(WORK / "val_scored.parquet")
    truth_v = gt.filter(pl.col("s1_id").is_in(val_s1.implode()))
    results = {}
    for name in ("seen", "unseen"):
        rows = []
        for thr in np.round(np.arange(0.30, 0.96, 0.05), 2):
            pred = assign(k_va, probs[name], thr).filter(pl.col("s1_id").is_in(val_s1.implode()))
            r = macro_f05(pred, truth_v, val_s1.to_list())
            r["thr"] = float(thr)
            rows.append(r)
            print(f"[val:{name}] thr={thr:.2f} F0.5={r['macro_f05']:.4f} "
                  f"P={r['micro_precision']:.4f} R={r['micro_recall']:.4f} "
                  f"singletonF={r['singleton_f']:.4f}", flush=True)
        results[name] = max(rows, key=lambda r: r["macro_f05"])
    ceil = truth_v.join(val_cand, on=["id", "s1_id"]).height / max(truth_v.height, 1)
    b = boosters["seen"]
    imp = dict(zip(b.feature_name(), b.feature_importance("gain").round(0).tolist()))
    train_countries = (pl.scan_parquet(WORK / "norm_train_s1.parquet").select("country")
                       .unique().collect()["country"].sort().to_list())
    meta = {"threshold": results["seen"]["thr"], "best_iteration": b.best_iteration,
            "features": b.feature_name(), "val": results["seen"],
            "train_countries": train_countries,
            "unseen": {"threshold": UNSEEN_THRESHOLD,
                       "best_iteration": boosters["unseen"].best_iteration,
                       "features": boosters["unseen"].feature_name(),
                       "val_in_country_best": results["unseen"]},
            "blocking_recall_val": ceil,
            "feature_gain": dict(sorted(imp.items(), key=lambda kv: -kv[1]))}
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"[train] seen model: threshold {results['seen']['thr']} -> val macro F0.5 "
          f"{results['seen']['macro_f05']:.4f}; unseen model uses threshold "
          f"{UNSEEN_THRESHOLD} (in-country val best {results['unseen']['macro_f05']:.4f}); "
          f"blocking recall ceiling {ceil:.4f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
