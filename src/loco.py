"""Leave-one-country-out validation: a local proxy for the unseen test country (France).

Train on country A only (decoy-word rates learned from A only), evaluate on held-out
S1 entities of country B with the full decision rule and macro F0.5. Several feature
subsets are trained on the same matrices so variants can be compared cheaply.
usage: python loco.py <train_country> <eval_country>
"""
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

import token_rates
from config import SEED, WORK
from evaluate import macro_f05
from features import FEATURES, set_token_rates
from featurize import load_vocab
from io_utils import read_ground_truth
from run_blocking import cand_path
from train import assign, build_matrix, hsel

VAL_PCT_X10 = 8          # 0.8% of the eval country's S1 entities
N_TRAIN_Q = 400_000
LRATIO = ["tok_extra_lratio_max", "tok_all_lratio_max"]
V5 = ["s_name_freq", "s_name_legal_freq", "q_name_freq", "q_name_legal_freq", "legal_eq"]
VARIANTS = {
    "all": FEATURES,
    "no_lratio": [f for f in FEATURES if f not in LRATIO],
    "v4_features": [f for f in FEATURES if f not in LRATIO + V5],
}
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=0, seed=SEED)


def main(train_c: str, eval_c: str):
    t0 = time.time()
    load_vocab("train")
    gt = read_ground_truth()
    cand = pl.scan_parquet(cand_path("train"))
    s1 = pl.scan_parquet(WORK / "norm_train_s1.parquet").select("entity_id", "country").collect()
    V = s1.filter((pl.col("country") == eval_c)
                  & (hsel("entity_id", 1000) < VAL_PCT_X10))["entity_id"]
    a_ids = s1.filter(pl.col("country") == train_c)["entity_id"]
    val_q = cand.filter(pl.col("s1_id").is_in(V.implode())).select("id").unique().collect()["id"]
    val_cand = cand.filter(pl.col("id").is_in(val_q.implode())).collect()
    tr_pool = (cand.filter(pl.col("s1_id").is_in(a_ids.implode())).select("id").unique()
               .collect())
    tr_q = tr_pool.sample(min(N_TRAIN_Q, tr_pool.height), seed=SEED)["id"]
    tr_cand = cand.filter(pl.col("id").is_in(tr_q.implode())).collect()
    set_token_rates(token_rates.build(country=train_c, save=False))
    print(f"[loco] train {train_c} ({tr_q.len():,} q) -> eval {eval_c} ({V.len():,} S1, "
          f"{val_q.len():,} q) ({time.time() - t0:.0f}s)", flush=True)
    es = hsel("id", 10) == 0
    _, Xtr, ytr = build_matrix(tr_cand.filter(~es), gt, "train")
    _, Xes, yes = build_matrix(tr_cand.filter(es), gt, "early-stop")
    kva, Xva, _ = build_matrix(val_cand, gt, "eval")
    Xtr, Xes, Xva = np.vstack(Xtr), np.vstack(Xes), np.vstack(Xva)
    truth = gt.filter(pl.col("s1_id").is_in(V.implode()))
    col = {f: i for i, f in enumerate(FEATURES)}
    for name, feats in VARIANTS.items():
        idx = [col[f] for f in feats]
        dtr = lgb.Dataset(Xtr[:, idx], ytr, feature_name=feats)
        dva = lgb.Dataset(Xes[:, idx], yes, reference=dtr)
        b = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        prob = b.predict(Xva[:, idx], num_iteration=b.best_iteration)
        res = []
        for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
            pred = assign(kva, prob, thr).filter(pl.col("s1_id").is_in(V.implode()))
            r = macro_f05(pred, truth, V.to_list())
            res.append(f"{thr}:{r['macro_f05']:.4f}(P{r['micro_precision']:.3f})")
        print(f"[loco] {train_c}->{eval_c} {name:12s} trees={b.best_iteration:4d} " + " ".join(res),
              flush=True)
    print(f"[loco] done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
