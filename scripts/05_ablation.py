#!/usr/bin/env python3
# =============================================================================
#   ai4earthquake :: scripts
# =============================================================================
#   Module       :: 05_ablation.py -- Stage 5: the input ablation of Fig. 2f
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Stage 5: the input ablation of Fig. 2f.

Three forests with the hyperparameters of stage 3, the same split and the
same target:
    instant   the two instantaneous statistics only
    rawlags   the two statistics plus their D raw lags each (the uncompressed
              content of the Hankel rows, 2D features)
    pipeline  the twelve features of stage 3 (the published model)
plus, for the raw-lag forest, a sweep of max_features (sqrt and the
ratio-matched 2D/6) so that it is not handicapped by the twelve-feature
model's setting; Fig. 2f shows the best of these.

Training rows without a full lag history are dropped from the training
segment only. Output: outputs/<system>/05_ablation/{ablation.csv, maxfeatures.csv}.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from common import parse
from labquake import io, model

args, cfg = parse(__doc__)
D = int(cfg["embedding"]["D"])
mdir = io.stage_dir(cfg, "03_model")
summ = json.load(open(mdir / "summary.json"))
P = dict(summ["params"], n_estimators=cfg["model"]["n_estimators"],
         random_state=cfg["model"].get("random_state", 0), n_jobs=-1)
train, test = io.load_channel(mdir / "features.h5")
ytr, yte = train["target"].values, test["target"].values
ntr, nte = len(train), len(test)
v = np.concatenate([train["var_var"].values, test["var_var"].values])
k = np.concatenate([train["kurt_kurt"].values, test["kurt_kurt"].values])


def lagmat(n0, n1):
    idx = np.arange(n0, n1)[:, None] - np.arange(D)[None, :]
    return np.hstack([v[idx], k[idx]]).astype(np.float32)


def run(name, Xtr, ytr_, Xte, params):
    t0 = time.time()
    rf = RandomForestRegressor(**params).fit(Xtr, ytr_)
    r2 = model.score(yte, rf.predict(Xte))
    print(f"  {name:10s} R^2 {r2:.4f}  ({Xtr.shape[1]} features, {time.time() - t0:.0f} s)")
    return r2


feat = [c for c in train.columns if c not in ("time", "target")]
res = {"system": cfg["system"], "D": D}
res["r2_instant"] = run("instant", train[["var_var", "kurt_kurt"]].values, ytr,
                        test[["var_var", "kurt_kurt"]].values, P)
XL_tr, XL_te = lagmat(D - 1, ntr), lagmat(ntr, ntr + nte)
res["r2_rawlags"] = run("rawlags", XL_tr, ytr[D - 1:], XL_te, P)
res["r2_pipeline"] = run("pipeline", train[feat].values, ytr, test[feat].values, P)
res["n_feat_rawlags"] = 2 * D
out = io.stage_dir(cfg, "05_ablation")
pd.DataFrame([res]).to_csv(out / "ablation.csv", index=False)

rows = []
for mf in ("sqrt", max(2, int(2 * D / 6))):
    r2 = run(f"rawlags mf={mf}", XL_tr, ytr[D - 1:], XL_te, dict(P, max_features=mf))
    rows.append({"system": cfg["system"], "max_features": str(mf), "r2_rawlags": r2})
pd.DataFrame(rows).to_csv(out / "maxfeatures.csv", index=False)
print(f"-> {out}")
