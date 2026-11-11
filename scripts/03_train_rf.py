#!/usr/bin/env python3
# =============================================================================
#   ai4earthquake :: scripts
# =============================================================================
#   Module       :: 03_train_rf.py -- Stage 3 (Module 3): tune, train and evaluate the random forest
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Stage 3 (Module 3): tune, train and evaluate the random forest.

Reads outputs/<system>/02_embedding/{var,kurt}.h5 and writes to
outputs/<system>/03_model/:
    features.h5      keys "train"/"test": time, target and the twelve features
    forecast.csv     time, target, forecast on the test segment
    forest.joblib    the fitted scikit-learn forest (for stage 4)
    summary.json     hyperparameters, cross-validation score, train and test R^2
"""
import json

import joblib
import pandas as pd

from common import parse
from labquake import io, model

args, cfg = parse(__doc__)
emb = io.stage_dir(cfg, "02_embedding")
channels = {s: io.load_channel(emb / f"{s}.h5") for s in ("var", "kurt")}
m = cfg["model"]
Xtr, Xte, ytr, yte, ttr, tte = model.feature_matrix(channels, m.get("feature_order", "lab"))
print(f"{cfg['system']}: {Xtr.shape[1]} features, train {len(Xtr):,} rows, test {len(Xte):,} rows")
print("  features:", list(Xtr.columns))

params, cv_score = model.tune(Xtr, ytr, m["n_estimators"], n_iter=m.get("n_iter", 25),
                              cv=m.get("cv", 3), random_state=m.get("random_state", 0),
                              search_random_state=m.get("search_random_state"))
print(f"  BayesSearchCV: {params}, cv R^2 {cv_score:.4f}")

forest, pred = model.fit(Xtr, ytr, Xte, params, m["n_estimators"],
                         random_state=m.get("random_state", 0), backend=m.get("backend", "darts"))
r2_train = model.score(ytr, forest.predict(Xtr.values))
r2_test = model.score(yte, pred)
print(f"  train R^2 {r2_train:.4f}   test R^2 {r2_test:.4f}")

out = io.stage_dir(cfg, "03_model")
io.save_table(pd.concat([pd.DataFrame({"time": ttr, "target": ytr}), Xtr], axis=1),
              out / "features.h5", "train", mode="w")
io.save_table(pd.concat([pd.DataFrame({"time": tte, "target": yte}), Xte], axis=1),
              out / "features.h5", "test", mode="a")
pd.DataFrame({"time": tte, "target": yte, "forecast": pred}).to_csv(out / "forecast.csv", index=False)
joblib.dump(forest, out / "forest.joblib")
json.dump({"system": cfg["system"], "params": params, "cv_r2": cv_score,
           "train_r2": r2_train, "test_r2": r2_test, "n_train": int(len(Xtr)),
           "n_test": int(len(Xte)), "features": list(Xtr.columns)},
          open(out / "summary.json", "w"), indent=2)
print(f"-> {out}")
