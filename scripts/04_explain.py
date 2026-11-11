#!/usr/bin/env python3
# =============================================================================
#   ai4earthquake :: scripts
# =============================================================================
#   Module       :: 04_explain.py -- Stage 4: the three-step explanation of the trained forest
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Stage 4: the three-step explanation of the trained forest.

Reads outputs/<system>/03_model/ (forest, feature table) and
outputs/<system>/02_embedding/ (channel tables) and writes to
outputs/<system>/04_explain/:
    shap_importance.csv        feature2target: SHAP importance of the twelve features
    decomposition.csv          feature2physics: Legendre weights (c0 level, c1 rate,
                               c2 curvature, c3, rest) and R^2 of every history coordinate
    correlations.csv           physics2target: |r| with the target of the instantaneous
                               value, level, rate and curvature, and of every feature
    importance_by_quantity.csv SHAP importance redistributed onto the physical quantities
    jets_<channel>.h5          level, rate, curvature of each statistic at every row

SHAP rows: the whole record in the laboratory, the training rows (every
second one) at Kīlauea, as in the paper (explain.shap_rows / shap_decimation).
"""
import joblib
import numpy as np
import pandas as pd

from common import parse
from labquake import explain, io

args, cfg = parse(__doc__)
ex = cfg["explain"]
D = int(cfg["embedding"]["D"])
mdir, edir = io.stage_dir(cfg, "03_model"), io.stage_dir(cfg, "02_embedding")
out = io.stage_dir(cfg, "04_explain")

# ---- feature2target ------------------------------------------------------
forest = joblib.load(mdir / "forest.joblib")
train, test = io.load_channel(mdir / "features.h5")
feat_cols = [c for c in train.columns if c not in ("time", "target")]
rows = train if ex.get("shap_rows", "all") == "train" else pd.concat([train, test])
X = rows[feat_cols].iloc[::int(ex.get("shap_decimation", 1))]
print(f"{cfg['system']}: SHAP over {len(X):,} rows ({ex.get('shap_rows', 'all')})")
shap_share = explain.shap_importance(forest, X)
shap_share.to_csv(out / "shap_importance.csv", header=True)
print("  top features:", shap_share.sort_values(ascending=False).head(3).round(3).to_dict())

# ---- feature2physics and physics2target, per channel -------------------
dec_rows, cor_rows, byq_rows = [], [], []
for chan in ("var", "kurt"):
    tr, te = io.load_channel(edir / f"{chan}.h5")
    table = pd.concat([tr, te]).reset_index(drop=True)
    dec, J = explain.decompose(table, D, n_jets=int(ex.get("n_jets", 8)))
    dec.insert(0, "channel", chan)
    dec_rows.append(dec)
    cor = explain.correlations(table, J)
    cor_rows.append({"channel": chan, **{f"r_{k}": v for k, v in cor["ladder"].items()},
                     "carrier": cor["carrier"],
                     **{f"r_feature_{k}": v for k, v in cor["features"].items()}})
    io.save_table(pd.DataFrame({"time": table["time"], "level_c0": J[:, 0], "rate_c1": J[:, 1],
                                "curvature_c2": J[:, 2]}), out / f"jets_{chan}.h5", "jets")
    # the channel's share of the SHAP importance, in the model's feature names
    share = pd.Series({("value" if k == f"{chan}_{chan}" else k.split("_", 1)[1]): v
                       for k, v in shap_share.items() if k.startswith(chan + "_")})
    byq_rows.append({"channel": chan, **explain.importance_by_quantity(share, dec, "value")})
    print(f"  {chan}: carrier {cor['carrier']} (|r| {cor['ladder'][cor['carrier']]:.2f}), "
          f"rate-to-rise factor {explain.rate_scale(D):.4f}")

pd.concat(dec_rows).to_csv(out / "decomposition.csv", index=False)
pd.DataFrame(cor_rows).to_csv(out / "correlations.csv", index=False)
pd.DataFrame(byq_rows).to_csv(out / "importance_by_quantity.csv", index=False)
print(f"-> {out}")
