# =============================================================================
#   ai4earthquake :: tests
# =============================================================================
#   Module       :: test_pipeline.py -- End-to-end test of stages 1-5 on a small synthetic stick-slip record
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""End-to-end test of stages 1-5 on a small synthetic stick-slip record.

The record is a saw-tooth "shear stress" with acoustic bursts near each
drop, so that the variance carries information about the stress and the
forest reaches R^2 > 0.5 on the test segment.

    pytest tests/                        (from the repository root)
"""
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from labquake import embedding, explain, features, io, model  # noqa: E402


def synthetic_record(n_cycles=40, cycle=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = n_cycles * cycle
    t = np.arange(n) * 1e-3
    phase = (np.arange(n) % cycle) / cycle
    stress = 5.0 + 0.4 * phase - 0.2 * phase ** 2 + 0.01 * rng.standard_normal(n)
    amp = 1.0 + 30.0 * np.exp(-((phase - 0.97) / 0.03) ** 2)    # bursts before the drop
    signal = amp * rng.standard_normal(n)
    return pd.DataFrame({"time": t, "signal": signal, "target": stress})


def test_features_alignment():
    rec = synthetic_record(n_cycles=3)
    W = 50
    st = features.rolling_statistics(rec["signal"].values, W)
    assert st.shape == (len(rec), 2)
    assert st["var"].isna().sum() == W - 1
    ch = features.rolling_statistics_chunked(rec["signal"].values, W, chunk=1000)
    # pandas' rolling estimators accumulate on-line, so a chunk that starts
    # afresh differs at the level of floating-point round-off
    np.testing.assert_allclose(ch.values, st.values, rtol=1e-5, atol=1e-8, equal_nan=True)


def test_hankel_and_jets():
    x = np.arange(10.0)
    H = embedding.hankel(x, 4)
    assert H.shape == (7, 4) and list(H[0]) == [0, 1, 2, 3] and list(H[-1]) == [6, 7, 8, 9]
    J = explain.legendre_jets(x, 4)
    assert np.isnan(J[:3]).all() and np.isfinite(J[3:]).all()
    # a linear series has no curvature: order 2 vanishes
    assert np.allclose(J[3:, 2], 0, atol=1e-9)


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    root = tmp_path_factory.mktemp("run")
    c = {"system": "synthetic", "label": "synthetic", "target_name": "shear_stress_MPa",
         "sampling_rate_hz": 1000, "recurrence_s": 2.0,
         "paths": {"raw": str(root), "outputs": str(root / "outputs")},
         "window": 50, "skip_rows": "nan",
         "embedding": {"S": 10, "D": 40, "n_pca": 3, "n_umap": 2, "n_neighbors": 30,
                       "min_dist": 0.01, "umap_random_state": 0, "pca_random_state": 0},
         "split": {"kind": "fraction", "test_fraction": 0.2},
         "model": {"n_estimators": 50, "n_iter": 4, "cv": 3, "random_state": 0,
                   "search_random_state": 0, "feature_order": "lab", "backend": "darts"},
         "explain": {"shap_rows": "all", "shap_decimation": 4, "n_jets": 8}}
    path = root / "synthetic.yaml"
    path.write_text(yaml.safe_dump(c))
    cfg = io.load_config(path)
    io.save_table(synthetic_record(), io.stage_dir(cfg, "00_raw") / "record.h5", "data")
    return cfg


def run(script, cfg, *extra):
    r = subprocess.run([sys.executable, str(REPO / "scripts" / script),
                        "--config", cfg["_config_path"], *extra],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_stages(cfg):
    run("01_compute_features.py", cfg)
    run("02_embed.py", cfg, "--target")
    tr, te = io.load_channel(io.stage_dir(cfg, "02_embedding") / "var.h5")
    assert list(tr.columns) == ["time", "value", "target", "PCA1", "PCA2", "PCA3", "Emb1", "Emb2"]
    ttr, tte = io.load_channel(io.stage_dir(cfg, "02_embedding") / "target.h5")
    assert len(ttr) == len(tr) and np.allclose(ttr["time"], tr["time"])   # row-aligned

    run("03_train_rf.py", cfg)
    s = json.load(open(io.stage_dir(cfg, "03_model") / "summary.json"))
    assert len(s["features"]) == 12 and s["test_r2"] > 0.5, s
    # the Darts path and the plain scikit-learn path give the same forest
    mdir = io.stage_dir(cfg, "03_model")
    train, test = io.load_channel(mdir / "features.h5")
    X = [c for c in train.columns if c not in ("time", "target")]
    _, p_sk = model.fit(train[X], train["target"], test[X], s["params"], 50, backend="sklearn")
    _, p_da = model.fit(train[X], train["target"], test[X], s["params"], 50, backend="darts")
    np.testing.assert_allclose(p_sk, p_da)

    run("04_explain.py", cfg)
    sh = pd.read_csv(io.stage_dir(cfg, "04_explain") / "shap_importance.csv", index_col=0)
    assert abs(sh.iloc[:, 0].sum() - 1) < 1e-6
    dec = pd.read_csv(io.stage_dir(cfg, "04_explain") / "decomposition.csv")
    assert set(dec.channel) == {"var", "kurt"} and ((dec.R2 >= 0) & (dec.R2 <= 1 + 1e-9)).all()

    run("05_ablation.py", cfg)
    ab = pd.read_csv(io.stage_dir(cfg, "05_ablation") / "ablation.csv")
    assert ab.n_feat_rawlags.iloc[0] == 2 * cfg["embedding"]["D"]
