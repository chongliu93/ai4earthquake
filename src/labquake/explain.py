# =============================================================================
#   ai4earthquake :: src/labquake
# =============================================================================
#   Module       :: explain.py -- The three-step explanation of a trained forest
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""The three-step explanation of a trained forest.

feature2target   ``shap_importance``: mean |SHAP| of each of the twelve
                 features (TreeExplainer, tree-path-dependent, no background
                 data), normalised to sum to one (Methods, Feature importance).
feature2physics  ``legendre_jets`` + ``decompose``: the level, rate and
                 curvature of a statistic over the embedding span are its
                 projections on orthonormal Legendre polynomials of orders
                 0, 1, 2 over the trailing D samples; each history coordinate
                 is regressed on the first eight orders and its weight on
                 each order is b_k^2 / sum(b^2) (Methods, Physics-based
                 interpretation of the learned coordinates).
physics2target   ``correlations``: |r| of each physical quantity, and of each
                 feature, with the target (Methods, Correlation of the
                 physical quantities with the target).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ORDERS = ("c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7")
QUANTITY = {"inst": "instantaneous", "c0": "level", "c1": "rate", "c2": "curvature"}


# ------------------------------------------------------------------ SHAP ---
def shap_importance(forest, X, chunk=2000):
    """Normalised mean |SHAP| per feature over the rows of X (feature2target).

    Rows are processed in chunks; mean |SHAP| is a sum over rows, so the
    chunking does not change the result. Returns a Series indexed by the
    column names of X.
    """
    import shap                                                   # slow import

    explainer = shap.TreeExplainer(forest)
    Xv = np.asarray(X, dtype=np.float64)
    acc = np.zeros(Xv.shape[1])
    for i in range(0, len(Xv), chunk):
        sv = explainer.shap_values(Xv[i:i + chunk], check_additivity=False)
        acc += np.abs(sv).sum(0)
    share = acc / acc.sum()
    names = list(X.columns) if hasattr(X, "columns") else [f"f{j}" for j in range(len(share))]
    return pd.Series(share, index=names, name="shap_importance")


# ------------------------------------------------------- Legendre jets ---
def legendre_basis(D, n=8):
    """Orthonormal discrete Legendre polynomials of orders 0..n-1 on D points (rows)."""
    x = np.linspace(-1, 1, D)
    K = []
    for o in range(n):
        q = np.polynomial.legendre.Legendre.basis(o)(x).copy()
        for k in K:                                   # Gram-Schmidt: exact
            q -= (q @ k) * k                          # discrete orthonormality
        q /= np.linalg.norm(q)
        K.append(q)
    return np.array(K)


def legendre_jets(v, D, n=8):
    """Projection of the trailing D-sample window of v on each Legendre order.

    Returns an (len(v), n) array; row i describes the window ending at i, so
    it is aligned with the history coordinates; the first D-1 rows are NaN.
    Column 0 is the level, 1 the rate, 2 the curvature.
    """
    v = np.asarray(v, dtype=np.float64)
    K = legendre_basis(D, n)
    out = np.full((len(v), n), np.nan)
    for o in range(n):
        out[D - 1:, o] = np.convolve(v, K[o][::-1], mode="full")[D - 1:len(v)]
    return out


def rate_scale(D):
    """Factor turning the rate coefficient into the rise across the span (K1[-1] - K1[0])."""
    K = legendre_basis(D, 2)
    return float(K[1][-1] - K[1][0])


def decompose(table, D, coords=("PCA1", "PCA2", "PCA3", "Emb1", "Emb2"),
              n_jets=8, n_show=4):
    """feature2physics for one channel table (time, value, target, coordinates).

    Each coordinate is standardised and regressed on the standardised jets;
    returns a DataFrame with one row per coordinate: R2 of the fit, the
    weight on c0..c(n_show-1) and ``rest`` for the higher orders.
    """
    J = legendre_jets(table["value"].values, D, n_jets)
    m = np.isfinite(J).all(1)
    Jz = (J[m] - J[m].mean(0)) / J[m].std(0)
    rows = []
    for c in coords:
        q = table[c].values[m]
        if q.std() == 0:
            continue
        qz = (q - q.mean()) / q.std()
        b, *_ = np.linalg.lstsq(Jz, qz, rcond=None)
        R2 = float(1 - np.sum((Jz @ b - qz) ** 2) / np.sum(qz ** 2))
        w = b ** 2 / np.sum(b ** 2)
        rows.append({"coord": c, "R2": R2,
                     **{ORDERS[o]: float(w[o]) for o in range(n_show)},
                     "rest": float(w[n_show:].sum())})
    return pd.DataFrame(rows), J


# ------------------------------------------------------- correlations ---
def _r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def correlations(table, J, coords=("PCA1", "PCA2", "PCA3", "Emb1", "Emb2")):
    """physics2target: |r| with the target of the instantaneous value, the
    level, rate and curvature (returned as ``ladder``) and of every feature
    (``features``); ``carrier`` names the quantity with the largest |r|."""
    m = np.isfinite(J).all(1)
    y = table["target"].values[m]
    v = table["value"].values[m]
    ladder = {"inst": abs(_r(v, y)), "c0": abs(_r(J[m, 0], y)),
              "c1": abs(_r(J[m, 1], y)), "c2": abs(_r(J[m, 2], y))}
    feats = {"value": _r(v, y)}
    for c in coords:
        feats[c] = _r(table[c].values[m], y)
    return {"ladder": ladder, "carrier": max(ladder, key=ladder.get),
            "features": feats}


def importance_by_quantity(shap_share, decomposition, is_value):
    """Redistribute SHAP importance from features to physical quantities.

    shap_share    : Series over the channel's features (value + coordinates)
    decomposition : DataFrame from ``decompose`` (coord, c0.., rest, R2)
    is_value      : name of the instantaneous feature in shap_share
    importance(c_k) = sum over coordinates of SHAP(coord) * weight_k(coord) * R2(coord)
    The part 1 - R2 that the jets do not describe is reported as ``unexplained``.
    """
    out = {"inst": float(shap_share[is_value]), "c0": 0.0, "c1": 0.0, "c2": 0.0,
           "higher": 0.0, "unexplained": 0.0}
    for _, row in decomposition.iterrows():
        s = float(shap_share.get(row["coord"], 0.0))
        for k in ("c0", "c1", "c2"):
            out[k] += s * row[k] * row["R2"]
        out["higher"] += s * (row.get("c3", 0.0) + row["rest"]) * row["R2"]
        out["unexplained"] += s * (1 - row["R2"])
    return out
