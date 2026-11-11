# =============================================================================
#   ai4earthquake :: src/labquake
# =============================================================================
#   Module       :: model.py -- Module 3: the random-forest regression from the twelve features to the target
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Module 3: the random-forest regression from the twelve features to the target.

Two phases, as in the runs behind the paper (Methods, Random-forest
integrator; Model training and evaluation):

1. ``tune``: Bayesian search (scikit-optimize ``BayesSearchCV``, 25
   iterations, 3-fold cross-validation inside the training segment) over
   max_features, min_samples_split and min_samples_leaf of a scikit-learn
   ``RandomForestRegressor``.
2. ``fit``: a forest with the selected hyperparameters, trained on the whole
   training segment and evaluated once on the test segment. The laboratory
   runs did this through Darts' ``RandomForestModel`` with the twelve
   features as future covariates at lag 0 and no lag of the target; the
   Kīlauea run called scikit-learn's ``RandomForestRegressor`` directly.
   With lag-0 covariates only, the Darts model is exactly that estimator on
   the same design matrix (Darts wraps scikit-learn), so ``backend`` only
   selects the call path; ``"darts"`` is the default and ``"sklearn"`` gives
   the same forest without the Darts dependency.

Feature names are ``<channel>_<coordinate>`` with channel ``var`` or ``kurt``
and coordinate the statistic itself (``var_var``, ``kurt_kurt``), ``PCA1``
to ``PCA3`` or ``Emb1``/``Emb2`` (UMAP). The column order matters only
through the random draw of candidate features at each split; ``feature_order``
in the configuration reproduces the order each published run used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

ORDER = {
    # value first, then UMAP, then PCA: the laboratory files' column order
    "lab": ("value", "Emb1", "Emb2", "PCA1", "PCA2", "PCA3"),
    # value, PCA, UMAP: the Kīlauea files' column order
    "field": ("value", "PCA1", "PCA2", "PCA3", "Emb1", "Emb2"),
}


def feature_matrix(channels, order="lab"):
    """Assemble X_train, X_test, y_train, y_test, t_train, t_test from the channel tables.

    channels : {"var": (train, test), "kurt": (train, test)} as written by stage 2
    The target and time axis are taken from the first channel; the channels
    are row-aligned by construction.
    """
    cols = ORDER[order]
    Xtr, Xte = [], []
    for chan, (tr, te) in channels.items():
        names = [f"{chan}_{chan}" if c == "value" else f"{chan}_{c}" for c in cols]
        Xtr.append(pd.DataFrame(tr[list(cols)].values, columns=names))
        Xte.append(pd.DataFrame(te[list(cols)].values, columns=names))
    first_tr, first_te = next(iter(channels.values()))
    return (pd.concat(Xtr, axis=1), pd.concat(Xte, axis=1),
            first_tr["target"].values, first_te["target"].values,
            first_tr["time"].values, first_te["time"].values)


def tune(X, y, n_estimators, n_iter=25, cv=3, max_features_cap=50,
         random_state=0, search_random_state=None, n_jobs=-1):
    """Phase 1: BayesSearchCV over the three forest hyperparameters. Returns (best_params, best_cv_score)."""
    from skopt import BayesSearchCV                              # slow import

    rf = RandomForestRegressor(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state)
    spaces = {"max_features": (1, min(X.shape[1], max_features_cap)),
              "min_samples_split": (2, 201),
              "min_samples_leaf": (1, 201)}
    search = BayesSearchCV(estimator=rf, search_spaces=spaces, n_iter=n_iter, cv=cv,
                           random_state=search_random_state)
    search.fit(np.asarray(X), np.asarray(y))
    best = {k: int(v) for k, v in search.best_params_.items()}
    return best, float(search.best_score_)


def fit(X_train, y_train, X_test, params, n_estimators, random_state=0,
        backend="darts", n_jobs=-1):
    """Phase 2: train the forest and forecast the test segment.

    Returns (forest, y_pred_test) where ``forest`` is the fitted
    scikit-learn RandomForestRegressor (also when trained through Darts).
    """
    kw = dict(n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state,
              max_features=params["max_features"],
              min_samples_split=params["min_samples_split"],
              min_samples_leaf=params["min_samples_leaf"])
    Xtr, Xte = np.asarray(X_train, dtype=np.float64), np.asarray(X_test, dtype=np.float64)
    ytr = np.asarray(y_train, dtype=np.float64)
    if backend == "sklearn":
        rf = RandomForestRegressor(**kw).fit(Xtr, ytr)
        return rf, rf.predict(Xte)
    if backend != "darts":
        raise ValueError(f"unknown backend {backend!r}")

    from darts import TimeSeries                                  # slow import
    from darts.models import RandomForestModel

    cov = TimeSeries.from_values(np.vstack([Xtr, Xte]))
    target = TimeSeries.from_values(ytr)
    model = RandomForestModel(lags=None, lags_future_covariates=[0],
                              output_chunk_length=1, **kw)
    model.fit(series=target, future_covariates=cov)
    # Darts' own predict() walks the test segment one step at a time (each
    # step a call to the forest), which took 37 min for the 32,349 Kīlauea
    # rows; the wrapped estimator gives the same values in one call
    # (checked on the published p4679 and Kīlauea runs: identical R^2).
    return model.model, model.model.predict(Xte).astype(np.float64)


def score(y_true, y_pred):
    """Coefficient of determination R^2."""
    return float(r2_score(y_true, y_pred))
