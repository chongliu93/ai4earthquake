# =============================================================================
#   ai4earthquake :: src/labquake
# =============================================================================
#   Module       :: embedding.py -- Module 2: time-delay embedding of one statistic, reduced by PCA and UMAP
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Module 2: time-delay embedding of one statistic, reduced by PCA and UMAP.

For one channel (the variance or the kurtosis) the strided series is turned
into a Hankel matrix whose row n holds the D values ending at n (the trailing
span L = W + (D-1)S), and the rows are reduced to three principal components
and two UMAP coordinates, both fitted on the training segment only and
applied to the test segment (Methods, Time-delay embedding). The same
function embeds the target itself (shear stress or displacement) for the
target-side reconstructions of Fig. 3.

Order of operations, kept identical to the runs behind the paper:
    1. drop the rows without a statistic (``skip_rows``: ``"nan"`` drops the
       W-1 leading NaN rows; an integer drops that many leading rows)
    2. split into train and test, by row fraction or by date
    3. keep every S-th row of each segment
    4. Hankel matrix with D taps; the first D-1 rows of each segment are lost
    5. UMAP(2) and PCA(3), fitted on the training rows
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from sklearn.decomposition import PCA

COORDS = ("PCA1", "PCA2", "PCA3", "Emb1", "Emb2")


def hankel(x, D):
    """Rows of D consecutive values: row i = x[i : i+D], oldest first (shape (n-D+1, D))."""
    x = np.asarray(x, dtype=np.float64)
    return toeplitz(x[::-1], x[0:D])[::-1, :][:-D + 1, :]


def align(y, D):
    """The value that goes with each Hankel row: the last sample of the row."""
    return np.asarray(y)[D - 1:]


def prepare_series(record, features, stat, skip_rows="nan"):
    """One channel as a DataFrame time, value, target with the leading rows removed."""
    df = pd.DataFrame({"time": record["time"].values, "value": features[stat].values,
                       "target": record["target"].values})
    if skip_rows == "nan":
        df = df.dropna()
    else:
        df = df.iloc[int(skip_rows):]
        df = df.dropna()
    return df.reset_index(drop=True)


def split(df, spec):
    """Train/test split of a time-ordered DataFrame.

    spec = {"kind": "fraction", "test_fraction": 0.2}
         : the last fraction of the rows is the test segment
    spec = {"kind": "date", "start": ..., "train_end": ..., "end": ...}
         : rows with start <= time < train_end train, train_end <= time < end test
           (times as POSIX seconds; the spec values as anything pandas parses)
    """
    if spec["kind"] == "fraction":
        cut = int(len(df) * (1 - spec["test_fraction"]))
        return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)
    if spec["kind"] == "date":
        t = df["time"].values
        t0 = pd.Timestamp(spec["start"]).timestamp()
        t1 = pd.Timestamp(spec["train_end"]).timestamp()
        t2 = pd.Timestamp(spec["end"]).timestamp()
        return (df[(t >= t0) & (t < t1)].reset_index(drop=True),
                df[(t >= t1) & (t < t2)].reset_index(drop=True))
    raise ValueError(f"unknown split kind {spec['kind']!r}")


def embed_segments(train, test, S, D, n_pca=3, n_umap=2, n_neighbors=200,
                   min_dist=0.01, umap_random_state=0, pca_random_state=0):
    """Stride, Hankel-embed and reduce one channel.

    train, test : DataFrames time, value, target (from ``split``)
    Returns (train_table, test_table, models); the tables hold
    time, value, target, PCA1..3, Emb1..2 and models = {"pca": ..., "umap": ...}.
    """
    import umap.umap_ as umap                                  # slow import

    def strided(df):
        return {c: df[c].values[::S] for c in ("time", "value", "target")}

    tr, te = strided(train), strided(test)
    X_tr, X_te = hankel(tr["value"], D), hankel(te["value"], D)

    reducer = umap.UMAP(n_components=n_umap, n_neighbors=n_neighbors, min_dist=min_dist,
                        random_state=umap_random_state, n_jobs=-1)
    E_tr = reducer.fit_transform(X_tr)
    E_te = reducer.transform(X_te)
    pca = PCA(n_components=n_pca, random_state=pca_random_state)
    P_tr = pca.fit_transform(X_tr)
    P_te = pca.transform(X_te)

    def table(seg, P, E):
        d = {c: align(seg[c], D) for c in ("time", "value", "target")}
        for i in range(n_pca):
            d[f"PCA{i + 1}"] = P[:, i]
        for i in range(n_umap):
            d[f"Emb{i + 1}"] = E[:, i]
        return pd.DataFrame(d)

    return table(tr, P_tr, E_tr), table(te, P_te, E_te), {"pca": pca, "umap": reducer}


def embed_channel(record, features, stat, cfg):
    """Stages 1-5 above for one statistic, driven by a system configuration."""
    emb = cfg["embedding"]
    df = prepare_series(record, features, stat, cfg.get("skip_rows", "nan"))
    train, test = split(df, cfg["split"])
    return embed_segments(train, test, S=emb["S"], D=emb["D"],
                          n_pca=emb.get("n_pca", 3), n_umap=emb.get("n_umap", 2),
                          n_neighbors=emb.get("n_neighbors", 200),
                          min_dist=emb.get("min_dist", 0.01),
                          umap_random_state=emb.get("umap_random_state", 0),
                          pca_random_state=emb.get("pca_random_state", 0))


def embed_target(record, features, cfg):
    """The target-side reconstruction: the same pipeline run on the target series.

    The rows are kept row-aligned with the seismic channels by masking the
    target with the NaN pattern of the statistics before the leading rows
    are dropped, so the two reconstructions describe simultaneous states.
    """
    ref = features.iloc[:, 0].values
    masked = np.where(np.isfinite(ref), record["target"].values, np.nan)
    return embed_channel(record, pd.DataFrame({"target_copy": masked}), "target_copy", cfg)
