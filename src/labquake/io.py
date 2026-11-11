# =============================================================================
#   ai4earthquake :: src/labquake
# =============================================================================
#   Module       :: io.py -- Reading and writing the records, feature tables and configurations
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Reading and writing the records, feature tables and configurations.

Tables are stored in HDF5 through h5py: one group per table (``key``), one
dataset per column, and the column order in the group attribute ``columns``.

File conventions
----------------
Raw record (stage 0 output), one HDF5 file per system, key ``data``:
    ``time``   seconds (laboratory: experiment clock; Kīlauea: POSIX)
    ``signal`` the raw seismic sample (AE in counts, or ground velocity in m/s)
    ``target`` shear stress (MPa) or displacement (mm) at the same sample
Rolling statistics (stage 1 output): the same file plus key ``features``
with the columns ``var`` and ``kurt`` (NaN for the first W-1 rows),
row-aligned with ``data``.
Embedded channel (stage 2 output), one HDF5 file per statistic, keys
``train`` and ``test`` with the columns
    time, value, target, PCA1, PCA2, PCA3, Emb1, Emb2
where ``value`` is the strided statistic itself and Emb1/Emb2 are the UMAP
coordinates (the paper calls them UMAP1 and UMAP2).
"""
from __future__ import annotations

import pathlib

import h5py
import numpy as np
import pandas as pd
import yaml

REPO = pathlib.Path(__file__).resolve().parents[2]


def load_config(path):
    """Read a YAML configuration; relative paths are taken from the repository root."""
    path = pathlib.Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    paths = cfg.setdefault("paths", {})
    for key, value in list(paths.items()):
        p = pathlib.Path(value).expanduser()
        paths[key] = str(p if p.is_absolute() else REPO / p)
    paths["outputs"] = str(pathlib.Path(paths.get("outputs", REPO / "outputs")) / cfg["system"])
    return cfg


def stage_dir(cfg, stage):
    """``outputs/<system>/<stage>/``, created on demand."""
    d = pathlib.Path(cfg["paths"]["outputs"]) / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_table(df, path, key, mode="w"):
    """Write a DataFrame as group ``key`` of an HDF5 file (mode "w" new file, "a" add)."""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, mode) as f:
        if key in f:
            del f[key]
        g = f.create_group(key)
        for c in df.columns:
            g.create_dataset(str(c), data=np.asarray(df[c].values), compression="gzip",
                             compression_opts=1, chunks=True)
        g.attrs["columns"] = [str(c) for c in df.columns]


def load_table(path, key, columns=None):
    """Read group ``key`` back as a DataFrame (optionally a subset of columns)."""
    with h5py.File(path, "r") as f:
        g = f[key]
        cols = list(g.attrs["columns"]) if columns is None else list(columns)
        return pd.DataFrame({c: g[c][()] for c in cols})


def load_record(path):
    """The raw record: a DataFrame with time, signal, target."""
    return load_table(path, "data")


def load_features(path):
    """The stage-1 file: (record, features) DataFrames, row-aligned."""
    return load_table(path, "data"), load_table(path, "features")


def load_channel(path):
    """The stage-2 file of one statistic: (train, test) DataFrames."""
    return load_table(path, "train"), load_table(path, "test")
