# =============================================================================
#   ai4earthquake :: src/labquake
# =============================================================================
#   Module       :: kilauea.py -- The Kīlauea 2018 collapse sequence: assembling the raw record (stage 0)
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""The Kīlauea 2018 collapse sequence: assembling the raw record (stage 0).

The seismic and GNSS data were preprocessed with the code shared by Johnson
& Johnson (2024), https://github.com/cjohnson-LANL/GRL_Kilauea, which is
not copied here. Its outputs that this module reads are

* the RIMD waveform file (HDF5) written by their seismic preprocessing:
  the three components ``HHE``, ``HHN``, ``HHZ`` as 100-Hz ground velocity
  (instrument response removed, detrended, band-passed 0.1-50 Hz) with the
  attributes ``starttime``, ``endtime``, ``sampling_rate`` and ``npts``;
* the CRIM displacement table (parquet, one or more files) with a UTC time
  index and the column ``hor_mag_detrend``: the detrended horizontal
  displacement sqrt(dE^2 + dN^2) at 5-s sampling, averaged to 30 s and
  smoothed between collapses as in Johnson et al. (2025).

Our only departure from Johnson & Johnson (2024) is that a single seismic
component enters the framework (``component``, HHN in the paper). The
displacement is linearly interpolated onto the seismic sample times, so the
record has one displacement value per seismic sample, and the rolling
statistics of stage 1 are taken over W = 3000 samples (30 s).

Time is POSIX seconds. The paper's temporal domain is 1 June to 2 August
2018, split at 23 July 00:00 UTC (train before, test from then on); the
values are in ``configs/kilauea.yaml``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FS = 100.0                                   # Hz, the RIMD sampling rate


def _posix(s):
    s = s.decode() if isinstance(s, bytes) else str(s)
    return pd.Timestamp(s.replace("Z", "")).timestamp()


def read_displacement(parquet_files):
    """CRIM displacement from the Johnson & Johnson parquet output: (posix time, mm)."""
    df = pd.concat([pd.read_parquet(f) for f in parquet_files])
    df = df[~df.index.duplicated()].sort_index()
    t = np.array([ts.timestamp() for ts in df.index])
    return t, df["hor_mag_detrend"].values.astype(np.float64)


def build_record(seismic_h5, parquet_files, component="HHN", start="2018-06-01",
                 end="2018-08-05 23:59:59", chunk=8_640_000):
    """One seismic component and the interpolated displacement on the 100-Hz grid.

    Returns a DataFrame time, signal, target (float32 signal to bound memory:
    the two months hold about 5.7e8 samples).
    """
    import h5py                                                   # slow import

    with h5py.File(seismic_h5, "r") as f:
        t_start = _posix(f.attrs["starttime"])
        fs = float(f.attrs["sampling_rate"])
        i0 = int(round((pd.Timestamp(start).timestamp() - t_start) * fs))
        i1 = int(round((pd.Timestamp(end).timestamp() - t_start) * fs))
        n = i1 - i0
        signal = np.empty(n, dtype=np.float32)
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            signal[a:b] = f[component][i0 + a:i0 + b]
    # sample times, with the half-sample offset of the block convention of
    # the reference preprocessing
    time = t_start + np.arange(i0, i1) / fs - 0.5 / fs
    gt, gd = read_displacement(parquet_files)
    target = np.interp(time, gt, gd)
    return pd.DataFrame({"time": time, "signal": signal, "target": target})
