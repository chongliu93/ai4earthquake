# =============================================================================
#   ai4earthquake :: src/labquake
# =============================================================================
#   Module       :: features.py -- Module 1: rolling variance and kurtosis of the raw signal
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Module 1: rolling variance and kurtosis of the raw signal.

Each statistic is computed over the trailing window of W samples ending at
sample n and stamped at n, so no future sample enters it (Methods, Rolling
statistics). The first W-1 rows are NaN.

Implementation note. The statistics are pandas' rolling estimators, exactly
as in the runs behind the paper and as written in Eq. (2) of the Methods:
``Series.rolling(W).var()`` is the unbiased sample variance (denominator
W-1) and ``Series.rolling(W).kurt()`` is the bias-corrected *excess*
kurtosis (Fisher's definition, zero for a Gaussian signal).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STATS = ("var", "kurt")


def rolling_statistics(signal, window, stats=STATS):
    """Trailing-window statistics of a 1-D signal.

    Parameters
    ----------
    signal : array-like of length N
    window : int, W in samples
    stats : subset of ("var", "kurt")

    Returns
    -------
    DataFrame with one column per statistic, N rows, NaN for rows < W-1.
    """
    s = pd.Series(np.asarray(signal, dtype=np.float64))
    return s.rolling(int(window)).agg({k: k for k in stats}).reset_index(drop=True)


def rolling_statistics_chunked(signal, window, stats=STATS, chunk=50_000_000):
    """The same, computed in overlapping chunks to bound memory on 10^9-sample records."""
    n = len(signal)
    if n <= chunk:
        return rolling_statistics(signal, window, stats)
    out = np.full((n, len(stats)), np.nan)
    start = 0
    while start < n:
        stop = min(n, start + chunk)
        lo = max(0, start - window + 1)              # W-1 samples of context
        block = rolling_statistics(signal[lo:stop], window, stats).values
        out[start:stop] = block[start - lo:]
        start = stop
    return pd.DataFrame(out, columns=list(stats))
