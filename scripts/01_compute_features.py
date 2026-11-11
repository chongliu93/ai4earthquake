#!/usr/bin/env python3
# =============================================================================
#   ai4earthquake :: scripts
# =============================================================================
#   Module       :: 01_compute_features.py -- Stage 1 (Module 1): rolling variance and kurtosis of the raw signal
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Stage 1 (Module 1): rolling variance and kurtosis of the raw signal.

Reads outputs/<system>/00_raw/record.h5 and writes
outputs/<system>/01_features/features.h5 with the record under key "data"
and the statistics (columns var, kurt) under key "features", row-aligned;
the first W-1 rows of the statistics are NaN.
"""
from common import parse
from labquake import features, io

args, cfg = parse(__doc__)
rec = io.load_record(io.stage_dir(cfg, "00_raw") / "record.h5")
W = int(cfg["window"])
print(f"{cfg['system']}: {len(rec):,} samples, W = {W:,} samples")
stats = features.rolling_statistics_chunked(rec["signal"].values, W)
out = io.stage_dir(cfg, "01_features") / "features.h5"
io.save_table(rec, out, "data", mode="w")
io.save_table(stats, out, "features", mode="a")
print(f"NaN rows: {int(stats['var'].isna().sum()):,} (expect {W - 1:,}) -> {out}")
