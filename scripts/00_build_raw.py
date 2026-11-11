#!/usr/bin/env python3
# =============================================================================
#   ai4earthquake :: scripts
# =============================================================================
#   Module       :: 00_build_raw.py -- Stage 0: assemble the raw record time, signal, target of one system
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""Stage 0: assemble the raw record time, signal, target of one system.

Laboratory (raw.kind = pennstate): reads the Penn State AE files and the
mechanical record and interpolates the shear stress onto every AE sample
(labquake.pennstate). Kīlauea (raw.kind = kilauea): reads the outputs of
the Johnson & Johnson (2024) preprocessing and interpolates the CRIM
displacement onto the 100-Hz seismic samples (labquake.kilauea).

Output: outputs/<system>/00_raw/record.h5, key "data".
Memory: about 10^9 samples for a laboratory experiment (25 GB as float64);
run on a machine with enough memory or restrict raw.start_s / raw.end_s.
"""
import pathlib

from common import parse
from labquake import io, kilauea, pennstate

args, cfg = parse(__doc__)
raw, base = cfg["raw"], pathlib.Path(cfg["paths"]["raw"])

if raw["kind"] == "pennstate":
    rec = pennstate.build_record(
        raw["start_s"], raw["end_s"], raw["channel"],
        str(base / raw["ac_dir"]), str(base / raw["time_vector_file"]),
        str(base / raw["settings_file"]), str(base / raw["mechanical_file"]))
elif raw["kind"] == "kilauea":
    rec = kilauea.build_record(
        str(base / raw["seismic_h5"]), [str(base / p) for p in raw["displacement_parquet"]],
        component=raw.get("component", "HHN"), start=raw["start"], end=raw["end"])
else:
    raise SystemExit(f"unknown raw.kind {raw['kind']!r}")

out = io.stage_dir(cfg, "00_raw") / "record.h5"
io.save_table(rec, out, "data")
print(f"{cfg['system']}: {len(rec):,} samples, {rec.time.iloc[0]:.2f}-{rec.time.iloc[-1]:.2f} s -> {out}")
