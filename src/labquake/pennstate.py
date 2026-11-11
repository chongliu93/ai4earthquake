# =============================================================================
#   ai4earthquake :: src/labquake
# =============================================================================
#   Module       :: pennstate.py -- The laboratory experiments: assembling the raw record (stage 0)
#   Author       :: Chong Liu (c.liu3@uu.nl)
#   Affiliation  :: Utrecht University
#   Date         :: November 2026
# =============================================================================
"""The laboratory experiments: assembling the raw record (stage 0).

The Penn State rock-mechanics archive (http://psudata.s3-website.us-east-2.amazonaws.com/)
holds, per experiment, the acoustic-emission (AE) waveforms as a series of
binary ``WF_<n>.ac`` files, an ``ACTimeVector`` file (the start time of each
waveform segment and the sample interval ``ts``), an acquisition-settings
``.mat`` file, and the mechanical record as a text file with the shear
stress at 1 kHz (ten samples per millisecond time stamp). This module reads
those files and builds one table time, signal, target with the shear stress
linearly interpolated onto every AE sample, as the runs behind the paper
did (originally ``LabQuakeAI_pennstate.py`` of R. Elbertsen).

The record segments used in the paper (Supplementary Table 1) are p4581 3816-4082 s,
p4679 4340-4539 s and p5198 6600-6898 s (experiment clock).
"""
from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd


def load_ac_time_vector(filename):
    """Start time of each waveform segment (s) and the sample interval (s)."""
    if filename.endswith(".mat"):
        from scipy.io import loadmat
        m = loadmat(filename)
        ac_time = m["acTime"][0]
        ts = m["ts_adjusted"][0][0] if "ts_adjusted" in m else m["ts"][0][0]
    else:
        ac_time = np.loadtxt(filename)
        ts = 0.2520                                   # microseconds
    return ac_time, ts / 1e6


def load_ac_settings(filename):
    """Layout of the .ac files from the acquisition-settings .mat file."""
    from scipy.io import loadmat
    m = loadmat(filename)
    n_sf_per_file = int(m["numFrames"][0][0] / 2)     # superframes per file
    n_wf_per_sf = m["numAcqs"][0][0]                  # waveforms per superframe and channel
    n_wf_per_file = int(n_sf_per_file * n_wf_per_sf)  # waveforms per file and channel
    n_ch = len(m["channels2save"][0])
    wf_len = m["Nsamples"][0][0]                      # samples per waveform
    return n_sf_per_file, n_wf_per_file, n_ch, wf_len


def _file_number(t, ac_time, n_wf_per_file):
    if not ac_time[0] <= t <= ac_time[-1]:
        raise ValueError(f"no acoustic data at t = {t}")
    return ceil(np.argwhere(ac_time > t)[0][0] / n_wf_per_file)


def load_ac_file(path, channel, n_ch, n_sf_per_file):
    """All samples of one channel of one WF_<n>.ac file (int16)."""
    ac = np.fromfile(path, np.int16)
    ac = np.reshape(ac, (-1, n_ch, n_sf_per_file), order="F")
    ac = np.transpose(ac, [0, 2, 1])
    ac = np.reshape(ac, (-1, n_ch, 1), order="F")
    return ac[:, channel, 0]


def ae_timeseries(start, end, channel, ac_dir, time_vector_file, settings_file):
    """The AE record of one channel between two experiment times: (time, signal)."""
    ac_time, ts = load_ac_time_vector(time_vector_file)
    n_sf, n_wf, n_ch, wf_len = load_ac_settings(settings_file)
    first, last = _file_number(start, ac_time, n_wf), _file_number(end, ac_time, n_wf)
    numbers = np.arange(first, last) if first < last else [first]
    times, sig = [], []
    wf_clock = np.arange(wf_len) * ts
    for k in numbers:
        sig.append(load_ac_file(f"{ac_dir}/WF_{k:d}.ac", channel, n_ch, n_sf))
        t = np.empty(wf_len * n_wf)
        for i in range(n_wf):
            t[i * wf_len:(i + 1) * wf_len] = ac_time[(k - 1) * n_wf + i] + wf_clock
        times.append(t)
    time, signal = np.concatenate(times), np.concatenate(sig)
    m = (time >= start) & (time <= end)
    return time[m], signal[m]


def load_mechanical(filename, start, end):
    """Shear stress (MPa) and its time (s) from the mechanical text file.

    The file stamps ten samples per millisecond with the same time; each
    group is spread over the millisecond, and groups with fewer than ten
    samples are padded with their mean, as in the original reader.
    """
    with open(filename) as f:
        first = f.readline().split()
        second = f.readline()
        third = f.readline().split()
    if first and first[0] == "number":
        data = pd.read_csv(filename, sep=r"\s+", skiprows=5, names=third)
    else:
        cols = open(filename).readline().rstrip("\n").split(", ")
        data = pd.read_csv(filename, skiprows=2, names=cols)
    data = data[(data.Time >= start) & (data.Time <= end)]
    counts = data["Time"].value_counts()
    short = counts[counts != 10]
    for t_missing, n_have in short.items():
        block = data[data.Time == t_missing]
        fill = block.mean()
        fill["Time"] = t_missing
        pad = pd.DataFrame([fill.values] * (10 - n_have), columns=block.columns,
                           index=np.mean(block.index) + 0.1 + np.arange(10 - n_have) / 100)
        data = pd.concat([data, pad])
    data = data.sort_index()
    time = data["Time"].values.astype(np.float64).copy()
    for i in range(0, len(time), 10):
        time[i:i + 10] += np.arange(0, 10)[:len(time[i:i + 10])] * 0.001
    return data["Shr_stress"].values.astype(np.float64), time


def build_record(start, end, channel, ac_dir, time_vector_file, settings_file,
                 mechanical_file):
    """time, signal (AE counts), target (shear stress, MPa) at every AE sample."""
    t_ae, ae = ae_timeseries(start, end, channel, ac_dir, time_vector_file, settings_file)
    tau, t_mech = load_mechanical(mechanical_file, start, end)
    target = np.interp(t_ae, t_mech, tau)
    return pd.DataFrame({"time": t_ae, "signal": ae, "target": target})
