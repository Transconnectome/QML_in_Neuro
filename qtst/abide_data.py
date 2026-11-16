"""ABIDE I resting-state fMRI ROI time series for the quantum time-series transformer demo.

Data source
-----------
Preprocessed Connectomes Project release of ABIDE I (Craddock et al., 2013), downloaded with
`nilearn.datasets.fetch_abide_pcp` (or `scripts/download_abide_parallel.py`, same files):
C-PAC pipeline, band-pass filtering on, global signal regression off, Craddock 200 (CC200) atlas,
quality-checked subjects only (nilearn's rule). Licence: ABIDE data are shared for non-commercial
research under CC BY-NC-SA; each user downloads the data directly from the source.

Outcome variable
----------------
`DX_GROUP` from the official ABIDE legend (ABIDE_LEGEND_V1.02): 1 = Autism, 2 = Control.
We recode it as y = 1 for autism and y = 0 for control, so the task is **binary classification**
(autism vs typically developing control). DX_GROUP has no missing values among the
quality-checked subjects.

What this module always reports
-------------------------------
How many subjects were excluded and why (too-short scans, empty files), the scan-length
distribution, and the label count per site, so that the sample never shrinks silently.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

OUTCOME_COLUMN = "DX_GROUP"
OUTCOME_CODING = {1: 1, 2: 0}  # ABIDE legend: 1 = Autism -> y=1, 2 = Control -> y=0


@dataclass
class LoadReport:
    n_quality_checked: int = 0
    n_file_found: int = 0
    n_empty: int = 0
    n_too_short: int = 0
    n_final: int = 0
    seq_len: int = 0
    n_rois: int = 0
    length_counts: dict = field(default_factory=dict)
    label_counts: dict = field(default_factory=dict)
    site_table: pd.DataFrame | None = None

    def render(self) -> str:
        top = sorted(self.length_counts.items(), key=lambda kv: -kv[1])[:8]
        lines = [
            "",
            "=" * 72,
            "ABIDE I data report  (outcome: DX_GROUP -> 1 = autism, 0 = control; binary classification)",
            "=" * 72,
            f"  quality-checked subjects in phenotype file : {self.n_quality_checked:,}",
            f"  ROI time-series file found                 : {self.n_file_found:,}",
            f"  empty or unreadable file (excluded)         : {self.n_empty:,}",
            f"  shorter than seq_len={self.seq_len} frames (excluded) : {self.n_too_short:,}",
            f"  subjects used                               : {self.n_final:,}",
            f"  ROIs per frame                              : {self.n_rois}",
            "  scan length in frames (most common)         : " + ", ".join(f"T={k}:{v}" for k, v in top),
            "  label counts                                : " + ", ".join(f"{k}: {v}" for k, v in self.label_counts.items()),
            "=" * 72,
        ]
        return "\n".join(lines)


def quality_checked(phenotype: pd.DataFrame) -> pd.DataFrame:
    """nilearn's `quality_checked=True` rule (nilearn/datasets/func.py)."""
    keep = (
        (phenotype["FILE_ID"] != "no_filename")
        & (phenotype["qc_rater_1"] == "OK")
        & phenotype["qc_anat_rater_2"].isin(["OK", "maybe"])
        & phenotype["qc_func_rater_2"].isin(["OK", "maybe"])
        & (phenotype["qc_anat_rater_3"] == "OK")
        & (phenotype["qc_func_rater_3"] == "OK")
    )
    return phenotype.loc[keep].copy()


def scan_cohort(data_dir: str, seq_len: int) -> tuple[list[dict], LoadReport]:
    """List usable subjects without keeping their arrays in memory.

    data_dir is nilearn's data_dir (the folder that contains `ABIDE_pcp`).
    """
    root = os.path.join(data_dir, "ABIDE_pcp")
    ts_dir = os.path.join(root, "cpac", "filt_noglobal")
    phenotype = pd.read_csv(os.path.join(root, "Phenotypic_V1_0b_preprocessed1.csv"))
    qc = quality_checked(phenotype)

    report = LoadReport(n_quality_checked=len(qc), seq_len=seq_len)
    files = {os.path.basename(p).replace("_rois_cc200.1D", ""): p for p in glob.glob(os.path.join(ts_dir, "*_rois_cc200.1D"))}

    entries = []
    for _, row in qc.iterrows():
        path = files.get(row["FILE_ID"])
        if path is None:
            continue
        report.n_file_found += 1
        try:
            arr = np.loadtxt(path, comments="#")
        except ValueError:
            arr = np.empty((0, 0))
        if arr.ndim != 2 or arr.shape[0] == 0:
            report.n_empty += 1
            continue
        report.length_counts[arr.shape[0]] = report.length_counts.get(arr.shape[0], 0) + 1
        report.n_rois = arr.shape[1]
        if arr.shape[0] < seq_len:
            report.n_too_short += 1
            continue
        entries.append(dict(
            file_id=row["FILE_ID"], site=row["SITE_ID"], path=path, n_frames=arr.shape[0],
            y=float(OUTCOME_CODING[int(row[OUTCOME_COLUMN])]),
        ))

    report.n_final = len(entries)
    y = np.array([e["y"] for e in entries])
    report.label_counts = {"autism (y=1)": int((y == 1).sum()), "control (y=0)": int((y == 0).sum())}
    sites = pd.DataFrame(entries).groupby("site")["y"].agg(autism="sum", n="count")
    sites["control"] = sites["n"] - sites["autism"]
    report.site_table = sites[["autism", "control"]].astype(int)
    return entries, report


def load_arrays(entries: list[dict], seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    """Read, crop and standardize the selected subjects.

    Each scan is cropped to its central `seq_len` frames. Each ROI is then z-scored within the
    subject (mean 0, SD 1 over time), because raw BOLD amplitude differs between scanners and
    sites. ROIs with zero variance in a subject (no coverage) become all zeros.

    Returns X of shape (N, seq_len, n_rois) and y of shape (N,).
    """
    xs = []
    for e in entries:
        arr = np.loadtxt(e["path"], comments="#")
        start = (arr.shape[0] - seq_len) // 2
        arr = arr[start : start + seq_len]
        sd = arr.std(axis=0, keepdims=True)
        arr = np.where(sd > 0, (arr - arr.mean(axis=0, keepdims=True)) / np.where(sd > 0, sd, 1.0), 0.0)
        xs.append(arr.astype(np.float32))
    return np.stack(xs), np.array([e["y"] for e in entries], dtype=np.float32)


def subsample(entries: list[dict], n_subjects: int, seed: int) -> list[dict]:
    """Draw n_subjects (stratified by diagnosis); a new draw for every seed. 0 = use everyone."""
    if n_subjects <= 0 or n_subjects >= len(entries):
        return entries
    y = np.array([e["y"] for e in entries])
    keep, _ = train_test_split(np.arange(len(entries)), train_size=n_subjects, random_state=seed, stratify=y)
    return [entries[i] for i in keep]


def split_indices(y, seed: int):
    """70 / 15 / 15 split stratified by diagnosis. Shared by every model so all see the same split."""
    idx_tr, idx_tmp = train_test_split(np.arange(len(y)), test_size=0.30, random_state=seed, stratify=y)
    idx_va, idx_te = train_test_split(idx_tmp, test_size=0.50, random_state=seed, stratify=y[idx_tmp])
    return idx_tr, idx_va, idx_te


def make_loaders(X, y, batch_size: int, seed: int):
    """DataLoaders for the split returned by `split_indices`."""
    idx_tr, idx_va, idx_te = split_indices(y, seed)

    def build(idx, shuffle):
        ds = TensorDataset(torch.tensor(X[idx]), torch.tensor(y[idx]))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(seed))

    return build(idx_tr, True), build(idx_va, False), build(idx_te, False), (len(idx_tr), len(idx_va), len(idx_te))
