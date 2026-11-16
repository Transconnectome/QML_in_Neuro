"""Parallel download of ABIDE CC200 ROI time series into nilearn's folder layout.

S3 throttles each connection (~40 KB/s here), so nilearn's one-file-at-a-time download takes
hours. This script fetches the same files with several connections at once, writes each file
atomically (temp file, then rename), and skips files that already exist. Afterwards
`fetch_abide_pcp(...)` finds every file locally and downloads nothing.

The subject list follows nilearn's `quality_checked=True` rule exactly
(nilearn/datasets/func.py): qc_rater_1 == OK, qc_anat_rater_2 and qc_func_rater_2 in {OK, maybe},
qc_anat_rater_3 == OK, qc_func_rater_3 == OK, and a preprocessed file exists.
"""
import argparse
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

BASE = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative"

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True, help="nilearn data_dir (the folder that contains ABIDE_pcp)")
p.add_argument("--workers", type=int, default=24)
args = p.parse_args()

root = os.path.join(args.data_dir, "ABIDE_pcp")
out_dir = os.path.join(root, "cpac", "filt_noglobal")
os.makedirs(out_dir, exist_ok=True)

csv_path = os.path.join(root, "Phenotypic_V1_0b_preprocessed1.csv")
if not os.path.exists(csv_path):
    urllib.request.urlretrieve(f"{BASE}/Phenotypic_V1_0b_preprocessed1.csv", csv_path)
ph = pd.read_csv(csv_path)

keep = (
    (ph["FILE_ID"] != "no_filename")
    & (ph["qc_rater_1"] == "OK")
    & ph["qc_anat_rater_2"].isin(["OK", "maybe"])
    & ph["qc_func_rater_2"].isin(["OK", "maybe"])
    & (ph["qc_anat_rater_3"] == "OK")
    & (ph["qc_func_rater_3"] == "OK")
)
file_ids = ph.loc[keep, "FILE_ID"].tolist()
print(f"quality-checked subjects: {len(file_ids)}")


def fetch(file_id):
    name = f"{file_id}_rois_cc200.1D"
    dest = os.path.join(out_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return "skip"
    tmp = dest + ".part"
    urllib.request.urlretrieve(f"{BASE}/Outputs/cpac/filt_noglobal/rois_cc200/{name}", tmp)
    os.replace(tmp, dest)
    return "ok"


counts = {"ok": 0, "skip": 0, "error": 0}
with ThreadPoolExecutor(max_workers=args.workers) as pool:
    futures = {pool.submit(fetch, f): f for f in file_ids}
    for i, fut in enumerate(as_completed(futures), 1):
        try:
            counts[fut.result()] += 1
        except Exception as e:  # report and continue; a re-run retries only the missing files
            counts["error"] += 1
            print(f"ERROR {futures[fut]}: {e}")
        if i % 100 == 0:
            print(f"{i}/{len(file_ids)} {counts}", flush=True)
print("done", counts)
