"""Download ABIDE I preprocessed ROI time series (Preprocessed Connectomes Project).

Settings used in the talk demo:
  pipeline            = C-PAC
  band-pass filtering = on   (0.01-0.1 Hz)
  global signal regr. = off
  atlas               = Craddock 200 (CC200), 200 ROIs
  quality_checked     = True (nilearn's rule: rater 1 OK, rater 2 OK/maybe, rater 3 OK)

nilearn skips files that already exist, so re-running resumes an interrupted download.
"""
import argparse

from nilearn.datasets import fetch_abide_pcp

p = argparse.ArgumentParser()
p.add_argument("--data-dir", required=True)
args = p.parse_args()

abide = fetch_abide_pcp(
    data_dir=args.data_dir,
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=["rois_cc200"],
    quality_checked=True,
    verbose=1,
)
print(f"downloaded subjects: {len(abide.rois_cc200)}")
print(f"first file shape (T, ROI): {abide.rois_cc200[0].shape}")
