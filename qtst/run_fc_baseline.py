"""Classical reference: functional connectivity + logistic regression (no neural network).

For each participant, the Pearson correlation between every pair of the 200 regions over the
same cropped time window is computed (19,900 values, Fisher z-transformed). A logistic
regression on these values has 19,901 parameters, almost exactly the size of the quantum
time-series transformer (19,472 with 8 qubits). This is the standard approach for ABIDE
classification (e.g., Abraham et al., 2017). It ignores the order of time points: only which
regions fluctuate together matters.

Same participants, same subsamples and same 70/15/15 splits as `run_abide.py` (same seeds).
The regularization strength C is chosen on the validation set; the test set is used once.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abide_data import load_arrays, scan_cohort, split_indices, subsample  # noqa: E402

C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def connectivity_features(X: np.ndarray) -> np.ndarray:
    """(N, T, R) z-scored time series -> (N, R(R-1)/2) Fisher-z correlations."""
    n, t, r = X.shape
    iu = np.triu_indices(r, k=1)
    feats = np.empty((n, len(iu[0])), dtype=np.float32)
    for i in range(n):
        corr = (X[i].T @ X[i]) / t  # X is z-scored per region, so this is the Pearson correlation
        feats[i] = np.arctanh(np.clip(corr[iu], -0.999, 0.999))  # zero-variance regions give 0
    return feats


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seq-len", type=int, required=True)
    p.add_argument("--n-subjects", type=int, default=100, help="0 = all subjects")
    p.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    entries, report = scan_cohort(args.data_dir, args.seq_len)
    print(report.render())

    tests = []
    for seed in args.seeds:
        np.random.seed(seed)
        picked = subsample(entries, args.n_subjects, seed)
        X, y = load_arrays(picked, args.seq_len)
        F = connectivity_features(X)
        i_tr, i_va, i_te = split_indices(y, seed)
        scaler = StandardScaler().fit(F[i_tr])
        Ftr, Fva, Fte = scaler.transform(F[i_tr]), scaler.transform(F[i_va]), scaler.transform(F[i_te])

        best = None
        for C in C_GRID:
            clf = LogisticRegression(C=C, max_iter=5000).fit(Ftr, y[i_tr])
            val = roc_auc_score(y[i_va], clf.decision_function(Fva))
            if best is None or val > best[0]:
                best = (val, C, clf)
        val, C, clf = best
        row = dict(
            model="fc_logreg", seed=seed, n_subjects=args.n_subjects or len(entries), seq_len=args.seq_len,
            qubits=None, n_params=int(clf.coef_.size + clf.intercept_.size), C=C,
            test_auroc=float(roc_auc_score(y[i_te], clf.decision_function(Fte))), val_auroc_best=float(val),
            train_auroc=float(roc_auc_score(y[i_tr], clf.decision_function(Ftr))),
            n_train=len(i_tr), n_val=len(i_va), n_test=len(i_te), seconds=None,
        )
        tests.append(row["test_auroc"])
        print(f"[seed {seed}] C={C:g}  val {val:.4f}  test {row['test_auroc']:.4f}  train {row['train_auroc']:.4f}")
        with open(os.path.join(args.out_dir, f"seed{seed}.json"), "w") as f:
            json.dump(row, f, indent=1)

    tests = np.array(tests)
    print(f"\nSUMMARY fc_logreg: test AUROC {tests.mean():.4f} ± {tests.std(ddof=1):.4f} over {len(tests)} seeds, "
          f"{row['n_params']:,} parameters, {row['n_test']} test subjects per seed")


if __name__ == "__main__":
    main()
