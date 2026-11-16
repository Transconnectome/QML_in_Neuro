"""Train the quantum time-series transformer or a parameter-matched classical model on ABIDE.

Task: autism (DX_GROUP = 1) vs control (DX_GROUP = 2) from resting-state fMRI ROI time series,
binary classification, metric AUROC. Adapted from the Connectome Lab challenge
`challenges/fmri-quantum-ts-transformer/train.py`; the models (`qts_transformer.py`,
`baselines.py`) are used unchanged.

Long-run safety: a full training-state checkpoint (model, optimizer, scheduler, best weights,
RNG states) is written after every epoch, and finished seeds are stored as JSON. Re-running the
same command (or passing --resume, the default) continues from the last completed epoch.

Example
-------
    python qtst/run_abide.py --data-dir data/nilearn_data --model quantum --qubits 8 \
        --n-subjects 100 --seeds 2024 2025 2026 --out-dir results/n100/quantum
    python qtst/run_abide.py --data-dir data/nilearn_data --model small_transformer --match-quantum \
        --qubits 8 --n-subjects 100 --seeds 2024 2025 2026 --out-dir results/n100/small_transformer
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from abide_data import load_arrays, make_loaders, scan_cohort, subsample  # noqa: E402
from baselines import build_classical, count_trainable_parameters  # noqa: E402
from qts_transformer import QuantumTSTransformer  # noqa: E402


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(args, feature_dim: int, n_timesteps: int):
    def quantum():
        return QuantumTSTransformer(
            n_qubits=args.qubits, n_timesteps=n_timesteps, degree=args.degree,
            n_ansatz_layers=args.ansatz_layers, feature_dim=feature_dim, output_dim=1,
            dropout=args.dropout, device=torch.device("cpu"),
        )

    if args.model == "quantum":
        model = quantum()
        return model, count_trainable_parameters(model)
    budget = count_trainable_parameters(quantum()) if args.match_quantum else args.param_budget
    return build_classical(args.model, feature_dim=feature_dim, n_timesteps=n_timesteps,
                           output_dim=1, dropout=args.dropout, param_budget=budget)


@torch.no_grad()
def evaluate(model, loader) -> float:
    model.eval()
    p, t = [], []
    for xb, yb in loader:
        p.append(model(xb).flatten())
        t.append(yb.flatten())
    t, p = torch.cat(t).numpy(), torch.cat(p).numpy()
    return float(roc_auc_score(t, p)) if len(np.unique(t)) == 2 else float("nan")


def train_one_seed(args, entries, seed):
    done_path = os.path.join(args.out_dir, f"seed{seed}.json")
    if os.path.exists(done_path):
        print(f"[seed {seed}] already finished -> {done_path}")
        with open(done_path) as f:
            return json.load(f)

    set_all_seeds(seed)
    picked = subsample(entries, args.n_subjects, seed)
    X, y = load_arrays(picked, args.seq_len)
    tr, va, te, sizes = make_loaders(X, y, args.batch_size, seed)

    set_all_seeds(seed)
    model, n_params = build_model(args, X.shape[2], X.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-8)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs))

    ckpt_path = os.path.join(args.out_dir, f"seed{seed}_ckpt.pt")
    start_epoch, best_val, best_state, elapsed_before = 1, None, None, 0.0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        best_val, best_state, elapsed_before = ck["best_val"], ck["best_state"], ck["elapsed"]
        torch.set_rng_state(ck["torch_rng"]); np.random.set_state(ck["np_rng"]); random.setstate(ck["py_rng"])
        tr.generator.set_state(ck["loader_rng"])
        start_epoch = ck["epoch"] + 1
        print(f"[seed {seed}] resumed after epoch {ck['epoch']}")

    print(f"\n[seed {seed}] train {sizes[0]} / val {sizes[1]} / test {sizes[2]} subjects, "
          f"{n_params:,} trainable parameters")
    t0 = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total, nb = 0.0, 0
        for xb, yb in tr:
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb).float(), yb.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            total += float(loss.item()); nb += 1
        sched.step()

        val = evaluate(model, va)
        better = not np.isnan(val) and (best_val is None or val > best_val)
        if better:
            best_val = val
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  epoch {epoch:2d}/{args.epochs}  train_loss={total / max(nb, 1):.4f}  val_AUROC={val:.4f}{'  *' if better else ''}", flush=True)

        torch.save(dict(
            epoch=epoch, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
            best_val=best_val, best_state=best_state, elapsed=elapsed_before + time.time() - t0,
            torch_rng=torch.get_rng_state(), np_rng=np.random.get_state(), py_rng=random.getstate(),
            loader_rng=tr.generator.get_state(),
        ), ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    if best_state is not None:
        model.load_state_dict(best_state)
    row = dict(
        model=args.model, seed=seed, n_subjects=args.n_subjects or len(entries), seq_len=args.seq_len,
        qubits=args.qubits if args.model == "quantum" or args.match_quantum else None,
        n_params=n_params, test_auroc=evaluate(model, te), val_auroc_best=best_val,
        train_auroc=evaluate(model, tr), n_train=sizes[0], n_val=sizes[1], n_test=sizes[2],
        seconds=round(elapsed_before + time.time() - t0, 1),
    )
    print(f"  -> test AUROC {row['test_auroc']:.4f} (best val {best_val:.4f}, train {row['train_auroc']:.4f})")
    with open(done_path, "w") as f:
        json.dump(row, f, indent=1)
    os.remove(ckpt_path)
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, help="nilearn data_dir that contains ABIDE_pcp")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seq-len", type=int, required=True, help="frames per subject; shorter scans are excluded and counted")
    p.add_argument("--model", default="quantum", choices=["quantum", "small_transformer", "conv1d", "mean_linear"])
    p.add_argument("--match-quantum", action="store_true", help="match the classical model's parameter count to the quantum model")
    p.add_argument("--param-budget", type=int, default=None)
    p.add_argument("--qubits", type=int, default=8)
    p.add_argument("--degree", type=int, default=2)
    p.add_argument("--ansatz-layers", type=int, default=3)
    p.add_argument("--n-subjects", type=int, default=100, help="0 = all subjects")
    p.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    os.makedirs(args.out_dir, exist_ok=True)
    entries, report = scan_cohort(args.data_dir, args.seq_len)
    print(report.render())
    rows = [train_one_seed(args, entries, s) for s in args.seeds]

    tests = np.array([r["test_auroc"] for r in rows])
    print(f"\nSUMMARY {args.model}: test AUROC {tests.mean():.4f} ± {tests.std(ddof=1) if len(tests) > 1 else 0:.4f} "
          f"over {len(rows)} seeds, {rows[0]['n_params']:,} parameters, {rows[0]['n_test']} test subjects per seed")


if __name__ == "__main__":
    main()
