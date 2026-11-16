# Quantum Machine Learning in Neuroscience Research — code examples

Two notebooks for the talk. Both run on a CPU; quantum circuits are simulated exactly with PennyLane, so no quantum hardware or account is needed.

| Notebook | What it shows | Data |
| --- | --- | --- |
| `notebooks/QCNN_demo.ipynb` | How a quantum circuit becomes a trainable layer (quantum convolutional neural network), compared with classical models of the same size | MNIST digits 0 vs 1, CIFAR-10 airplane vs automobile (downloaded by torchvision) |
| `notebooks/QTST_ABIDE_demo.ipynb` | Quantum time-series transformer on resting-state fMRI, compared with parameter-matched classical models | ABIDE I preprocessed ROI time series (downloaded inside the notebook) |

Both notebooks are self-contained: every line of code they use is inside them, and they run from a single file (for example on Google Colab). The ABIDE notebook trains one quantum model live and loads the other results, which were produced in advance by the scripts in `qtst/` with the same code; when the `qtst/` and `results/` folders are present, the notebook checks that its code gives identical outputs and splits and that its results table matches the saved files.

## Outcome variable for ABIDE

`DX_GROUP` from the official ABIDE legend (ABIDE_LEGEND_V1.02): 1 = Autism, 2 = Control. It is recoded as y = 1 (autism) and y = 0 (control), so the task is **binary classification**, evaluated with AUROC. The notebook lists the other candidate variables and why they were not used.

## Layout

| Path | Content |
| --- | --- |
| `qtst/qts_transformer.py`, `qtst/baselines.py` | Quantum time-series transformer and parameter-matched classical models, unchanged from the Connectome Lab challenge `fmri-quantum-ts-transformer` |
| `qtst/abide_data.py` | ABIDE loader: quality-checked participants, outcome coding, scan-length cropping with exclusion report, z-scoring, stratified splits |
| `qtst/run_abide.py` | Training script with a checkpoint after every epoch (re-running resumes) |
| `qtst/run_fc_baseline.py` | Functional connectivity + logistic regression (19,901 parameters), same participants and splits |
| `results/` | Per-seed results (`n100/` = 100 participants per seed, `all/` = all 846) |
| `run_all.sh` | The exact commands behind every ABIDE result in the notebook |
| `scripts/download_abide.py`, `scripts/download_abide_parallel.py` | ABIDE download with nilearn, or the same files with parallel connections (faster) |
| `notebooks/*.py` | Plain-text (jupytext) source of each notebook |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch pennylane scikit-learn nilearn pandas matplotlib torchvision jupyter
python scripts/download_abide_parallel.py --data-dir data/nilearn_data
bash run_all.sh
```

ABIDE data are shared for non-commercial research (CC BY-NC-SA); download them directly from the source rather than redistributing them.
