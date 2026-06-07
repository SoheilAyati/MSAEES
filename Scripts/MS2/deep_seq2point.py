"""
deep_seq2point.py  --  Milestone 2, Stage 5 (deep-learning model)
================================================================

Reference seq2point disaggregation network (Zhang et al., AAAI 2018) adapted to
the PAC4200 channel set.  This is the deep-learning side of the classical ->
deep progression: it is permitted here because lab NILM is not a "critical
infrastructure safety component" under the EU AI Act (it is monitoring /
informational, an excluded category).

seq2point idea: slide a window of length W over the aggregate signal; a CNN maps
the whole window to the *single* target value (one appliance's power) at the
window midpoint.  One model per appliance; the aggregate is the input, the
per-appliance P_contribution from /ground_truth is the target -- exactly the
supervision our synthetic data provides for free.

This file is intentionally dependency-light to read but needs torch to RUN:
    pip install torch
It is NOT executed by run_ms2_demo.py (which stays sklearn-only). Use train_demo()
below once torch is available and several scenarios have been generated.

Input channels (configurable): we recommend [P_total, Q_total, THD_I_L1] so the
network can exploit the same multi-feature fusion the classical models use.
For synthetic->real transfer keep to COMMON_CHANNELS (nilm_io).
"""

from __future__ import annotations

import numpy as np

WINDOW = 599          # odd window; midpoint is the prediction target
INPUT_CHANNELS = ["P_total", "Q_total", "THD_I_L1"]


def make_windows(X, y, window=WINDOW, stride=1):
    """X: (T, C) aggregate inputs;  y: (T,) target appliance power.

    Returns (N, C, window) input tensors and (N,) midpoint targets.
    Standardisation should be applied by the caller (store the stats!).
    """
    T = X.shape[0]
    half = window // 2
    idx = np.arange(half, T - half, stride)
    Xw = np.stack([X[i - half:i + half + 1].T for i in idx])   # (N, C, W)
    yw = y[idx]
    return Xw.astype(np.float32), yw.astype(np.float32)


# --------------------------------------------------------------------------
# Torch model (imported lazily so the file is importable without torch)
# --------------------------------------------------------------------------
def build_model(n_channels=len(INPUT_CHANNELS), window=WINDOW):
    import torch.nn as nn

    class Seq2Point(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(n_channels, 30, 10, padding="same"), nn.ReLU(),
                nn.Conv1d(30, 40, 8, padding="same"), nn.ReLU(),
                nn.Conv1d(40, 50, 6, padding="same"), nn.ReLU(),
                nn.Conv1d(50, 50, 5, padding="same"), nn.ReLU(),
                nn.Conv1d(50, 50, 5, padding="same"), nn.ReLU(),
            )
            self.head = nn.Sequential(
                nn.Flatten(), nn.Linear(50 * window, 1024), nn.ReLU(),
                nn.Linear(1024, 1),
            )

        def forward(self, x):
            return self.head(self.conv(x)).squeeze(-1)

    return Seq2Point()


def train_demo(scenario_paths, appliance, epochs=10, batch=256, lr=1e-3,
               device="cpu"):
    """Minimal training loop for one appliance across several scenarios.

    Returns the trained model and (input_mean, input_std, target_max) needed to
    de-normalise predictions.  Requires torch and >1 scenario for a real test.
    """
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from nilm_io import load_scenario

    Xs, ys = [], []
    for p in scenario_paths:
        scn = load_scenario(p)
        X = np.stack([scn.meas[c] for c in INPUT_CHANNELS], axis=1)  # (T, C)
        a = scn.appliance_names.index([n for n in scn.appliance_names
                                       if n.startswith(appliance)][0])
        y = scn.P_contribution[:, a]
        Xw, yw = make_windows(X, y)
        Xs.append(Xw); ys.append(yw)
    Xw = np.concatenate(Xs); yw = np.concatenate(ys)

    mu, sd = Xw.mean((0, 2), keepdims=True), Xw.std((0, 2), keepdims=True) + 1e-6
    tmax = float(np.abs(yw).max() + 1e-6)
    Xw = (Xw - mu) / sd
    yw = yw / tmax

    ds = TensorDataset(torch.tensor(Xw), torch.tensor(yw))
    dl = DataLoader(ds, batch_size=batch, shuffle=True)
    model = build_model().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    for ep in range(epochs):
        model.train(); tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward(); opt.step()
            tot += loss.item() * len(xb)
        print(f"epoch {ep+1}/{epochs}  mse={tot/len(ds):.4f}")
    return model, (mu, sd, tmax)


if __name__ == "__main__":
    print(__doc__)
    print("This module is a reference implementation; run train_demo() with torch "
          "installed and multiple generated scenarios.")
