"""Small LSTM (Stage 5) for the 15-min and 1-h horizons.

Input of one sample: the last `WINDOW` rows of the Stage 2 feature table of
the same station, ending at the target row T. Row T-k holds the features
that were known when the forecast for T-k was issued, so every value in the
window is known at the issue time of T (no leakage; the same features as
ridge and the GBMs, only as a sequence).

Same rules as the ridge baseline:
- missing values -> train median, then standardised with train mean / std
  (both fitted on TRAIN rows only);
- trained on seen stations, usable daytime targets;
- predictions capped to [0, 1.2] and set to 0 at night.

Windows are built lazily: only one batch of windows exists in memory at a
time (PLAN Stage 5), never the full window tensor.
"""
import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.baselines import finish
from src.models.ridge import feature_columns, fit_rows

SEED = 42
WINDOW = {"15min": 16, "1h": 24}     # 4 hours of 15-min steps; 1 day of hourly steps
BATCH = 512
MAX_EPOCHS = 30
PATIENCE = 4                         # stop after 4 epochs without improvement
LEARNING_RATE = 1e-3


class Net(nn.Module):
    """One LSTM layer, then a linear layer on the last hidden state."""

    def __init__(self, n_features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def fit_scaler(table, splits):
    """Median, mean and std of every feature on the seen rows of `splits`."""
    rows = (table["role"] == "seen") & table["split"].isin(list(splits))
    X = table.loc[rows, feature_columns(table)]
    median = X.median()
    filled = X.fillna(median)
    std = filled.std().replace(0, 1.0)
    return {"median": median, "mean": filled.mean(), "std": std}


def scale(table, scaler):
    """Feature matrix (float32) of every row: filled and standardised."""
    X = table[feature_columns(table)].fillna(scaler["median"])
    X = (X - scaler["mean"]) / scaler["std"]
    return X.to_numpy(dtype=np.float32)


def window_ends(table, window, mask):
    """Row numbers that may end a window: `mask` is true and the station has
    at least `window` rows up to this one."""
    position = table.groupby("station").cumcount().to_numpy()
    ok = mask.to_numpy() & (position >= window - 1)
    return np.flatnonzero(ok)


def make_batch(X, ends, window):
    """Windows for the rows in `ends`: shape (batch, window, features).
    Built only for this batch (lazy)."""
    steps = np.arange(-window + 1, 1)
    return torch.from_numpy(X[ends[:, None] + steps])


def run_epoch(net, X, y, ends, window, optimizer=None, rng=None):
    """One pass over `ends`. Trains when an optimizer is given.
    Returns the RMSE over the pass."""
    if optimizer is not None:
        ends = rng.permutation(ends)
        net.train()
    else:
        net.eval()

    total, n = 0.0, 0
    size = BATCH if optimizer is not None else 4096
    for start in range(0, len(ends), size):
        part = ends[start:start + size]
        xb = make_batch(X, part, window)
        yb = torch.from_numpy(y[part])
        if optimizer is not None:
            loss = nn.functional.mse_loss(net(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        else:
            with torch.no_grad():
                loss = nn.functional.mse_loss(net(xb), yb)
        total += loss.item() * len(part)
        n += len(part)
    return float(np.sqrt(total / n))


def train(table, horizon, hidden, fit_splits, val_split=None, epochs=MAX_EPOCHS):
    """Train one LSTM.

    - With `val_split`: early stopping on that split; the best weights are
      kept. Returns (net, scaler, history) where history has one row per epoch.
    - Without: exactly `epochs` epochs (the final refit on train + validation).
    """
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    window = WINDOW[horizon]

    scaler = fit_scaler(table, fit_splits)
    X = scale(table, scaler)
    y = table["target"].to_numpy(dtype=np.float32)
    train_ends = window_ends(table, window, fit_rows(table, list(fit_splits)))
    if val_split:
        val_ends = window_ends(table, window, fit_rows(table, [val_split]))

    net = Net(X.shape[1], hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

    history, best, best_state, waited = [], np.inf, None, 0
    for epoch in range(1, epochs + 1):
        train_rmse = run_epoch(net, X, y, train_ends, window, optimizer, rng)
        row = {"epoch": epoch, "train_rmse": train_rmse}
        if val_split:
            row["val_rmse"] = run_epoch(net, X, y, val_ends, window)
            if row["val_rmse"] < best:
                best, waited = row["val_rmse"], 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
        history.append(row)
        print(f"      epoch {epoch:2d}: " + ", ".join(f"{k} {v:.5f}" for k, v in row.items() if k != "epoch"))
        if val_split and waited >= PATIENCE:
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    return net, scaler, pd.DataFrame(history)


def predict(net, scaler, table, horizon, mask=None):
    """Prediction for the rows in `mask` (default: all rows), NaN for rows
    without a full window, with the shared rules (cap, night = 0)."""
    window = WINDOW[horizon]
    if mask is None:
        mask = pd.Series(True, index=table.index)
    X = scale(table, scaler)
    ends = window_ends(table, window, mask)

    # `net` may also be an exported model wrapped as a function (Stage 6)
    if hasattr(net, "eval"):
        net.eval()
    out = np.full(len(table), np.nan, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(ends), 4096):
            part = ends[start:start + 4096]
            out[part] = net(make_batch(X, part, window)).numpy().ravel()
    pred = finish(out, table)
    pred[np.isnan(out)] = np.nan
    # night rows are 0 by rule even without a window
    pred[(table["sun_elevation"] < 5).to_numpy() & mask.to_numpy()] = 0.0
    return pred


def save(net, scaler, path, info):
    """Weights, scaler numbers and settings in one file."""
    torch.save({
        "state": net.state_dict(),
        "median": scaler["median"].to_numpy(),
        "mean": scaler["mean"].to_numpy(),
        "std": scaler["std"].to_numpy(),
        "features": list(scaler["median"].index),
        "info": info,
    }, path)


def load(path):
    """Returns (net, scaler, info) from a file written by `save`."""
    data = torch.load(path, weights_only=False)
    names = data["features"]
    scaler = {k: pd.Series(data[k], index=names) for k in ["median", "mean", "std"]}
    net = Net(len(names), data["info"]["hidden"])
    net.load_state_dict(data["state"])
    net.eval()
    return net, scaler, data["info"]
