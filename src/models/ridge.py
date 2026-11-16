"""Ridge regression baseline (Stage 3).

One global model per horizon, trained on all seen stations together.
Steps (a scikit-learn pipeline, so every step is fitted on TRAIN rows only):
    1. fill missing feature values with the train median
       (e.g. `tilt_deg` of the `Mixed` stations, missing lags);
    2. standardise every feature (train mean and standard deviation);
    3. ridge regression (linear regression with an L2 penalty `alpha`).

Rows used for fitting: role == "seen", usable target, daytime (sun >= 5 deg).
Night rows are left out because their target is 0 and the prediction is set
to 0 anyway (PLAN §3).
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.models.baselines import finish

NOT_FEATURES = ["target", "target_is_filled", "target_is_fault", "usable",
                "station", "role", "split"]
ALPHAS = [0.1, 1, 10, 100, 1000, 10000, 100000]


def feature_columns(table):
    """All columns that are model inputs."""
    return [c for c in table.columns if c not in NOT_FEATURES]


def fit_rows(table, splits):
    """Rows a model may learn from: seen stations, usable daytime targets,
    in the given splits (e.g. ["train"] or ["train", "val"])."""
    return (
        (table["role"] == "seen")
        & table["usable"]
        & (table["sun_elevation"] >= 5)
        & table["split"].isin(splits)
    )


def fit(table, alpha, splits=("train",)):
    """Fit the pipeline on the rows of `splits` only."""
    rows = fit_rows(table, list(splits))
    X = table.loc[rows, feature_columns(table)]
    y = table.loc[rows, "target"]

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        Ridge(alpha=alpha),
    )
    model.fit(X, y)
    return model


def predict(model, table):
    """Prediction for every row, with the shared rules (cap, night = 0)."""
    pred = model.predict(table[feature_columns(table)])
    return finish(pred, table)


def choose_alpha(table, alphas=ALPHAS):
    """Fit on train, score on validation (seen, usable, daytime); the alpha
    with the lowest validation RMSE wins. Returns (best alpha, scores)."""
    val = fit_rows(table, ["val"])
    scores = {}
    for alpha in alphas:
        model = fit(table, alpha)
        err = predict(model, table)[val] - table.loc[val, "target"]
        scores[alpha] = float(np.sqrt((err ** 2).mean()))
    best = min(scores, key=scores.get)
    return best, scores
