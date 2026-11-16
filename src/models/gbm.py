"""Gradient-boosting models (Stage 4): LightGBM (and XGBoost, added later).

One global model per horizon, trained on all seen stations together, with
the same rows as the ridge baseline: seen stations, usable targets, daytime
(sun >= 5 deg), in the chosen splits. Tree models handle missing values
themselves, so there is no fill step and no scaler.
"""
import lightgbm as lgb

from src.models.baselines import finish
from src.models.ridge import feature_columns, fit_rows

SEED = 42


def fit_lgbm(table, params=None, splits=("train",)):
    """Fit a LightGBM regressor on the rows of `splits`.
    `params` are LightGBM settings; None = library defaults."""
    rows = fit_rows(table, list(splits))
    X = table.loc[rows, feature_columns(table)]
    y = table.loc[rows, "target"]

    settings = {"random_state": SEED, "n_jobs": 4, "verbose": -1}
    settings.update(params or {})
    model = lgb.LGBMRegressor(**settings)
    model.fit(X, y)
    return model


def predict(model, table):
    """Prediction for every row, with the shared rules (cap, night = 0)."""
    pred = model.predict(table[feature_columns(table)])
    return finish(pred, table)
