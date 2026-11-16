"""Gradient-boosting models (Stage 4): LightGBM and XGBoost.

One global model per horizon, trained on all seen stations together, with
the same rows as the ridge baseline: seen stations, usable targets, daytime
(sun >= 5 deg), in the chosen splits. Tree models handle missing values
themselves, so there is no fill step and no scaler.

Tuning (PLAN Change log 2026-11-16): random search, fixed seed; each setting
is fitted on TRAIN with early stopping on VALIDATION; the setting with the
lowest validation RMSE wins and its best number of trees is kept for the
final refit on train + validation. The test set is not used here.
"""
import lightgbm as lgb
import numpy as np
import xgboost as xgb

from src.models.baselines import finish
from src.models.ridge import feature_columns, fit_rows

SEED = 42
MAX_TREES = 2000          # upper limit; early stopping picks the real number
EARLY_STOP = 50           # stop when validation has not improved for 50 trees

# Search spaces: each setting is drawn from these lists.
SPACE = {
    "lgbm": {
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12, 0.2],
        "num_leaves": [15, 31, 63, 127],
        "min_child_samples": [20, 50, 100, 200],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.0, 1.0, 10.0],
    },
    "xgb": {
        "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.12, 0.2],
        "max_depth": [4, 6, 8, 10],
        "min_child_weight": [1, 5, 20, 50],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_lambda": [0.1, 1.0, 10.0],
    },
}


def sample_settings(model_name, n, seed=SEED):
    """`n` random settings from the search space (reproducible)."""
    rng = np.random.default_rng(seed)
    settings = []
    for _ in range(n):
        one = {}
        for name, choices in SPACE[model_name].items():
            one[name] = choices[rng.integers(len(choices))]
            one[name] = one[name].item() if hasattr(one[name], "item") else one[name]
        settings.append(one)
    return settings


def make_model(model_name, params, n_trees, early_stop=False):
    """An untrained LightGBM or XGBoost regressor."""
    if model_name == "lgbm":
        return lgb.LGBMRegressor(
            n_estimators=n_trees, subsample_freq=1,
            random_state=SEED, n_jobs=4, verbose=-1, **params)
    return xgb.XGBRegressor(
        n_estimators=n_trees, tree_method="hist",
        early_stopping_rounds=EARLY_STOP if early_stop else None,
        random_state=SEED, n_jobs=4, **params)


def data(table, splits):
    """Features and target of the fit rows of `splits`."""
    rows = fit_rows(table, list(splits))
    return table.loc[rows, feature_columns(table)], table.loc[rows, "target"]


def fit_early_stopping(table, model_name, params):
    """Fit on train, stop early on validation. Returns (model, best trees)."""
    X, y = data(table, ["train"])
    Xv, yv = data(table, ["val"])
    model = make_model(model_name, params, MAX_TREES, early_stop=True)

    if model_name == "lgbm":
        model.fit(X, y, eval_X=(Xv,), eval_y=(yv,),
                  callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
        best_trees = model.best_iteration_
    else:
        model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
        best_trees = model.best_iteration + 1      # xgboost counts from 0
    return model, int(best_trees)


def fit_fixed(table, model_name, params, n_trees, splits):
    """Fit with a fixed number of trees (no early stopping), e.g. the final
    refit on train + validation."""
    X, y = data(table, splits)
    model = make_model(model_name, params, n_trees)
    model.fit(X, y)
    return model


def fit_lgbm(table, params=None, splits=("train",)):
    """LightGBM with library defaults (used by the Stage 4 vertical slice)."""
    X, y = data(table, splits)
    model = lgb.LGBMRegressor(random_state=SEED, n_jobs=4, verbose=-1, **(params or {}))
    model.fit(X, y)
    return model


def predict(model, table):
    """Prediction for every row, with the shared rules (cap, night = 0)."""
    pred = model.predict(table[feature_columns(table)])
    return finish(pred, table)
