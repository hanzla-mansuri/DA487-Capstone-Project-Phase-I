"""Tests for the LSTM windows, SHAP values and fault flagging (Stages 5-8).

All tests use small made-up data, so they run without `data/`.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd

from src import explain, faults
from src.models import lstm
from tests.test_leakage import make_base


# ---------------------------------------------------------------- helpers

def make_table(rows_per_station=50):
    """Two made-up stations in the Stage 2 table format (hourly rows)."""
    rng = np.random.default_rng(0)
    parts = []
    for station in ["A", "B"]:
        index = pd.date_range("2022-09-29", periods=rows_per_station, freq="1h")
        part = pd.DataFrame(index=index)
        part["power_lag1"] = rng.uniform(0, 1, len(index))
        part["irr_lag1"] = rng.uniform(0, 1000, len(index))
        part["sun_elevation"] = 30.0
        part["target"] = rng.uniform(0, 1, len(index))
        part["target_is_filled"] = False
        part["target_is_fault"] = False
        part["usable"] = True
        part["station"] = station
        part["role"] = "seen"
        part["split"] = np.where(index <= pd.Timestamp("2022-09-30 23:59"), "train", "test")
        parts.append(part)
    return pd.concat(parts)


# ---------------------------------------------------------------- LSTM windows

def test_window_stays_inside_one_station():
    """The first `window - 1` rows of every station cannot end a window,
    otherwise the window would reach into the station before it."""
    table = make_table()
    ends = lstm.window_ends(table, 24, pd.Series(True, index=table.index))
    position = table.groupby("station").cumcount().to_numpy()
    assert position[ends].min() == 23
    assert len(ends) == 2 * (50 - 23)


def test_window_ends_at_the_target_row_and_has_no_later_rows():
    X = np.arange(100, dtype=np.float32).reshape(100, 1)
    batch = lstm.make_batch(X, np.array([30, 60]), 5).numpy()
    assert batch.shape == (2, 5, 1)
    assert list(batch[0, :, 0]) == [26, 27, 28, 29, 30]
    assert batch[1, -1, 0] == 60


def test_scaler_uses_train_rows_only():
    table = make_table()
    changed = table.copy()
    changed.loc[changed["split"] == "test", ["power_lag1", "irr_lag1"]] = 9999.0
    a = lstm.fit_scaler(table, ["train"])
    b = lstm.fit_scaler(changed, ["train"])
    for key in ["median", "mean", "std"]:
        assert np.allclose(a[key], b[key])


# ---------------------------------------------------------------- SHAP

def test_shap_values_add_up_to_the_prediction():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(500, 4)), columns=["power_lag1", "irr_lag1", "hour", "tilt_deg"])
    y = X["power_lag1"] * 2 + X["irr_lag1"] + rng.normal(scale=0.1, size=500)
    model = lgb.LGBMRegressor(n_estimators=20, verbose=-1).fit(X, y)
    shap, base = explain.shap_values(model.booster_, X)
    assert np.allclose(shap.sum(axis=1) + base, model.predict(X), atol=1e-6)
    # the two features that drive y are the most important ones
    assert set(explain.importance(shap).nlargest(2).index) == {"power_lag1", "irr_lag1"}


def test_feature_groups():
    assert explain.group_of("power_lag96") == "own power (lags)"
    assert explain.group_of("power_d1_daymean") == "own power (lags)"
    assert explain.group_of("csi_mean4") == "measured irradiance / clear-sky index"
    assert explain.group_of("clearsky_ghi") == "solar geometry"
    assert explain.group_of("hour_sin") == "calendar"
    assert explain.group_of("temperature_lag1") == "other weather"
    assert explain.group_of("tilt_deg") == "station metadata"


# ---------------------------------------------------------------- faults

def test_nowcast_features_do_not_use_own_power():
    """The key design point of Stage 8: no own-power input, so a lasting
    fault cannot hide in the features."""
    out = faults.features_nowcast(make_base("1h"))
    inputs = out.columns.drop(["target", "target_is_filled", "target_is_fault"])
    assert not [c for c in inputs if c.startswith("power")]

    # changing the power changes no input feature
    base = make_base("1h")
    lower = base.copy()
    lower["power_norm"] = lower["power_norm"] * 0.5
    a = faults.features_nowcast(base)[inputs]
    b = faults.features_nowcast(lower)[inputs]
    assert np.allclose(a.to_numpy(float), b.to_numpy(float), equal_nan=True)


def test_fault_factor():
    days = pd.date_range("2023-06-01", periods=10)
    start = pd.Timestamp("2023-06-04")
    drop = faults.fault_factor(days, start, 3, "drop", 0.2)
    assert list(drop) == [1, 1, 1, 0.8, 0.8, 0.8, 1, 1, 1, 1]
    gradual = faults.fault_factor(days, start, 3, "gradual", 0.3)
    assert np.allclose(gradual[3:6], [0.9, 0.8, 0.7])
    assert faults.fault_factor(days, start, 2, "zero", 1.0)[3] == 0


def make_daily(days=30):
    return pd.DataFrame({
        "station": "A", "day": pd.date_range("2023-06-01", periods=days),
        "actual": 4.0, "expected": 5.0, "irradiance_kwh": 6.0})


def test_score_is_zero_when_normal_and_drops_with_a_fault():
    daily = make_daily()
    normal = 4.0 / 5.0
    score = faults.rolling_score(daily["day"], faults.raw_ratio(daily, "model"), normal)
    assert np.allclose(score.dropna(), 0)

    factor = faults.fault_factor(daily["day"], pd.Timestamp("2023-06-10"), 7, "drop", 0.3)
    faulty = faults.rolling_score(daily["day"], faults.raw_ratio(daily, "model", factor), normal)
    assert np.isclose(faulty["2023-06-14"], -0.3)        # 3 full fault days in the window
    assert np.isclose(faulty["2023-06-05"], 0)           # before the fault
    assert np.isclose(faulty["2023-06-25"], 0)           # after the fault


def test_episodes_group_consecutive_flagged_days():
    scores = pd.DataFrame({
        "station": "A", "day": pd.date_range("2023-06-01", periods=8),
        "model": [0, -0.5, -0.5, 0, 0, -0.5, 0, 0]})
    runs = faults.episodes(scores, -0.2, "model")
    assert list(runs["days"]) == [2, 1]
    assert runs["start"].iloc[0] == pd.Timestamp("2023-06-02")
