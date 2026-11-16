"""Leakage tests (Stage 2). Run from the repo root with:  python -m pytest

The feature tests use small made-up data, so they run without `data/`.
The idea of the main test: replace every MEASURED value from the forecast
issue time onward with junk. If any feature at that issue time changes, the
feature used information it could not have had (leakage).
"""
import numpy as np
import pandas as pd
import pytest

from src import features as f
from src import preprocess as p
from src import splits as s

LAT, LON = 22.34, 114.26
MEASURED = ["power_norm", "irradiance", "irr_high_share", "temperature",
            "pressure", "visibility", "wind_speed", "wind_direction", "csi"]
NOT_FEATURES = ["target", "target_is_filled", "target_is_fault"]


# ---------------------------------------------------------------- helpers

def make_base(freq, days=20):
    """A made-up station on a regular grid, built with the real build_base."""
    index = pd.date_range("2022-06-01", periods=int(days * pd.Timedelta("1D") / pd.Timedelta(freq)), freq=freq)
    rng = np.random.default_rng(0)

    power = pd.DataFrame(index=index)
    power["power_norm"] = rng.uniform(0, 1, len(index))
    power["is_filled"] = False

    weather = pd.DataFrame(index=index)
    weather["irradiance"] = rng.uniform(0, 1000, len(index))
    weather["irr_high_share"] = 0.0
    weather["temperature"] = rng.uniform(20, 35, len(index))
    weather["pressure"] = rng.uniform(1000, 1010, len(index))
    weather["visibility"] = rng.uniform(1, 15, len(index))
    weather["wind_speed"] = rng.uniform(0, 10, len(index))
    weather["wind_direction"] = rng.uniform(0, 360, len(index))

    return f.build_base(power, weather, freq, LAT, LON)


def junk_from(base, cut):
    """Copy of `base` with every measured value at/after `cut` set to 9999."""
    junk = base.copy()
    junk.loc[junk.index >= cut, MEASURED] = 9999.0
    return junk


def changed_features(builder, base, T, cut):
    """Names of the features at row T that change when data from `cut` on is junk."""
    a = builder(base).loc[T].drop(NOT_FEATURES)
    b = builder(junk_from(base, cut)).loc[T].drop(NOT_FEATURES)
    a = a.astype(float).to_numpy()
    b = b.astype(float).to_numpy()
    same = np.isclose(a, b, equal_nan=True)
    names = builder(base).columns.drop(NOT_FEATURES)
    return list(names[~same])


# ------------------------------------- 1. no feature uses data from >= T

@pytest.mark.parametrize("builder, freq", [
    (f.features_15min, "15min"),
    (f.features_1h, "1h"),
])
def test_intraday_features_use_only_the_past(builder, freq):
    base = make_base(freq)
    for T in pd.date_range("2022-06-10 06:00", "2022-06-10 18:00", freq="3h"):
        assert changed_features(builder, base, T, cut=T) == []


def test_dayahead_features_use_only_data_before_23h():
    base = make_base("1h")
    issue = pd.Timestamp("2022-06-09 23:00")  # forecast for 2022-06-10
    for hour in range(24):
        T = pd.Timestamp("2022-06-10") + pd.Timedelta(hours=hour)
        assert changed_features(f.features_dayahead, base, T, cut=issue) == []


def test_dayahead_hour23_has_no_d1_lag():
    out = f.features_dayahead(make_base("1h"))
    assert out.loc[out.index.hour == 23, "power_d1"].isna().all()


# --------------------- 2. control: the test above is able to find a leak

def test_control_cut_one_step_early_is_detected():
    base = make_base("15min")
    T = pd.Timestamp("2022-06-10 12:00")
    changed = changed_features(f.features_15min, base, T, cut=T - pd.Timedelta("15min"))
    assert "power_lag1" in changed


def test_control_dayahead_cut_at_22h_is_detected():
    base = make_base("1h")
    T = pd.Timestamp("2022-06-10 12:00")
    changed = changed_features(f.features_dayahead, base, T, cut=pd.Timestamp("2022-06-09 22:00"))
    assert "temperature_d1_22h" in changed


# ------------------------------- 3. no filled or faulty target is used

def test_usable_target_drops_filled_fault_and_missing():
    out = pd.DataFrame({
        "target":           [0.5, 0.5, 0.0, np.nan, 0.4],
        "target_is_filled": [False, True, False, False, False],
        "target_is_fault":  [False, False, True, False, False],
    })
    assert f.usable_target(out).tolist() == [True, False, False, False, True]


def test_filled_rows_are_marked():
    index = pd.date_range("2022-06-01 08:00", periods=10, freq="15min")
    v = [1, 2, np.nan, np.nan, 5, 6, 7, 8, 9, 10.0]
    df = pd.DataFrame({"generation(kWh)": v, "power(W)": v}, index=index)
    out, _ = p.reindex_and_fill(df, "15min", 4)
    assert out["is_filled"].tolist() == [False, False, True, True] + [False] * 6


# ---------- 4. cleaning and scaling never use later or other-split data

def test_gap_fill_is_causal():
    """Changing values after a gap must not change how the gap is filled."""
    index = pd.date_range("2022-06-01 08:00", periods=8, freq="15min")
    v = [1, 2, np.nan, np.nan, 5, 6, 7, 8.0]
    df = pd.DataFrame({"generation(kWh)": v, "power(W)": v}, index=index)
    later = df.copy()
    later.iloc[4:] = 999.0
    a, _ = p.reindex_and_fill(df, "15min", 4)
    b, _ = p.reindex_and_fill(later, "15min", 4)
    assert a["power(W)"].iloc[:4].tolist() == b["power(W)"].iloc[:4].tolist()


def test_normalisation_uses_metadata_only():
    """Train rows are scaled the same whatever the test rows contain."""
    index = pd.date_range("2022-09-30 10:00", periods=48, freq="1h")
    df = pd.DataFrame({"power(W)": np.linspace(0, 20000, 48)}, index=index)
    other = df.copy()
    other.loc[other.index > s.TRAIN_END, "power(W)"] = 1e9
    train = s.time_split(index) == "train"
    a = p.normalise_power(df, 25000)["power_norm"][train]
    b = p.normalise_power(other, 25000)["power_norm"][train]
    assert a.equals(b)


# ------------------------------------------------- 5. splits and stations

def test_time_split_boundaries():
    index = pd.DatetimeIndex(["2022-09-30 23:45", "2022-10-01 00:00",
                              "2022-12-31 23:45", "2023-01-01 00:00"])
    assert s.time_split(index).tolist() == ["train", "val", "val", "test"]


def test_unseen_and_seen_stations_do_not_overlap():
    for horizon in ["15min", "1h", "dayahead"]:
        seen = set(s.stations_for(horizon, "seen"))
        unseen = set(s.stations_for(horizon, "unseen"))
        excluded = set(s.stations_for(horizon, "excluded"))
        assert not seen & unseen
        assert not seen & excluded
        assert not unseen & excluded


def test_folds_contain_only_seen_stations():
    for horizon in ["15min", "1h"]:
        folds = s.station_folds(horizon)
        assert set(folds.index) == set(s.stations_for(horizon, "seen"))


# ------------------------------- 6. baselines use only the past as well

def baseline_preds(builder, base, models):
    table = builder(base)
    table["station"] = "made-up"
    return {name: fn(table) for name, fn in models.items()}


@pytest.mark.parametrize("builder, freq, horizon", [
    (f.features_15min, "15min", "15min"),
    (f.features_1h, "1h", "1h"),
])
def test_intraday_baselines_use_only_the_past(builder, freq, horizon):
    from src.models import baselines
    base = make_base(freq)
    T = pd.Timestamp("2022-06-10 12:00")
    a = baseline_preds(builder, base, baselines.BASELINES[horizon])
    b = baseline_preds(builder, junk_from(base, T), baselines.BASELINES[horizon])
    for name in a:
        assert a[name][T] == b[name][T], name


def test_dayahead_baselines_use_only_data_before_23h():
    from src.models import baselines
    base = make_base("1h")
    issue = pd.Timestamp("2022-06-09 23:00")
    a = baseline_preds(f.features_dayahead, base, baselines.BASELINES["dayahead"])
    b = baseline_preds(f.features_dayahead, junk_from(base, issue), baselines.BASELINES["dayahead"])
    for hour in range(24):
        T = pd.Timestamp("2022-06-10") + pd.Timedelta(hours=hour)
        for name in a:
            assert np.isclose(a[name][T], b[name][T], equal_nan=True), (name, hour)


# ------------- 7. ridge: median fill and scaler are fitted on train only

def test_ridge_fill_and_scaler_use_train_rows_only():
    from src.models import ridge
    table = f.features_1h(make_base("1h"))
    table["usable"] = f.usable_target(table)
    table["station"] = "made-up"
    table["role"] = "seen"
    table["split"] = "train"
    table.loc[table.index >= "2022-06-12", "split"] = "val"
    table.loc[table.index >= "2022-06-16", "split"] = "test"

    other = table.copy()
    not_train = other["split"] != "train"
    features = ridge.feature_columns(other)
    other.loc[not_train, features] = 9999.0      # junk outside train
    other.loc[not_train, "target"] = 9999.0

    a = ridge.fit(table, alpha=1.0)
    b = ridge.fit(other, alpha=1.0)
    assert np.allclose(a[0].statistics_, b[0].statistics_)    # medians
    assert np.allclose(a[1].mean_, b[1].mean_)                # scaler mean
    assert np.allclose(a[1].scale_, b[1].scale_)              # scaler std
    assert np.allclose(a[2].coef_, b[2].coef_)                # ridge weights
