"""Tests for src/metrics.py, with small hand-made numbers."""
import numpy as np
import pandas as pd

from src import metrics


def make(values_target, values_pred, values_bench, rated=10.0, start="2022-06-01", freq="6h"):
    index = pd.date_range(start, periods=len(values_target), freq=freq)
    target = pd.Series(values_target, index=index, dtype=float)
    pred = pd.Series(values_pred, index=index, dtype=float)
    bench = pd.Series(values_bench, index=index, dtype=float)
    rated_kW = pd.Series(rated, index=index)
    return target, pred, bench, rated_kW


def test_known_numbers():
    # errors (power/rated): +0.1, -0.1, +0.2, 0.0 ; rated 10 kW
    t, p, b, r = make([0.5, 0.5, 0.5, 0.5], [0.6, 0.4, 0.7, 0.5], [0.3, 0.3, 0.3, 0.3])
    s = metrics.score(t, p, b, r, n_boot=0)
    assert np.isclose(s["nmae_pct"], 100 * 0.4 / 4)            # 10 %
    assert np.isclose(s["mae_kW"], 10 * 0.4 / 4)                # 1 kW
    assert np.isclose(s["rmse_norm"], np.sqrt(0.06 / 4))
    assert np.isclose(s["rmse_kW"], 10 * np.sqrt(0.06 / 4))
    assert np.isclose(s["skill"], 1 - np.sqrt(0.06 / 4) / 0.2)  # bench RMSE = 0.2


def test_perfect_model_has_skill_one_and_benchmark_has_skill_zero():
    t, p, b, r = make([0.1, 0.5, 0.9, 0.3], [0.1, 0.5, 0.9, 0.3], [0.2, 0.4, 0.6, 0.1])
    assert metrics.score(t, p, b, r, n_boot=0)["skill"] == 1.0
    assert metrics.score(t, b, b, r, n_boot=0)["skill"] == 0.0


def test_rows_with_a_missing_value_are_dropped_for_both():
    t, p, b, r = make([0.5, 0.5, 0.5], [0.6, np.nan, 0.5], [0.5, 0.0, np.nan])
    s = metrics.score(t, p, b, r, n_boot=0)
    assert s["n_rows"] == 1                                      # only the first row


def test_day_sums_give_the_same_result_as_direct_computation():
    rng = np.random.default_rng(0)
    n = 40
    t, p, b, r = make(rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n))
    s = metrics.score(t, p, b, r, n_boot=0)
    assert np.isclose(s["mae_kW"], ((p - t).abs() * r).mean())
    assert np.isclose(s["rmse_norm"], np.sqrt(((p - t) ** 2).mean()))
    assert s["n_days"] == 10                                     # 40 rows, 4 per day


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(1)
    n = 200
    t, p, b, r = make(rng.uniform(0, 1, n), rng.uniform(0, 1, n), rng.uniform(0, 1, n))
    s1 = metrics.score(t, p, b, r, n_boot=200)
    s2 = metrics.score(t, p, b, r, n_boot=200)
    for m in metrics.METRICS:
        assert s1[f"{m}_low"] <= s1[m] <= s1[f"{m}_high"]
        assert s1[f"{m}_low"] == s2[f"{m}_low"]


def test_per_station_and_mean_skill():
    t, p, b, r = make([0.5] * 8, [0.5] * 4 + [0.6] * 4, [0.6] * 8)
    station = pd.Series(["A"] * 4 + ["B"] * 4, index=t.index)
    table = metrics.per_station(t, p, b, r, station)
    assert table.loc["A", "skill"] == 1.0                        # perfect
    assert table.loc["B", "skill"] == 0.0                        # same as benchmark
    assert metrics.mean_station_skill(t, p, b, r, station) == 0.5
