"""Persistence baselines (Stage 3).

Every function takes a Stage 2 feature table (`data/processed/features_<h>.parquet`:
one row per station and target time T, regular grid per station) and returns
a prediction of `target` (power / rated) for every row.

Shared rules (PLAN §3):
- night: the prediction is 0 when the sun is below 5 degrees at T;
- predictions are capped to [0, 1.2] (the physical range used in cleaning);
- a prediction is missing (NaN) when its input is missing.
"""
import numpy as np
import pandas as pd

MIN_ELEVATION = 5      # degrees, the daytime definition (PLAN §3)
MAX_NORM = 1.2         # power / rated, the cleaning limit
MIN_CLEARSKY = 50      # W/m2: below this the clear-sky ratio is unstable


def finish(pred, table):
    """Apply the shared rules: cap to [0, 1.2] and set night to 0."""
    pred = pd.Series(pred, index=table.index).clip(lower=0, upper=MAX_NORM)
    pred[table["sun_elevation"] < MIN_ELEVATION] = 0.0
    return pred


def shift_per_station(table, column, steps):
    """`column` moved `steps` rows forward within each station.

    Each station is on a regular grid (Stage 2), so `steps` rows back is
    exactly `steps` time steps back.
    """
    return table.groupby("station")[column].shift(steps)


# ------------------------------------------------------ 15 min and 1 h

def last_value(table):
    """Last-value persistence: the newest known value (lag 1).
    This is the skill-score benchmark for 15 min and 1 h (PLAN §3)."""
    return finish(table["power_lag1"], table)


def smart_persistence(table):
    """Clear-sky-index persistence for 15 min and 1 h.

    The ratio power / clear-sky at the last known step is assumed to stay
    the same, so: prediction = power(T-1) x clearsky(T) / clearsky(T-1).
    This follows the daily rise and fall of the sun, which plain
    persistence ignores. When clearsky(T-1) is below 50 W/m2 (dawn), the
    ratio is unstable, so the last value is used instead.
    """
    clearsky_before = shift_per_station(table, "clearsky_ghi", 1)
    ratio = table["clearsky_ghi"] / clearsky_before
    pred = table["power_lag1"] * ratio

    low_sun = clearsky_before < MIN_CLEARSKY
    pred[low_sun] = table["power_lag1"][low_sun]
    return finish(pred, table)


# ----------------------------------------------------------- day-ahead

def same_hour_yesterday(table):
    """Same hour on D-1. The skill-score benchmark for day-ahead (PLAN §3).
    Hour 23 of D-1 is not known at the 23:00 issue time (it is night, so the
    night rule gives 0)."""
    return finish(table["power_d1"], table)


def same_hour_mean7(table):
    """Mean of the same hour over D-1 .. D-7 (missing days are skipped).

    Uses the cleaned power of earlier days (24 x k hourly rows back).
    The D-1 value of hour 23 is left out: it ends after the issue time.
    """
    days = []
    for k in range(1, 8):
        value = shift_per_station(table, "target", 24 * k)
        if k == 1:
            value = value.where(table.index.hour != 23)
        days.append(value)
    pred = pd.concat(days, axis=1).mean(axis=1)
    return finish(pred, table)


def smart_persistence_dayahead(table):
    """Clear-sky-index persistence for day-ahead.

    How cloudy D-1 was = (D-1 daytime mean power) / (D-1 daytime mean
    clear-sky irradiance). Day D is assumed equally cloudy, so:
    prediction = that ratio x clearsky(T).
    """
    day = table["sun_elevation"] >= MIN_ELEVATION
    daytime = table[day]
    dates = daytime.index.normalize()
    clearsky_daymean = daytime.groupby(["station", dates])["clearsky_ghi"].mean()

    # D-1 value for each row: look up (station, date - 1 day)
    keys = pd.MultiIndex.from_arrays(
        [table["station"], table.index.normalize() - pd.Timedelta("1D")])
    clearsky_d1 = clearsky_daymean.reindex(keys).to_numpy()

    ratio = table["power_d1_daymean"].to_numpy() / clearsky_d1
    pred = ratio * table["clearsky_ghi"].to_numpy()
    return finish(pred, table)


BASELINES = {
    "15min": {"last_value": last_value, "smart_persistence": smart_persistence},
    "1h": {"last_value": last_value, "smart_persistence": smart_persistence},
    "dayahead": {"same_hour_yesterday": same_hour_yesterday,
                 "same_hour_mean7": same_hour_mean7,
                 "smart_persistence": smart_persistence_dayahead},
}
