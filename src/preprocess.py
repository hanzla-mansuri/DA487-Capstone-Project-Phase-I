"""Cleaning steps for the site-level power files (Stage 2).

Each step is a small function so it can be explained and tested on its own.
"""
import pandas as pd


def resolve_duplicates(df):
    """Sort by time and make every timestamp appear once.

    Rule (decided from the Stage 2 duplicate check):
    - if all copies of a timestamp have the same power, keep one copy;
    - if the copies disagree, we cannot know which is right, so the
      timestamp is kept but its values are set to missing (NaN).

    Returns the cleaned table and a small dict of counts.
    """
    df = df.sort_index(kind="stable")
    is_dup = df.index.duplicated(keep=False)

    # how many different power values does each duplicated timestamp have?
    n_values = df[is_dup].groupby(level=0)["power(W)"].nunique(dropna=False)
    disagree = n_values[n_values > 1].index

    out = df[~df.index.duplicated(keep="first")].copy()
    out.loc[disagree, :] = float("nan")

    counts = {
        "rows_in": len(df),
        "duplicate_timestamps": int(n_values.shape[0]),
        "disagreeing_timestamps": int(len(disagree)),
        "rows_out": len(out),
    }
    return out, counts


def to_interval_start(df, has_optimizer):
    """Label every row with the START of its time interval.

    From Stage 1 (timestamp convention):
    - 15-min optimizer stations are already labelled at the interval start;
    - hourly stations are labelled at the interval end, so we subtract 1 hour.

    Note: a row labelled t is only known at the END of its interval
    (t + 15 min or t + 1 h). Features and forecast issue times must respect this.
    """
    out = df.copy()
    if not has_optimizer:
        out.index = out.index - pd.Timedelta("1h")
    return out


def mask_power_outliers(df, rated_W):
    """Set power values outside [0, 1.2 x rated] to missing (NaN).

    Rule (Stage 1, outlier section): the glitches are far outside the
    physical range (down to -8.7 MW, up to 31.8 MW), so clipping them would
    invent a value. We set them to missing instead; the gap is handled later.
    Both columns of the row are set to missing, because they describe the
    same reading.

    Returns the cleaned table and a small dict of counts.
    """
    out = df.copy()
    power = out["power(W)"]
    too_low = power < 0
    too_high = power > 1.2 * rated_W

    out.loc[too_low | too_high, :] = float("nan")

    counts = {
        "negative": int(too_low.sum()),
        "above_1.2_rated": int(too_high.sum()),
    }
    return out, counts


def mask_irradiance_outliers(irr):
    """Clean the 1-min irradiance series (W/m2).

    Rule (Stage 1, outlier section; PLAN Change log 2026-11-15):
    - above 2000 W/m2 is a sensor error, set to missing;
    - 1500-2000 W/m2 is kept (mostly cloud-enhancement peaks near noon)
      but flagged, so we can check later whether these rows matter.

    `irr` is a Series. Returns the cleaned Series and a boolean flag Series.
    """
    out = irr.copy()
    out[out > 2000] = float("nan")
    high_flag = (out > 1500) & (out <= 2000)
    return out, high_flag


def flag_zero_fault(power, irr):
    """Flag rows where the station produces nothing although the sun is strong.

    Rule (Stage 2 zero-output count, results/tables/02_zero_output.csv):
    power is exactly 0 W while irradiance over the same interval is at least
    400 W/m2. These rows are outages or faulty readings: they are left out of
    training and evaluation targets, but kept (flagged) for Stage 8.

    `power` and `irr` are Series on the same time index (interval-start label).
    Returns a boolean Series.
    """
    return (power == 0) & (irr >= 400)
