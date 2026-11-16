"""Cleaning steps for the site-level power files (Stage 2).

Each step is a small function so it can be explained and tested on its own.
"""
import pandas as pd
import pvlib


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


def reindex_and_fill(df, freq, max_steps):
    """Put a station on a regular time grid and fill only short gaps.

    Rules (Stage 2; PLAN Change log 2026-11-16):
    - grid: every `freq` step from the first to the last timestamp;
    - a gap (run of missing power) of at most `max_steps` steps is filled
      by carrying the last known value forward (causal: never uses a
      future value);
    - longer gaps stay missing (Stage 1: they are mostly whole lost days);
    - `is_filled` marks filled rows. Filled values may be used as features,
      but filled targets are never used for evaluation.

    Returns the gridded table and a small dict of counts.
    """
    grid = pd.date_range(df.index.min(), df.index.max(), freq=freq)
    off_grid = int((~df.index.isin(grid)).sum())
    out = df.reindex(grid)

    # length of each run of missing values
    missing = out["power(W)"].isna()
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")
    short_gap = missing & (run_len <= max_steps)

    # forward fill, then keep the filled values only inside short gaps
    filled = out.ffill()
    out.loc[short_gap, :] = filled.loc[short_gap, :]

    # a gap at the very start has nothing before it, so it stays missing
    out["is_filled"] = short_gap & out["power(W)"].notna()

    counts = {
        "rows_in": len(df),
        "off_grid_rows_dropped": off_grid,
        "grid_rows": len(out),
        "missing_before_fill": int(missing.sum()),
        "filled": int(out["is_filled"].sum()),
        "missing_after_fill": int(out["power(W)"].isna().sum()),
    }
    return out, counts


def sun_elevation(index, freq, lat, lon):
    """Sun elevation (degrees) at the MIDDLE of each interval.

    Rows are labelled with the interval start, so the middle is t + freq/2.
    Timestamps are local Hong Kong time (UTC+8, no daylight saving; Stage 1).
    """
    middle = index + pd.Timedelta(freq) / 2
    middle = middle.tz_localize("Etc/GMT-8")  # "GMT-8" means UTC+8 in this naming
    pos = pvlib.solarposition.get_solarposition(middle, lat, lon)
    return pd.Series(pos["elevation"].to_numpy(), index=index)


def set_night_to_zero(df, elevation, min_elevation=5):
    """Set power and energy to 0 when the sun is below `min_elevation` degrees.

    Rule (Stage 1 to-do item 16; PLAN Change log 2026-11-16): night values are
    already zero almost everywhere (Stage 1), so this mainly removes the rare
    non-zero night readings and fills missing night rows with a known 0.

    Returns the cleaned table and a small dict of counts.
    """
    out = df.copy()
    night = elevation < min_elevation
    power = out["power(W)"]

    counts = {
        "night_rows": int(night.sum()),
        "night_nonzero_set_to_0": int((night & (power > 0)).sum()),
        "night_missing_set_to_0": int((night & power.isna()).sum()),
        "energy_removed_share_%": round(100 * power[night].sum() / power.sum(), 3),
    }
    out.loc[night, ["generation(kWh)", "power(W)"]] = 0.0
    return out, counts
