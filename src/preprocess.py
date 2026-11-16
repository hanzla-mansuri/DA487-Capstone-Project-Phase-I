"""Cleaning and resampling steps for the power and weather data (Stage 2).

Each step is a small function so it can be explained and tested on its own.
"""
import numpy as np
import pandas as pd
import pvlib

from src.data_io import read_weather


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


def cut_to_study_period(df, end="2023-12-31 23:59:59"):
    """Drop rows after the end of the study period (Dryad data: 2021-2023).

    Stage 1 to-do item 3: `Zone L1 (Station 1)` runs into 2024-01-01.
    Apply this AFTER `to_interval_start`: an hourly row labelled
    2024-01-01 00:00 is the hour 23:00-24:00 of 2023-12-31 and must be kept.

    Returns the cut table and the number of rows dropped.
    """
    out = df[df.index <= end]
    return out, len(df) - len(out)


def resample_weather(series, freq, how="mean", min_share=0.8):
    """Resample a 1-min weather series to `freq` steps (e.g. "15min", "1h").

    Rules (Stage 1 to-do item 10):
    - labelled at the interval START, the same as the power data
      (row t = minutes [t, t + freq));
    - an interval is kept only if at least `min_share` (80 %) of its
      minutes are present, otherwise it is missing;
    - `how` is "mean" for most variables and "sum" for rainfall
      (rainfall is an amount per minute, so the interval total is the sum).
    """
    groups = series.resample(freq, label="left", closed="left")
    if how == "sum":
        out = groups.sum()
    else:
        out = groups.mean()

    minutes_needed = pd.Timedelta(freq) / pd.Timedelta("1min")
    enough = groups.count() >= min_share * minutes_needed
    return out.where(enough)


def resample_wind_direction(degrees, freq, min_share=0.8):
    """Average wind direction (degrees) over `freq` steps.

    Directions are angles, so a plain mean is wrong (the mean of 350 and 10
    would be 180). We average the sine and cosine and turn them back into
    an angle between 0 and 360.
    """
    radians = np.deg2rad(degrees)
    sin_mean = resample_weather(np.sin(radians), freq, "mean", min_share)
    cos_mean = resample_weather(np.cos(radians), freq, "mean", min_share)
    angle = np.rad2deg(np.arctan2(sin_mean, cos_mean))
    return angle % 360


def build_weather(freq):
    """All weather variables at `freq` steps, one column each.

    Irradiance outliers are cleaned first (`mask_irradiance_outliers`), and
    the 1500-2000 W/m2 flag is kept as the share of flagged minutes.
    Rainfall is NOT used: it exists for 2022-2023 only (PLAN Change log
    2026-11-16). `resample_weather(..., how="sum")` would be used for it.
    Humidity is NOT used: the sensor is faulty in 2021-01..03 (stuck at
    10-15 %, 0 % at noon) and on 2023-05-17..24 (jumps between 0 and 100 %).
    """
    irr = read_weather("irradiance")["Irradiance (W/m2)"]
    irr, high_flag = mask_irradiance_outliers(irr)

    wind = read_weather("wind")
    # wind speed above 20 m/s: isolated 1-min spikes (up to 91.9 m/s), almost
    # all in 2023-01..06; the 99.99th percentile is 18.9 m/s. Set to missing,
    # together with the direction of the same minute.
    spike = wind["Wind Speed (m/s)"] > 20
    wind.loc[spike, :] = float("nan")

    columns = {
        "irradiance": resample_weather(irr, freq),
        "irr_high_share": resample_weather(high_flag.astype(float), freq),
        "temperature": resample_weather(read_weather("temperature")["Temp (Degree Celsius)"], freq),
        "pressure": resample_weather(read_weather("pressure")["SLP (hPa)"], freq),
        "visibility": resample_weather(read_weather("visibility")["Vis (km)"], freq),
        "wind_speed": resample_weather(wind["Wind Speed (m/s)"], freq),
        "wind_direction": resample_wind_direction(wind["Wind Direction (degree)"], freq),
    }
    return pd.DataFrame(columns)


def rated_power_W(inventory, station):
    """Rated power of a station in W, from the .ttl metadata.

    `inventory` is results/tables/01_station_inventory.csv. All 60 stations
    have a rated value in kW (checked in Stage 2), so no fallback is needed.
    If a value were missing, the plan's fallback is the 99th percentile of
    power computed on TRAIN data only.
    """
    row = inventory.set_index("station").loc[station]
    if row["rated_unit"] != "KW":
        raise ValueError(f"unexpected unit {row['rated_unit']!r} for {station}")
    return row["rated_value"] * 1000


def normalise_power(df, rated_W):
    """Add `power_norm` = power / rated power (0 = nothing, 1 = nameplate).

    This makes stations of different sizes comparable, so one global model
    can learn from all of them. It uses only metadata, not data from any
    split, so it cannot leak information from validation or test.
    """
    out = df.copy()
    out["power_norm"] = out["power(W)"] / rated_W
    return out


# Stations whose capacity changed during the data period. Only data from the
# .ttl connection date onward matches the rated power, so earlier rows are
# dropped (PLAN Change log 2026-11-16).
# Zone A2: output about 0.25 x rated until 2023-05, then about 0.75 x rated;
# the .ttl connection date is 2023-05-31 (rated 249.5 kW).
USE_FROM = {"Zone A2": "2023-05-31"}


def cut_before_capacity_change(df, station):
    """Drop rows before the date in `USE_FROM` (only for the stations listed).

    Returns the cut table and the number of rows dropped.
    """
    if station not in USE_FROM:
        return df, 0
    out = df[df.index >= USE_FROM[station]]
    return out, len(df) - len(out)
