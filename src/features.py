"""Feature building (Stage 2).

Every function takes a time index labelled at the interval START (see
`src/preprocess.py::to_interval_start`).

Leakage note: calendar and solar-geometry features depend only on the clock
and the station location. They are known in advance for any future time, so
they may be computed for the TARGET time. Features built from measured data
(power, weather, clear-sky index of measured irradiance) may only use values
that are known at the forecast issue time; those are built separately.
"""
import numpy as np
import pandas as pd
import pvlib

from src.preprocess import flag_zero_fault


def calendar_features(index):
    """Hour of day and day of year, as numbers and as sine/cosine pairs.

    The sine/cosine pairs tell the model that 23:00 is close to 00:00 and
    31 December is close to 1 January (useful for linear models and the LSTM;
    tree models can use the plain numbers).
    """
    hour = index.hour + index.minute / 60
    doy = index.dayofyear

    out = pd.DataFrame(index=index)
    out["hour"] = hour
    out["day_of_year"] = doy
    out["month"] = index.month
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return out


def solar_features(index, freq, lat, lon):
    """Sun position and clear-sky irradiance at the MIDDLE of each interval.

    - `sun_elevation`, `sun_azimuth` (degrees), from pvlib;
    - `clearsky_ghi` (W/m2): irradiance on a horizontal surface under a
      cloudless sky (pvlib Haurwitz model, which needs only the sun position).

    Timestamps are local Hong Kong time (UTC+8, no daylight saving; Stage 1).
    """
    middle = index + pd.Timedelta(freq) / 2
    middle = middle.tz_localize("Etc/GMT-8")  # "GMT-8" means UTC+8 in this naming
    pos = pvlib.solarposition.get_solarposition(middle, lat, lon)
    clearsky = pvlib.clearsky.haurwitz(pos["apparent_zenith"])

    out = pd.DataFrame(index=index)
    out["sun_elevation"] = pos["elevation"].to_numpy()
    out["sun_azimuth"] = pos["azimuth"].to_numpy()
    out["clearsky_ghi"] = clearsky["ghi"].to_numpy()
    return out


def clear_sky_index(irradiance, clearsky_ghi, sun_elevation,
                    min_elevation=5, max_index=1.5):
    """Measured irradiance divided by clear-sky irradiance.

    About 1 on a clear day, lower when it is cloudy. When the sun is low the
    ratio divides by a tiny number and becomes meaningless, so it is set to 0
    below `min_elevation` (the same 5 degrees as the night rule), and it is
    clipped at `max_index` (short cloud-enhancement peaks can exceed 1).

    This uses MEASURED irradiance, so it must only be used for times that are
    already known at the forecast issue time (as a lagged feature).
    """
    index = irradiance / clearsky_ghi
    index = index.clip(lower=0, upper=max_index)
    index[sun_elevation < min_elevation] = 0.0
    return index


def parse_angle(text):
    """Turn a .ttl angle text into (number, is_multi).

    - "10deg"     -> (10.0, False)  one value;
    - "0/15deg"   -> (7.5, True)    two faces: the mean, flagged as multi;
    - "Mixed"     -> (NaN, True)    no number given, flagged as multi.
    """
    text = str(text).replace("deg", "").strip()
    if text == "Mixed" or text == "nan":
        return float("nan"), True
    parts = [float(x) for x in text.split("/")]
    return sum(parts) / len(parts), len(parts) > 1


def station_features(inventory):
    """Static features per station from the .ttl metadata (one row per station).

    - `is_optimizer`: 1 for the 37 SolarEdge optimizer stations, else 0;
    - `tilt_deg`: tilt in degrees (mean of the two faces for a pair; NaN for
      "Mixed"); all known tilts are 0-15 degrees;
    - no azimuth feature: every single-orientation station is flat (tilt 0,
      where azimuth has no meaning), and every tilted station has a pair
      (e.g. "90/-90") or "Mixed", where one number has no meaning either.
      The 0-degree reference of the .ttl azimuth is also not stated;
    - `multi_orientation`: 1 if tilt or azimuth has several values or "Mixed".

    NaN is kept on purpose: tree models handle it; for linear models and the
    LSTM it is filled later with a value computed on train stations only.
    These features are fixed per station, so they cannot leak future data.
    """
    rows = []
    for _, r in inventory.iterrows():
        tilt, tilt_multi = parse_angle(r["tilt_raw"])
        _, az_multi = parse_angle(r["azimuth_raw"])
        rows.append({
            "station": r["station"],
            "is_optimizer": int(r["has_optimizer"]),
            "tilt_deg": tilt,
            "multi_orientation": int(tilt_multi or az_multi),
        })
    return pd.DataFrame(rows).set_index("station")


# ---------------------------------------------------------------------------
# Lag and rolling features (PLAN Change log 2026-11-16, "lags and windows").
#
# Timing rule: every table here is on a regular grid with one row per step,
# labelled at the interval START. Row t is only known at t + step. The output
# row is labelled with the TARGET time T (forecast issued at T), so the newest
# known input row is T - step: that is lag 1. `shift(k)` moves row T - k*step
# onto row T. Rolling windows are computed on `shift(1)`, so they end at lag 1.
# ---------------------------------------------------------------------------


def add_lags(out, series, lags, name):
    """Add one column per lag: `name_lag{k}` = value k steps before T."""
    for k in lags:
        out[f"{name}_lag{k}"] = series.shift(k)


def add_rolling(out, series, window, name, stats=("mean",)):
    """Add trailing statistics over the last `window` known rows (lag 1..window).

    The window must be complete; if any value in it is missing, the result
    is missing.
    """
    past = series.shift(1).rolling(window, min_periods=window)
    if "mean" in stats:
        out[f"{name}_mean{window}"] = past.mean()
    if "std" in stats:
        out[f"{name}_std{window}"] = past.std()


def build_base(power, weather, freq, lat, lon):
    """One station's data on one grid: power, weather and solar features.

    `power` has `power_norm` and `is_filled`; `weather` comes from
    `preprocess.build_weather(freq)`. Rows are kept for the station's own
    time range only. Adds the clear-sky index of the measured irradiance.
    """
    base = power[["power_norm", "is_filled"]].join(weather, how="left")
    solar = solar_features(base.index, freq, lat, lon)
    base = base.join(solar)
    base["csi"] = clear_sky_index(base["irradiance"], base["clearsky_ghi"],
                                  base["sun_elevation"])
    return base


def target_time_features(base):
    """Features known in advance for the target time T (calendar and sun)."""
    out = calendar_features(base.index)
    out["sun_elevation"] = base["sun_elevation"]
    out["sun_azimuth"] = base["sun_azimuth"]
    out["clearsky_ghi"] = base["clearsky_ghi"]
    return out


def add_target(out, base):
    """The value to predict at T, with two quality flags.

    - `target_is_filled`: the value was gap-filled. Filled targets must never
      be used to evaluate a model (PLAN §3, "Evaluation targets").
    - `target_is_fault`: zero output although irradiance is at least
      400 W/m2 (`preprocess.flag_zero_fault`). Left out of training and
      evaluation, kept for Stage 8 (PLAN Change log 2026-11-16).

    The fault flag looks at irradiance at T itself. That is fine because it
    only labels the target row; it is never used as an input feature.
    """
    out["target"] = base["power_norm"]
    out["target_is_filled"] = base["is_filled"]
    out["target_is_fault"] = flag_zero_fault(base["power_norm"], base["irradiance"])


def usable_target(out):
    """True for rows whose target may be used for training and evaluation:
    observed (not missing, not filled) and not a zero-output fault."""
    return out["target"].notna() & ~out["target_is_filled"] & ~out["target_is_fault"]


def features_15min(base):
    """Features for the 15-min horizon (15-min rows; target = row T)."""
    out = target_time_features(base)

    add_lags(out, base["power_norm"], [1, 2, 3, 4, 8, 96], "power")
    add_rolling(out, base["power_norm"], 4, "power", stats=("mean", "std"))

    add_lags(out, base["irradiance"], [1, 2, 4], "irr")
    add_rolling(out, base["irradiance"], 4, "irr")

    add_lags(out, base["csi"], [1, 2], "csi")
    add_rolling(out, base["csi"], 4, "csi")

    for col in ["temperature", "wind_speed", "pressure", "visibility"]:
        add_lags(out, base[col], [1], col)

    add_target(out, base)
    return out


def features_1h(base):
    """Features for the 1-h horizon (hourly rows; target = hour T)."""
    out = target_time_features(base)

    add_lags(out, base["power_norm"], [1, 2, 3, 24], "power")
    add_rolling(out, base["power_norm"], 3, "power", stats=("mean", "std"))

    add_lags(out, base["irradiance"], [1, 2], "irr")
    add_rolling(out, base["irradiance"], 3, "irr")

    add_lags(out, base["csi"], [1, 2], "csi")
    add_rolling(out, base["csi"], 3, "csi")

    for col in ["temperature", "wind_speed", "pressure", "visibility"]:
        add_lags(out, base[col], [1], col)

    add_target(out, base)
    return out


def previous_day_value(base, daily):
    """Map a daily Series (indexed by date D-1) onto the hourly rows of day D."""
    shifted = daily.copy()
    shifted.index = shifted.index + pd.Timedelta("1D")
    target_dates = base.index.normalize()
    return shifted.reindex(target_dates).to_numpy()


def features_dayahead(base):
    """Features for day-ahead (hourly rows of day D; issued 23:00 on D-1).

    The newest known row at issue time is 22:00 on D-1 (it covers
    22:00-23:00). The 23:00 row of D-1 is only known at midnight.
    """
    out = target_time_features(base)
    power = base["power_norm"]

    # same hour on D-1, D-2, D-7 (24, 48, 168 hourly rows back)
    out["power_d1"] = power.shift(24)
    out["power_d2"] = power.shift(48)
    out["power_d7"] = power.shift(168)
    # hour 23 of D-1 ends at midnight, after the issue time
    out.loc[out.index.hour == 23, "power_d1"] = float("nan")

    # D-1 summaries (daytime rows end by 19:00, so all are known at 23:00)
    day = base["sun_elevation"] >= 5
    out["power_d1_daymean"] = previous_day_value(
        base, power[day].groupby(power[day].index.normalize()).mean())
    out["csi_d1_daymean"] = previous_day_value(
        base, base["csi"][day].groupby(base.index[day].normalize()).mean())

    # D-1 mean temperature: rows 00:00-22:00 only (the 23:00 row is not known)
    known = base.index.hour <= 22
    temp = base["temperature"][known]
    out["temp_d1_mean"] = previous_day_value(
        base, temp.groupby(temp.index.normalize()).mean())

    # weather from the 22:00 row of D-1
    at22 = base[base.index.hour == 22]
    at22 = at22.set_index(at22.index.normalize())
    for col in ["temperature", "pressure", "visibility"]:
        out[f"{col}_d1_22h"] = previous_day_value(base, at22[col])
    pressure_change = at22["pressure"] - at22["pressure"].shift(1, freq="1D")
    out["pressure_change_24h"] = previous_day_value(base, pressure_change)

    add_target(out, base)
    return out


def add_station_features(out, station_row):
    """Copy the static station features (one row of `station_features`)."""
    for col, value in station_row.items():
        out[col] = value
    return out
