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
