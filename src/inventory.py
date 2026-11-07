"""Station inventory: metadata from the .ttl file joined with facts read from each power CSV.

Kept deliberately simple (regex on the .ttl, no RDF library, plain pandas).
Raw values are kept as stored (e.g. azimuth and tilt stay as text such as "90/-90deg");
parsing them into numbers is a later, documented step.
"""
import re

import pandas as pd

from src import data_io

# CSV file names that do not match their .ttl name after normalising (checked by hand).
NAME_FIXES = {"indoorsportscentre": "indoorsportcenter"}


def normalise(name):
    """Lower-case and keep letters/digits only, so 'Zone L1(Station 2)' == 'Zone_L1__Station_2_'."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return NAME_FIXES.get(key, key)


def _find(pattern, block):
    match = re.search(pattern, block, re.S)
    return match.group(1) if match else None


def read_ttl():
    """One row per PV generation system in the .ttl file."""
    text = data_io.METADATA_TTL.read_text(encoding="utf-8")
    rows = []
    for block in re.split(r"\n\s*\n", text):
        match = re.match(r"pvsystem:(\S+) a brick:PV_Generation_System", block.strip())
        if not match:
            continue
        rows.append({
            "ttl_name": match.group(1),
            "key": normalise(match.group(1)),
            "latitude": _find(r"brick:latitude ([\d.\-]+)", block),
            "longitude": _find(r"brick:longitude ([\d.\-]+)", block),
            "altitude_m": _find(r"ext:altitude \[.*?brick:value ([\d.\-]+)", block),
            "connection_date": _find(r'ext:connectionDate \[ brick:value "([^"]+)"', block),
            "rated_unit": _find(r"ext:ratedPowerOutput \[ brick:hasUnit unit:(\w+)", block),
            "rated_value": _find(r"ext:ratedPowerOutput \[.*?brick:value ([\d.]+)", block),
            "azimuth_raw": _find(r'ext:azimuth \[ brick:value "([^"]+)"', block),
            "tilt_raw": _find(r'ext:tiltAngle \[ brick:value "([^"]+)"', block),
            "contractor": _find(r'ext:contractor \[ brick:value "([^"]+)"', block),
        })
    ttl = pd.DataFrame(rows)
    for col in ["latitude", "longitude", "altitude_m", "rated_value"]:
        ttl[col] = pd.to_numeric(ttl[col])
    return ttl


def file_facts(path):
    """Facts read from one power CSV: resolution, first/last timestamp, row counts."""
    power = data_io.read_station_power(path)
    times = power.index.drop_duplicates().sort_values()
    step = times.to_series().diff().median()
    return {
        "resolution_min": step.total_seconds() / 60,
        "start": times[0],
        "end": times[-1],
        "n_rows": len(power),
        "n_duplicate_timestamps": int(power.index.duplicated().sum()),
    }


def build_inventory():
    """Join file facts with .ttl metadata. One row per site-level power file."""
    files = data_io.list_station_files()
    files["key"] = files["station"].map(normalise)

    facts = [file_facts(path) for path in files["path"]]
    files = pd.concat([files, pd.DataFrame(facts)], axis=1)

    # number of inverter files per station (optimizer stations only)
    inverters = data_io.list_inverter_files()
    inverters["key"] = inverters["station"].map(normalise)
    counts = inverters.groupby("key").size().rename("n_inverter_files")
    files = files.merge(counts, on="key", how="left")
    files["n_inverter_files"] = files["n_inverter_files"].fillna(0).astype(int)

    return files.merge(read_ttl(), on="key", how="left").drop(columns=["path", "key"])
