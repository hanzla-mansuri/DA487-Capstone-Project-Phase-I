"""Loaders for the HKUST rooftop PV dataset (Dryad, doi:10.5061/dryad.m37pvmd99).

Design rules (PLAN.md Stage 1):
- Files are read in chunks (generators) so large files never need to be fully in memory;
  the ``read_*`` helpers concatenate the chunks for files that are small enough.
- Timestamps are parsed with an explicit format. Nothing is inferred, so a malformed
  value raises instead of being silently mis-parsed.
- Timestamps are returned exactly as stored (naive, local campus time assumed).
  Timezone and interval-start/end convention are confirmed later in Stage 1, not here.
- Zone.Identifier files (Windows download markers) are ignored.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[1] / "data" / "hkust" / "Dataset"
TS_ROOT = DATA_ROOT / "Time series dataset"
PV_ROOT = TS_ROOT / "PV generation dataset"
MET_ROOT = TS_ROOT / "Meteorological dataset"
METADATA_TTL = DATA_ROOT / "Metadata" / "PV generation system metadata.ttl"

OPT_DIR = PV_ROOT / "PV stations with panel level optimizer"
NOOPT_DIR = PV_ROOT / "PV stations without panel level optimizer"

# Explicit timestamp formats (verified by sampling the files).
# Power/inverter files use one of two styles (uniform within a file, verified on all 94 files):
#   ISO-like  2023-07-01 00:15:00   (84 files)
#   slash     2021/7/1 0:00         (20 files, not zero-padded)
# Weather files all use the slash style.
ISO_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
SLASH_TIME_FORMAT = "%Y/%m/%d %H:%M"
PV_TIME_FORMATS = (ISO_TIME_FORMAT, SLASH_TIME_FORMAT)
MET_TIME_FORMAT = SLASH_TIME_FORMAT

# Weather variable -> folder name. Columns keep their original names.
WEATHER_VARIABLES = {
    "irradiance": "Irradiance",
    "temperature": "Temperature",
    "humidity": "Relative Humidity",
    "pressure": "Sea Level Pressure",
    "visibility": "Visibility",
    "wind": "Wind",
    "rainfall": "Rainfall",
}

DEFAULT_CHUNKSIZE = 200_000


def _csv_files(folder: Path) -> list[Path]:
    """CSV files in a folder, sorted, excluding ``*.csv:Zone.Identifier`` markers."""
    return sorted(p for p in folder.glob("*.csv") if p.suffix == ".csv")


def _detect_format(path: Path, formats: tuple[str, ...]) -> str:
    """Pick the one explicit format (from a whitelist) that parses the file's first timestamp.

    Parsing of the whole file is still strict, so a file that mixes styles raises.
    """
    first = pd.read_csv(path, nrows=1, usecols=["Time"], dtype=str)["Time"].iloc[0]
    for fmt in formats:
        try:
            pd.to_datetime(first, format=fmt)
            return fmt
        except ValueError:
            continue
    raise ValueError(f"{path.name}: timestamp {first!r} matches none of {formats}")


def _iter_csv(path: Path, time_format: str | tuple[str, ...], chunksize: int) -> Iterator[pd.DataFrame]:
    """Yield chunks with ``Time`` parsed (explicit format) and set as the index.

    ``time_format`` may be a tuple of allowed formats; the file's format is detected once.
    """
    if not isinstance(time_format, str):
        time_format = _detect_format(path, time_format)
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk["Time"] = pd.to_datetime(chunk["Time"], format=time_format)
        yield chunk.set_index("Time")


def _read_csv(path: Path, time_format: str | tuple[str, ...], chunksize: int) -> pd.DataFrame:
    return pd.concat(_iter_csv(path, time_format, chunksize))


# --------------------------------------------------------------------------- stations

def list_station_files() -> pd.DataFrame:
    """One row per site-level power file: ``station``, ``has_optimizer``, ``path``.

    This is a *file* inventory. One file may cover several physical stations
    (e.g. ``SQ Apartment1-12``); the station-set gate in Stage 1 settles the counts.
    """
    rows = []
    for has_opt, root in ((True, OPT_DIR), (False, NOOPT_DIR)):
        for p in _csv_files(root / "Site level dataset"):
            rows.append({"station": p.stem, "has_optimizer": has_opt, "path": p})
    return pd.DataFrame(rows)


def iter_station_power(path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> Iterator[pd.DataFrame]:
    """Chunks of a site-level file: columns ``generation(kWh)``, ``power(W)``."""
    return _iter_csv(Path(path), PV_TIME_FORMATS, chunksize)


def read_station_power(path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    """Whole site-level file (small: at most a few MB per station)."""
    return _read_csv(Path(path), PV_TIME_FORMATS, chunksize)


def list_inverter_files() -> pd.DataFrame:
    """One row per inverter file (optimizer stations only): ``station``, ``inverter``, ``path``.

    ``station`` is the file name with the ``_Inverter`` suffix (and any trailing
    inverter number) removed, e.g. ``Library_Inverter_2`` -> station ``Library``.
    """
    rows = []
    for p in _csv_files(OPT_DIR / "Inverter level dataset"):
        base, _, rest = p.stem.partition("_Inverter")
        rows.append({"station": base, "inverter": rest.lstrip("_") or "1", "path": p})
    return pd.DataFrame(rows)


def iter_inverter(path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> Iterator[pd.DataFrame]:
    """Chunks of an inverter file (5-min; dc voltage, total and per-phase AC quantities)."""
    return _iter_csv(Path(path), PV_TIME_FORMATS, chunksize)


def read_inverter(path: Path, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    return _read_csv(Path(path), PV_TIME_FORMATS, chunksize)


# --------------------------------------------------------------------------- weather

def weather_files(variable: str) -> list[Path]:
    """Yearly CSV files for one weather variable (key of ``WEATHER_VARIABLES``).

    Note: ``Rainfall_2021.xlsx`` is not a CSV and is not returned; rainfall is
    available for 2022-2023 only (see DATA_SOURCES.md).
    """
    if variable not in WEATHER_VARIABLES:
        raise KeyError(f"unknown variable {variable!r}; choose from {sorted(WEATHER_VARIABLES)}")
    return _csv_files(MET_ROOT / WEATHER_VARIABLES[variable])


def iter_weather(variable: str, chunksize: int = DEFAULT_CHUNKSIZE) -> Iterator[pd.DataFrame]:
    """Chunks of one weather variable across all years (1-min resolution)."""
    for path in weather_files(variable):
        yield from _iter_csv(path, MET_TIME_FORMAT, chunksize)


def read_weather(variable: str, chunksize: int = DEFAULT_CHUNKSIZE) -> pd.DataFrame:
    """One weather variable, all years (~1.5 M rows for 3 years; fine in memory)."""
    return pd.concat(iter_weather(variable, chunksize))
