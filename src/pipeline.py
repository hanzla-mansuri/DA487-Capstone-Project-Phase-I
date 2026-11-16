"""One command from raw data to a model-ready table (Stage 2).

    python -m src.pipeline --horizon 1h

Steps for every station in the horizon set (seen and unseen; excluded
stations are skipped):
    clean (preprocess) -> hourly -> weather + solar -> features + target
    -> station features -> time split and station role
Output:
    data/processed/features_<horizon>.parquet   (git-ignored)
    results/tables/02_split_summary.csv          (one block per horizon)
"""
import argparse
from pathlib import Path

import pandas as pd

from src import data_io, features, preprocess, splits

INVENTORY = "results/tables/01_station_inventory.csv"
OUT_DIR = Path("data/processed")
SUMMARY = Path("results/tables/02_split_summary.csv")

BUILDERS = {
    "1h": features.features_1h,
}


def clean_station(station, path, has_optimizer, inventory):
    """All Stage 2 cleaning rules for one station, in order.

    Returns the cleaned table on its own grid (15 min or 1 h) with
    `power(W)`, `power_norm` and `is_filled`.
    """
    rated = preprocess.rated_power_W(inventory, station)
    row = inventory.set_index("station").loc[station]

    df = data_io.read_station_power(path)
    df, _ = preprocess.resolve_duplicates(df)
    df, _ = preprocess.mask_power_outliers(df, rated)
    df = preprocess.to_interval_start(df, has_optimizer)
    df, _ = preprocess.cut_to_study_period(df)
    df, _ = preprocess.cut_before_capacity_change(df, station)

    if has_optimizer:
        freq, max_steps = "15min", 4
    else:
        freq, max_steps = "1h", 1
    df, _ = preprocess.reindex_and_fill(df, freq, max_steps)

    elevation = preprocess.sun_elevation(df.index, freq, row["latitude"], row["longitude"])
    df, _ = preprocess.set_night_to_zero(df, elevation)
    df = preprocess.normalise_power(df, rated)
    return df


def build_station(station, path, has_optimizer, inventory, weather, horizon):
    """Feature table of one station for one horizon."""
    row = inventory.set_index("station").loc[station]
    df = clean_station(station, path, has_optimizer, inventory)

    # 1-h and day-ahead use hourly data for all stations
    if has_optimizer:
        df = preprocess.to_hourly(df)

    base = features.build_base(df, weather, "1h", row["latitude"], row["longitude"])
    out = BUILDERS[horizon](base)
    out["usable"] = features.usable_target(out)
    out["station"] = station
    return out


def build_horizon(horizon):
    """Build the table for all stations of one horizon and save it."""
    inventory = pd.read_csv(INVENTORY)
    files = data_io.list_station_files().set_index("station")
    sets = splits.station_sets().set_index("station")
    meta = features.station_features(inventory)
    weather = preprocess.build_weather("1h")

    stations = splits.stations_for(horizon, "seen") + splits.stations_for(horizon, "unseen")
    tables = []
    for i, station in enumerate(sorted(stations), start=1):
        print(f"[{i}/{len(stations)}] {station}")
        r = files.loc[station]
        out = build_station(station, r["path"], r["has_optimizer"], inventory, weather, horizon)
        out = features.add_station_features(out, meta.loc[station])
        out["role"] = sets.loc[station, "role"]
        out["split"] = splits.time_split(out.index)
        tables.append(out)

    table = pd.concat(tables)
    table.index.name = "time"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"features_{horizon}.parquet"
    table.to_parquet(path)
    print(f"saved {path}: {len(table)} rows, {table['station'].nunique()} stations")

    write_summary(table, horizon)
    return table


def write_summary(table, horizon):
    """Rows, usable rows, stations and date range per role and split.

    Replaces this horizon's block in the summary file and keeps the others.
    """
    t = table.reset_index()
    groups = t.groupby(["role", "split"])
    summary = pd.DataFrame({
        "rows": groups.size(),
        "usable_rows": groups["usable"].sum(),
        "stations": groups["station"].nunique(),
        "first": groups["time"].min(),
        "last": groups["time"].max(),
    }).reset_index()
    summary.insert(0, "horizon", horizon)

    if SUMMARY.exists():
        old = pd.read_csv(SUMMARY)
        old = old[old["horizon"] != horizon]
        summary = pd.concat([old, summary])
    summary.to_csv(SUMMARY, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the model-ready table for one horizon.")
    parser.add_argument("--horizon", choices=sorted(BUILDERS), required=True)
    args = parser.parse_args()
    build_horizon(args.horizon)
