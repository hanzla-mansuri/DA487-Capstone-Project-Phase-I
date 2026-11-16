"""Station sets and splits (Stage 2).

Station roles (per station, the same for every horizon it belongs to):
- "seen":     used in train, validation and test;
- "unseen":   test only, never used for training (little or no training
              data); evaluated as unseen stations in Stage 9;
- "excluded": not used at all.

Horizon sets:
- 15 min: the 37 optimizer stations (15-min data);
- 1 h and day-ahead: all 60 stations (optimizer stations averaged to hourly).
"""
import pandas as pd

STAGE1_TABLE = "results/tables/01_station_sets.csv"

# Decisions made after the Stage 1 gate (PLAN Change log 2026-11-16).
DROPPED = {
    "UG Hall4 Flexible PV": "zero output in 64 % of high-irradiance hours",
}
TEST_ONLY = {
    "Zone A2": "capacity change; used from 2023-05-31 only, so no training data",
}

# Stage 1 group name -> role
ROLE_OF_GROUP = {
    "seen (train, val, test)": "seen",
    "test only (little or no train data)": "unseen",
    "other": "excluded",
}
REASON_OF_GROUP = {
    "seen (train, val, test)": "enough data in train, validation and test",
    "test only (little or no train data)": "little or no training data (Stage 1 gate)",
    "other": "no training data and under 180 test days (Stage 1 gate)",
}


def station_sets():
    """One row per station: role, reason, and which horizon sets it is in."""
    table = pd.read_csv(STAGE1_TABLE)

    table["role"] = table["group"].map(ROLE_OF_GROUP)
    table["reason"] = table["group"].map(REASON_OF_GROUP)

    for station, reason in TEST_ONLY.items():
        table.loc[table["station"] == station, "role"] = "unseen"
        table.loc[table["station"] == station, "reason"] = reason

    # Zone A2 keeps only 2023-05-31 .. 2023-12-31 (215 days, all in test)
    table.loc[table["station"] == "Zone A2", ["days_train", "days_val"]] = 0
    table.loc[table["station"] == "Zone A2", "days_test"] = 215

    for station, reason in DROPPED.items():
        table.loc[table["station"] == station, "role"] = "excluded"
        table.loc[table["station"] == station, "reason"] = reason

    table["in_15min"] = table["has_optimizer"]
    table["in_1h_dayahead"] = True

    columns = ["station", "has_optimizer", "role", "reason",
               "in_15min", "in_1h_dayahead",
               "days_train", "days_val", "days_test"]
    return table[columns]


def stations_for(horizon, role):
    """Names of the stations with `role` in the set for `horizon`
    ("15min", "1h" or "dayahead")."""
    table = station_sets()
    if horizon == "15min":
        table = table[table["in_15min"]]
    table = table[table["role"] == role]
    return sorted(table["station"])


# Chronological split (PLAN §2 Stage 2). Dates are the TARGET times.
TRAIN_END = "2022-09-30 23:59:59"
VAL_END = "2022-12-31 23:59:59"


def time_split(index):
    """Label each target time as "train", "val" or "test".

    - train: start .. 2022-09-30
    - val:   2022-10-01 .. 2022-12-31
    - test:  2023-01-01 .. 2023-12-31 (the full year)

    The split uses the target time T. Lag features of the first validation
    rows use the last training days: that is allowed, because those values
    are in the past at the forecast time (a real forecast would use them too).
    """
    labels = pd.Series("test", index=index)
    labels[index <= VAL_END] = "val"
    labels[index <= TRAIN_END] = "train"
    return labels


def season(index):
    """Meteorological season: winter = Dec-Feb, spring = Mar-May,
    summer = Jun-Aug, autumn = Sep-Nov."""
    names = {12: "winter", 1: "winter", 2: "winter",
             3: "spring", 4: "spring", 5: "spring",
             6: "summer", 7: "summer", 8: "summer",
             9: "autumn", 10: "autumn", 11: "autumn"}
    return pd.Series(index.month.map(names), index=index)
