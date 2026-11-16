"""Residual-based fault / underperformance flagging (Stage 8).

    python -m src.faults --build     # build the nowcast table (once, ~2 min)
    python -m src.faults             # train, set thresholds, inject faults, evaluate

Design (PLAN Stage 8, "Key design point"):
A forecast that uses the station's own lagged power adapts to a lasting fault
(the lags drop too), so its residual shrinks and hides the fault. Flagging
therefore uses a dedicated EXPECTED-POWER model with no own-power inputs:
calendar, sun position, clear-sky irradiance, and the weather measured during
the SAME hour (a nowcast), plus static station metadata.

Two detectors, built the same way so the comparison is fair:
- "model":  daily ratio  actual energy / expected energy (LightGBM nowcast);
- "pr":     daily performance ratio  actual energy / irradiance (baseline).
For each: divide by the station's normal level, subtract 1 (0 = normal,
-0.2 = 20 % below normal), take a 3-day rolling mean, and flag a day when it
is below a threshold.

Calibration (normal level and threshold) uses the train + validation period,
which covers every season. So that the model's expected power is not
in-sample there, it comes from 4 folds: each month is predicted by a model
that was fitted without that month ("out-of-fold"). The test year is
predicted by one model fitted on all of train + validation.
- normal level of a station = median daily ratio in the calibration period;
- threshold = -3 robust standard deviations of the calibration scores of all
  stations (robust = from the median absolute deviation, so real outages in
  the calibration period do not move it).

There are no fault labels, so the detectors are scored on SYNTHETIC faults
injected into the 2023 test data (several severities and durations).
"""
import argparse

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import data_io, evaluate, features, pipeline, preprocess, splits
from src.models import gbm
from src.models.baselines import finish

SEED = 42
NOWCAST = evaluate.PROCESSED / "features_nowcast.parquet"
DAILY = evaluate.PROCESSED / "fault_daily.parquet"

MIN_IRRADIANCE = 200      # W/m2: hours used in the daily sums (stable ratios)
MIN_HOURS = 3             # a day needs at least 3 such hours
ROLL_DAYS = 3             # rolling window of the score
N_SIGMA = 3               # threshold = -3 robust standard deviations
N_FOLDS = 4               # month blocks for the out-of-fold expected power
DETECTORS = ["model", "pr"]

# Synthetic faults: (kind, severity). Severity = share of output lost.
FAULTS = [("drop", 0.10), ("drop", 0.20), ("drop", 0.30), ("drop", 0.50),
          ("zero", 1.00), ("gradual", 0.30)]
DURATIONS = [3, 7, 14]    # days
STARTS_PER_STATION = 5

PARAMS = {"learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
          "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0}


# ------------------------------------------------------------ nowcast table

def features_nowcast(base):
    """Features of hour T that do not use the station's own power."""
    out = features.target_time_features(base)
    for col in ["irradiance", "csi", "temperature", "wind_speed"]:
        out[col] = base[col]
    features.add_target(out, base)
    return out


def build_nowcast_table():
    """Hourly nowcast table for all stations of the 1-h set."""
    inventory = pd.read_csv(pipeline.INVENTORY)
    files = data_io.list_station_files().set_index("station")
    sets = splits.station_sets().set_index("station")
    meta = features.station_features(inventory)
    weather = preprocess.build_weather("1h")

    stations = splits.stations_for("1h", "seen") + splits.stations_for("1h", "unseen")
    tables = []
    for i, station in enumerate(sorted(stations), start=1):
        print(f"[{i}/{len(stations)}] {station}")
        row = inventory.set_index("station").loc[station]
        r = files.loc[station]
        df = pipeline.clean_station(station, r["path"], r["has_optimizer"], inventory)
        if r["has_optimizer"]:
            df = preprocess.to_hourly(df)
        base = features.build_base(df, weather, "1h", row["latitude"], row["longitude"])
        out = features_nowcast(base)
        out["usable"] = features.usable_target(out)
        out["station"] = station
        out = features.add_station_features(out, meta.loc[station])
        out["role"] = sets.loc[station, "role"]
        out["split"] = splits.time_split(out.index)
        tables.append(out)

    table = pd.concat(tables)
    table.index.name = "time"
    table.to_parquet(NOWCAST)
    print(f"saved {NOWCAST}: {len(table)} rows")
    return table


# ------------------------------------------------------------ daily scores

def daily_table(table, expected):
    """Daily sums per station over the bright hours with an observed target.

    Zero-output hours (`target_is_fault`) are KEPT here: they are real faults
    and the detector should see them.
    """
    ok = (table["target"].notna() & ~table["target_is_filled"]
          & (table["sun_elevation"] >= 5) & (table["irradiance"] >= MIN_IRRADIANCE)
          & expected.notna())
    d = pd.DataFrame({
        "station": table["station"][ok],
        "day": table.index[ok].normalize(),
        "hours": 1,
        "actual": table["target"][ok],
        "expected": expected[ok],
        "irradiance_kwh": table["irradiance"][ok] / 1000,
        "known_zero_hours": table["target_is_fault"][ok].astype(int),
    })
    daily = d.groupby(["station", "day"]).sum().reset_index()
    daily = daily[daily["hours"] >= MIN_HOURS]
    daily["split"] = splits.time_split(pd.DatetimeIndex(daily["day"])).to_numpy()
    return daily


def raw_ratio(daily, detector, factor=1.0):
    """Daily ratio of one detector; `factor` scales the actual energy
    (1 = no fault, 0.8 = 20 % of the output lost)."""
    actual = daily["actual"] * factor
    if detector == "model":
        return actual / daily["expected"]
    return actual / daily["irradiance_kwh"]


def station_normal(daily):
    """Normal level of each station and detector: the median daily ratio on
    calibration days (train + validation) without known zero-output hours."""
    val = daily[(daily["split"] != "test") & (daily["known_zero_hours"] == 0)]
    normal = pd.DataFrame({
        d: raw_ratio(val, d).groupby(val["station"]).median() for d in DETECTORS})
    return normal


def rolling_score(days, ratio, normal):
    """Score of one station: ratio / normal - 1, as a 3-day rolling mean on
    the calendar (a day without data is skipped; at least 2 days needed)."""
    score = pd.Series((ratio / normal - 1).to_numpy(), index=pd.DatetimeIndex(days))
    full = score.asfreq("D")
    rolled = full.rolling(ROLL_DAYS, min_periods=2).mean()
    return rolled


def all_scores(daily, normal, split):
    """Rolling score of every station and detector (no injected fault) on
    the test year (`split` = "test") or the calibration period (anything else)."""
    part = daily[(daily["split"] == "test") == (split == "test")]
    out = []
    for station, d in part.groupby("station"):
        if station not in normal.index:
            continue
        frame = pd.DataFrame({
            det: rolling_score(d["day"], raw_ratio(d, det), normal.loc[station, det])
            for det in DETECTORS})
        frame["station"] = station
        out.append(frame)
    scores = pd.concat(out)
    scores.index.name = "day"
    return scores.reset_index()


def robust_sd(values):
    """Standard deviation estimated from the median absolute deviation.
    A few extreme values (real outages) hardly change it."""
    values = values.dropna()
    return float(1.4826 * (values - values.median()).abs().median())


def thresholds_from(calibration_scores):
    """-N_SIGMA robust standard deviations of the calibration scores, per detector."""
    return {det: -N_SIGMA * robust_sd(calibration_scores[det]) for det in DETECTORS}


def expected_power(table):
    """Expected power of every row from the model without own-power inputs.

    - train + validation rows: out-of-fold (the row's month was not used to
      fit the model that predicts it);
    - test rows: one model fitted on all of train + validation.
    Returns (expected, final model, number of trees).
    """
    # the number of trees comes from early stopping on validation
    _, n_trees = gbm.fit_early_stopping(table, "lgbm", PARAMS)

    month_number = table.index.year * 12 + table.index.month
    fold = pd.Series(month_number % N_FOLDS, index=table.index)
    is_test = table["split"] == "test"
    expected = pd.Series(np.nan, index=table.index)

    for k in range(N_FOLDS):
        others = table[(fold != k).to_numpy()]
        model = gbm.fit_fixed(others, "lgbm", PARAMS, n_trees, ["train", "val"])
        rows = ((fold == k) & ~is_test).to_numpy()
        expected[rows] = gbm.predict(model, table[rows]).to_numpy()

    final = gbm.fit_fixed(table, "lgbm", PARAMS, n_trees, ["train", "val"])
    expected[is_test.to_numpy()] = gbm.predict(final, table[is_test.to_numpy()]).to_numpy()
    return expected, final, n_trees


# ------------------------------------------------------------ synthetic faults

def fault_factor(days, start, duration, kind, severity):
    """Share of the output that is left on each day (1 = healthy)."""
    days = pd.DatetimeIndex(days)
    offset = (days - start).days
    inside = (offset >= 0) & (offset < duration)
    factor = np.ones(len(days))
    if kind == "gradual":
        # loss grows linearly from 0 to `severity` over the fault
        factor[inside] = 1 - severity * (offset[inside] + 1) / duration
    else:
        factor[inside] = 1 - severity
    return factor


def pick_starts(daily_test, rng):
    """Random fault start days per station (fixed seed), early enough that
    the longest fault fits before the end of the year."""
    latest = pd.Timestamp("2023-12-31") - pd.Timedelta(days=max(DURATIONS) + ROLL_DAYS)
    starts = {}
    for station, d in daily_test.groupby("station"):
        days = d["day"][d["day"] <= latest]
        if len(days) < 60:
            continue
        chosen = rng.choice(days.to_numpy(), size=STARTS_PER_STATION, replace=False)
        starts[station] = [pd.Timestamp(c) for c in sorted(chosen)]
    return starts


def run_injections(daily, normal, thresholds):
    """Inject every fault at every start into every station; record whether
    and when each detector flags it.

    Detected = a flagged day between the fault start and (fault end +
    ROLL_DAYS - 1 days): the rolling mean still contains fault days then.
    """
    test = daily[daily["split"] == "test"]
    test = test[test["station"].isin(normal.index)]
    rng = np.random.default_rng(SEED)
    starts = pick_starts(test, rng)

    events = []
    for station, d in test.groupby("station"):
        for start in starts.get(station, []):
            for kind, severity in FAULTS:
                for duration in DURATIONS:
                    factor = fault_factor(d["day"], start, duration, kind, severity)
                    end = start + pd.Timedelta(days=duration + ROLL_DAYS - 2)
                    for det in DETECTORS:
                        score = rolling_score(d["day"], raw_ratio(d, det, factor),
                                              normal.loc[station, det])
                        window = score[start:end]
                        flagged = window[window < thresholds[det]]
                        events.append({
                            "station": station, "start": start, "kind": kind,
                            "severity": severity, "duration_days": duration,
                            "detector": det, "detected": len(flagged) > 0,
                            "delay_days": (flagged.index[0] - start).days if len(flagged) else np.nan,
                        })
    return pd.DataFrame(events)


def episodes(scores, threshold, detector):
    """Runs of consecutive flagged days (one row per run)."""
    rows = []
    for station, d in scores.groupby("station"):
        d = d.sort_values("day")
        flag = (d[detector] < threshold).to_numpy()
        run_id = np.cumsum(np.r_[True, flag[1:] != flag[:-1]])
        for _, run in d[flag].groupby(run_id[flag]):
            rows.append({"detector": detector, "station": station,
                         "start": run["day"].iloc[0], "end": run["day"].iloc[-1],
                         "days": len(run), "lowest_score": run[detector].min()})
    return pd.DataFrame(rows)


def false_alarms(test_scores, thresholds):
    """Flags on the test data as it is (no injected fault).

    These are false alarms OR real problems: there are no labels to tell
    them apart, so this is an upper limit of the false-alarm rate.
    """
    rows, all_runs = [], []
    for det in DETECTORS:
        runs = episodes(test_scores, thresholds[det], det)
        all_runs.append(runs)
        valid = test_scores[det].notna()
        station_months = valid.sum() / 30.4
        rows.append({
            "detector": det, "threshold": thresholds[det],
            "stations": test_scores["station"].nunique(),
            "scored_days": int(valid.sum()),
            "flagged_days": int((test_scores[det] < thresholds[det]).sum()),
            "flagged_day_share_pct": 100 * (test_scores[det] < thresholds[det]).sum() / valid.sum(),
            "episodes": len(runs),
            "episodes_per_station_month": len(runs) / station_months,
        })
    return pd.DataFrame(rows), pd.concat(all_runs)


def summarise(events, alarms):
    """Recall, precision and delay per detector, fault kind, severity, duration.

    Precision is at event level and assumes ONE fault per station-year:
    true alarms = detected faults; false alarms = that detector's flagged
    episodes on the unmodified test year of the same station (counted once
    per injected scenario).
    """
    clean_per_station = (alarms.set_index("detector")["episodes"]
                         / alarms.set_index("detector")["stations"])
    keys = ["detector", "kind", "severity", "duration_days"]
    rows = []
    for key, g in events.groupby(keys):
        true_alarms = g["detected"].sum()
        false = len(g) * clean_per_station[key[0]]
        rows.append(dict(zip(keys, key)) | {
            "n_events": len(g),
            "recall": g["detected"].mean(),
            "precision": true_alarms / (true_alarms + false) if true_alarms + false > 0 else np.nan,
            "median_delay_days": g["delay_days"].median(),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ lag example

def lag_example(nowcast, expected, normal, thresholds):
    """One fault (-30 % for 14 days) seen by three residuals:
    the nowcast detector, the PR baseline, and the Stage 4 1-h forecast
    whose inputs include the station's own (now also reduced) lagged power.
    """
    station, start, duration, severity = "LSK North", pd.Timestamp("2023-07-10"), 14, 0.30
    end = start + pd.Timedelta(days=duration)

    is_station = nowcast["station"] == station
    mine = nowcast[is_station]
    d = daily_table(mine, expected[is_station])
    d = d[d["split"] == "test"]
    factor = fault_factor(d["day"], start, duration, "drop", severity)
    lines = {det: rolling_score(d["day"], raw_ratio(d, det, factor), normal.loc[station, det])
             for det in DETECTORS}

    # the 1-h forecast with faulty power: the target AND the own-power inputs drop
    table = evaluate.load_table("1h")
    table = table[(table["station"] == station) & (table["split"] == "test")].copy()
    inside = (table.index >= start) & (table.index < end)
    power_columns = [c for c in table.columns if c.startswith("power_")]
    table.loc[inside, power_columns] = table.loc[inside, power_columns] * (1 - severity)
    table.loc[inside, "target"] = table.loc[inside, "target"] * (1 - severity)

    booster = lgb.Booster(model_file=str(evaluate.MODELS / "gbm_lgbm_1h.txt"))
    forecast = finish(booster.predict(table[booster.feature_name()]), table)
    irr = mine["irradiance"].reindex(table.index)
    bright = (table["target"].notna() & (table["sun_elevation"] >= 5)
              & (irr >= MIN_IRRADIANCE).to_numpy() & forecast.notna())
    day = table.index[bright].normalize()
    ratio = table["target"][bright].groupby(day).sum() / forecast[bright].groupby(day).sum()
    lag_line = (ratio - 1).asfreq("D").rolling(ROLL_DAYS, min_periods=2).mean()

    show = slice(start - pd.Timedelta(days=21), end + pd.Timedelta(days=21))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.axvspan(start, end, color="#f3d9c9", label="injected fault: -30 % for 14 days")
    ax.plot(lines["model"][show], color="#2a78d6", lw=2, label="expected-power model (no own-power inputs)")
    ax.plot(lines["pr"][show], color="#7a7974", lw=1.3, label="performance-ratio baseline")
    ax.plot(lag_line[show], color="#d6602a", lw=2, ls="--", label="1-h forecast with own-power lags")
    ax.axhline(thresholds["model"], color="#2a78d6", lw=0.8, ls=":")
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_ylabel("3-day mean of (actual / expected - 1)")
    ax.set_title(f"{station}: the lag-based forecast adapts to the fault and hides it", fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    ax.grid(color="#e6e5e1", lw=0.8)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / "08_lag_vs_nowcast.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)

    during = slice(start + pd.Timedelta(days=ROLL_DAYS), end - pd.Timedelta(days=1))
    return {"station": station, "start": start.date(), "duration_days": duration,
            "severity": severity,
            "mean_score_model": lines["model"][during].mean(),
            "mean_score_pr": lines["pr"][during].mean(),
            "mean_score_lag_forecast": lag_line[during].mean()}


# ------------------------------------------------------------ figures

def plot_recall(summary):
    """Recall against fault severity (constant drops), one panel per duration."""
    drops = summary[summary["kind"].isin(["drop", "zero"])]
    fig, axes = plt.subplots(1, len(DURATIONS), figsize=(12, 3.6), sharey=True)
    fig.patch.set_facecolor("#fcfcfb")
    style = {"model": ("#2a78d6", "expected-power model"), "pr": ("#7a7974", "performance-ratio baseline")}
    for ax, duration in zip(axes, DURATIONS):
        for det, (color, label) in style.items():
            d = drops[(drops["duration_days"] == duration) & (drops["detector"] == det)]
            d = d.sort_values("severity")
            ax.plot(100 * d["severity"], d["recall"], marker="o", color=color, lw=2, label=label)
        ax.set_facecolor("#fcfcfb")
        ax.set_title(f"fault lasts {duration} days")
        ax.set_xlabel("output lost (%)")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(color="#e6e5e1", lw=0.8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("share of injected faults detected (recall)")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Synthetic faults on the 2023 test year, seen stations", fontsize=11)
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / "08_recall_by_severity.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_score_spread(val_scores, test_scores, thresholds):
    """How wide the fault-free score is for each detector (narrow = sensitive)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    names = {"model": "expected-power model", "pr": "performance-ratio baseline"}
    for ax, det in zip(axes, DETECTORS):
        ax.set_facecolor("#fcfcfb")
        bins = np.linspace(-0.6, 0.6, 61)
        ax.hist(test_scores[det].dropna(), bins=bins, color="#2a78d6", alpha=0.85, label="test 2023")
        ax.hist(val_scores[det].dropna(), bins=bins, color="#d6602a", alpha=0.5, label="calibration (train + val)")
        ax.axvline(thresholds[det], color="#0b0b0b", ls="--", lw=1,
                   label=f"threshold {thresholds[det]:.2f}")
        ax.set_title(names[det])
        ax.set_xlabel("3-day mean of (ratio / normal - 1)")
        ax.legend(fontsize=8, frameon=False)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("station-days")
    fig.suptitle("Score without injected faults (0 = normal)", fontsize=11)
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / "08_score_spread.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


# ------------------------------------------------------------ main

def main():
    table = pd.read_parquet(NOWCAST)

    # station identity as a category: the model may learn each station's own
    # response to the weather (shading, orientation). It is a fixed label,
    # not a measurement, so a fault cannot change it.
    table["station_code"] = table["station"].astype("category")

    print("== expected-power model (out-of-fold on train + val, final model on test)")
    expected, model, n_trees = expected_power(table)
    evaluate.MODELS.mkdir(exist_ok=True)
    model.booster_.save_model(str(evaluate.MODELS / "expected_power_lgbm.txt"))
    quality = []
    for split in ["train", "val", "test"]:
        mask = evaluate.eval_mask(table, split, "daytime")
        err = expected[mask] - table["target"][mask]
        quality.append({"split": split, "expected_from": "final model" if split == "test" else "out-of-fold",
                        "n_trees": n_trees, "daytime_rmse_norm": float(np.sqrt((err ** 2).mean())),
                        "daytime_nmae_pct": float(100 * err.abs().mean())})
        print(f"   {n_trees} trees, {split} daytime RMSE {quality[-1]['daytime_rmse_norm']:.4f} (power/rated)")
    pd.DataFrame(quality).to_csv(evaluate.TABLES / "08_expected_power_model.csv", index=False,
                                 float_format="%.4f")

    seen = table["role"] == "seen"
    daily = daily_table(table[seen], expected[seen])
    normal = station_normal(daily)
    val_scores = all_scores(daily, normal, "calibration")
    test_scores = all_scores(daily, normal, "test")
    thresholds = thresholds_from(val_scores)
    print(f"   thresholds (-{N_SIGMA} robust sd of calibration scores): {thresholds}")

    alarms, runs = false_alarms(test_scores, thresholds)
    events = run_injections(daily, normal, thresholds)
    summary = summarise(events, alarms)

    commit = evaluate.git_commit()
    for frame in [summary, alarms]:
        frame["commit"] = commit
    summary.to_csv(evaluate.TABLES / "08_fault_detection.csv", index=False, float_format="%.4f")
    alarms.to_csv(evaluate.TABLES / "08_false_alarms.csv", index=False, float_format="%.4f")

    # real flagged periods of the model detector (candidates, not ground truth)
    real = runs[runs["detector"] == "model"].sort_values("lowest_score")
    zero_days = daily[daily["known_zero_hours"] > 0].groupby("station")["day"].apply(set)
    real["has_known_zero_output_hours"] = [
        any(day in zero_days.get(r.station, set())
            for day in pd.date_range(r.start - pd.Timedelta(days=ROLL_DAYS - 1), r.end))
        for r in real.itertuples()]
    real.to_csv(evaluate.TABLES / "08_flagged_periods.csv", index=False, float_format="%.4f")

    normal.to_csv(evaluate.TABLES / "08_station_normal.csv", float_format="%.4f")
    daily.to_parquet(DAILY)      # daily sums, read by the dashboard bundle

    plot_recall(summary)
    plot_score_spread(val_scores, test_scores, thresholds)
    example = lag_example(table, expected, normal, thresholds)
    pd.DataFrame([example]).to_csv(evaluate.TABLES / "08_lag_example.csv", index=False, float_format="%.4f")

    print(alarms.to_string(index=False))
    print(summary.to_string(index=False))
    print(example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 8: fault flagging.")
    parser.add_argument("--build", action="store_true", help="build the nowcast table first")
    args = parser.parse_args()
    if args.build:
        build_nowcast_table()
    main()
