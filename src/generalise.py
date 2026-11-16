"""Generalisation checks (Stage 9).

    python -m src.generalise              # all three parts
    python -m src.generalise --part unseen|kfold|kaggle

1. UNSEEN STATIONS. The final models never trained on the unseen stations.
   Their 2023 test error is compared with the seen stations, per group:
   - "hold-out": the 6 random hold-outs (full 2023, the same period as seen);
   - "new":      stations with little or no data before 2023 (they cover only
                 part of 2023, so the period differs: read with care).
2. STATION-GROUPED K-FOLD for the best GBM. The seen stations are split into
   5 folds; each fold is predicted by a model trained without its stations
   and compared with the final model that did see them. This gives a spread
   instead of one draw of 6 stations.
3. KAGGLE TRANSFER SANITY CHECK (1 h only). A reduced model with only the
   features that exist in both datasets is trained on HKUST and applied
   unchanged to the Kaggle rooftop data (2024-2025). Time, hardware and
   weather station all differ, so this is a transfer check, not a benchmark.

Outputs:
    results/tables/09_generalisation.csv, 09_unseen_per_station.csv,
    results/tables/09_kfold.csv, 09_kaggle_transfer.csv
    results/figures/09_seen_vs_unseen.png, 09_kfold.png, 09_kaggle_transfer.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import evaluate, features, metrics, preprocess, splits
from src.explain import best_model_name
from src.models import baselines, gbm
from src.models.ridge import feature_columns
from src.train_gbm import load_config

KAGGLE = (evaluate.REPO / "data" / "raw" / "processed_data" / "processed_data"
          / "Downsampled" / "downsampled")
GROUP_NAMES = {"seen": "seen", "holdout": "unseen: random hold-out", "new": "unseen: new in 2023"}


# ------------------------------------------------------------ 1. unseen stations

def station_groups():
    """station -> "seen", "holdout" or "new"."""
    sets = splits.station_sets()
    group = sets["role"].where(sets["role"] != "unseen", "new")
    group[sets["reason"].str.startswith("random hold-out")] = "holdout"
    return pd.Series(group.to_numpy(), index=sets["station"])


def test_predictions(horizon):
    """Test predictions of the final models of one horizon (all stations)."""
    preds = pd.read_parquet(evaluate.PROCESSED / f"pred_gbm_{horizon}.parquet")
    lstm_file = evaluate.PROCESSED / f"pred_lstm_{horizon}.parquet"
    if lstm_file.exists():
        preds["lstm"] = pd.read_parquet(lstm_file)["lstm"].to_numpy()
    preds["group"] = preds["station"].map(station_groups())
    preds["rated_kW"] = evaluate.rated_kW(preds)
    return preds


def unseen_rows(horizon):
    """Daytime test metrics (with CIs) per station group and model."""
    preds = test_predictions(horizon)
    day = preds["usable"] & (preds["sun_elevation"] >= baselines.MIN_ELEVATION)
    models = [m for m in ["lgbm", "xgb", "lstm"] if m in preds]

    rows, per_station = [], []
    for group in ["seen", "holdout", "new"]:
        mask = day & (preds["group"] == group)
        for model in models:
            args = (preds["target"][mask], preds[model][mask], preds["benchmark"][mask],
                    preds["rated_kW"][mask])
            result = metrics.score(*args)
            result["mean_station_skill"] = metrics.mean_station_skill(*args, preds["station"][mask])
            result.update({"horizon": horizon, "model": model, "group": GROUP_NAMES[group],
                           "stations": preds["station"][mask].nunique()})
            rows.append(result)

            stations = metrics.per_station(*args, preds["station"][mask]).reset_index()
            stations.insert(0, "group", GROUP_NAMES[group])
            stations.insert(0, "model", model)
            stations.insert(0, "horizon", horizon)
            per_station.append(stations)
    return rows, pd.concat(per_station)


def plot_unseen(results):
    best = {h: best_model_name(h) for h in evaluate.HORIZONS}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    fig.patch.set_facecolor("#fcfcfb")
    colors = ["#2a78d6", "#d6602a", "#e0a526"]
    for ax, horizon in zip(axes, evaluate.HORIZONS):
        d = results[(results["horizon"] == horizon) & (results["model"] == best[horizon])]
        errors = [d["nmae_pct"] - d["nmae_pct_low"], d["nmae_pct_high"] - d["nmae_pct"]]
        ax.bar(range(len(d)), d["nmae_pct"], color=colors[:len(d)], width=0.6)
        ax.errorbar(range(len(d)), d["nmae_pct"], yerr=errors, fmt="none", ecolor="#52514e", capsize=3)
        for x, (v, n) in enumerate(zip(d["nmae_pct"], d["stations"])):
            ax.text(x, 0.15, f"{v:.1f}\n({n} st.)", ha="center", va="bottom", fontsize=8, color="white")
        ax.set_xticks(range(len(d)))
        ax.set_xticklabels([g.replace("unseen: ", "unseen:\n") for g in d["group"]], fontsize=8)
        ax.set_title(f"{horizon} ({best[horizon]})")
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("test nMAE, daytime (% of rated)")
    fig.suptitle("Seen vs unseen stations, 2023 test year (lines = 95 % CI)", fontsize=11)
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / "09_seen_vs_unseen.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def part_unseen():
    rows, stations = [], []
    for horizon in evaluate.HORIZONS:
        r, s = unseen_rows(horizon)
        rows += r
        stations.append(s)
    first = ["horizon", "model", "group", "stations", "n_rows", "n_days"]
    results = pd.DataFrame(rows)
    results = results[first + [c for c in results.columns if c not in first]]
    results["commit"] = evaluate.git_commit()
    results.to_csv(evaluate.TABLES / "09_generalisation.csv", index=False, float_format="%.5f")
    pd.concat(stations).to_csv(evaluate.TABLES / "09_unseen_per_station.csv", index=False,
                               float_format="%.5f")
    plot_unseen(results)
    print(results[["horizon", "model", "group", "stations", "nmae_pct", "nmae_pct_low",
                   "nmae_pct_high", "skill", "mean_station_skill"]].to_string(index=False))


# ------------------------------------------------------------ 2. grouped K-fold

def kfold_horizon(horizon):
    """One row per fold: error on the fold's stations when they were left
    out of training, and when they were in it (the final model)."""
    name = best_model_name(horizon)
    config = load_config(name, horizon)
    table = evaluate.load_table(horizon)
    bench = evaluate.benchmark(table, horizon)
    rated = evaluate.rated_kW(table)
    folds = splits.station_folds(horizon)

    final = gbm.fit_fixed(table, name, config["params"], config["n_trees"], ["train", "val"])
    pred_final = gbm.predict(final, table)

    rows = []
    for fold in sorted(folds.unique()):
        stations = folds.index[folds == fold]
        without = table.copy()
        without.loc[without["station"].isin(stations), "role"] = "left out"
        model = gbm.fit_fixed(without, name, config["params"], config["n_trees"], ["train", "val"])
        pred = gbm.predict(model, table)

        mask = evaluate.eval_mask(table, "test", "daytime") & table["station"].isin(stations)
        left_out = metrics.score(table["target"][mask], pred[mask], bench[mask], rated[mask], n_boot=0)
        trained = metrics.score(table["target"][mask], pred_final[mask], bench[mask], rated[mask], n_boot=0)
        rows.append({
            "horizon": horizon, "model": name, "fold": fold, "stations": len(stations),
            "nmae_left_out": left_out["nmae_pct"], "nmae_trained_on": trained["nmae_pct"],
            "skill_left_out": left_out["skill"], "skill_trained_on": trained["skill"],
            "nmae_gap_pct_points": left_out["nmae_pct"] - trained["nmae_pct"],
            "nmae_gap_relative_pct": 100 * (left_out["nmae_pct"] / trained["nmae_pct"] - 1),
        })
        print(f"   {horizon} fold {fold}: left out {left_out['nmae_pct']:.2f} %, "
              f"trained on {trained['nmae_pct']:.2f} %")
    return rows


def plot_kfold(results):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    fig.patch.set_facecolor("#fcfcfb")
    for ax, horizon in zip(axes, evaluate.HORIZONS):
        d = results[results["horizon"] == horizon]
        x = np.arange(len(d))
        ax.bar(x - 0.2, d["nmae_trained_on"], width=0.4, color="#2a78d6", label="stations in training")
        ax.bar(x + 0.2, d["nmae_left_out"], width=0.4, color="#d6602a", label="stations left out")
        ax.set_xticks(x)
        ax.set_xticklabels([f"fold {f}" for f in d["fold"]], fontsize=8)
        ax.set_title(f"{horizon}: mean gap {d['nmae_gap_pct_points'].mean():+.2f} points")
        ax.set_facecolor("#fcfcfb")
        ax.grid(axis="y", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("test nMAE, daytime (% of rated)")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Station-grouped 5-fold: the same stations, with and without them in training",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / "09_kfold.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def part_kfold():
    rows = []
    for horizon in evaluate.HORIZONS:
        rows += kfold_horizon(horizon)
    results = pd.DataFrame(rows)
    results["commit"] = evaluate.git_commit()
    results.to_csv(evaluate.TABLES / "09_kfold.csv", index=False, float_format="%.5f")
    plot_kfold(results)
    summary = results.groupby("horizon")[["nmae_gap_pct_points", "nmae_gap_relative_pct"]].agg(
        ["mean", "std", "min", "max"])
    print(summary.round(3).to_string())


# ------------------------------------------------------------ 3. Kaggle transfer

# features that exist in both datasets (no wind / pressure / visibility:
# different sensors and units; no station metadata: unknown for Kaggle)
def common_features(table):
    drop = ["wind_speed_lag1", "pressure_lag1", "visibility_lag1",
            "is_optimizer", "tilt_deg", "multi_orientation"]
    return [c for c in feature_columns(table) if c not in drop]


def kaggle_table(peak_share):
    """The Kaggle site as one "station" in the 1-h feature format.

    - power: sum of the 31 panel optimizers (hours where all 31 report);
      hours with the sun below 5 degrees are set to 0, as for HKUST;
    - capacity is NOT given in the files. Proxy: the 99th percentile of the
      site's daytime power is assumed to sit at the same share of rated power
      as for a typical HKUST station (`peak_share`, the median over the seen
      stations' training data);
    - location: the HKUST campus (median of the station coordinates);
    - irradiance: its own sensor, so the reading is put on the HKUST
      training-period scale with the same trailing clear-day level rule.
    """
    power = pd.read_csv(KAGGLE / "power" / "power_merged.csv", parse_dates=["DateTime"],
                        index_col="DateTime")
    weather = pd.read_csv(KAGGLE / "weather" / "weather_merged.csv", parse_dates=["DateTime"],
                          index_col="DateTime")
    inventory = pd.read_csv(evaluate.INVENTORY)
    lat, lon = inventory["latitude"].median(), inventory["longitude"].median()

    total_W = power.sum(axis=1, min_count=power.shape[1])
    solar = features.solar_features(total_W.index, "1h", lat, lon)
    night = solar["sun_elevation"] < baselines.MIN_ELEVATION
    total_W[night.to_numpy()] = 0.0

    rated_W = total_W[~night.to_numpy()].quantile(0.99) / peak_share
    station = pd.DataFrame({"power_norm": total_W / rated_W, "is_filled": False})
    station = station[weather.index.min():weather.index.max()]

    # the same causal sensor-level correction as for HKUST, onto the HKUST
    # training-period scale (`preprocess.irradiance_level`)
    hkust_level = preprocess.irradiance_level(preprocess.build_weather("1h")["irradiance_raw"])
    reference = float(hkust_level[:preprocess.LEVEL_REFERENCE_END].median())
    level = preprocess.irradiance_level(weather["GHI_Wm2"], lat, lon)
    irradiance, _ = preprocess.calibrate_irradiance(weather["GHI_Wm2"], level, reference)
    print(f"   Kaggle sensor level: median {level.median():.2f} of clear-sky "
          f"(HKUST training period: {reference:.2f})")

    w = pd.DataFrame({
        "irradiance": irradiance, "temperature": weather["Air_Temp_C"],
        "wind_speed": weather["WindSpeed_ms"], "pressure": weather["Pressure_Pa"] / 100,
        "visibility": weather["Visibility_m"] / 1000})
    base = features.build_base(station, w, "1h", lat, lon)
    out = features.features_1h(base)
    out["usable"] = features.usable_target(out)
    out["station"] = "Kaggle rooftop"
    out["role"], out["split"] = "seen", "test"
    return out, rated_W / 1000


def part_kaggle():
    table = evaluate.load_table("1h")
    columns = common_features(table)
    config = load_config("lgbm", "1h")

    # reduced model: the tuned 1-h LightGBM setting, common features only
    reduced = table[columns + ["target", "usable", "station", "role", "split"]]
    model = gbm.fit_fixed(reduced, "lgbm", config["params"], config["n_trees"], ["train", "val"])

    train = table[(table["role"] == "seen") & (table["split"] == "train")
                  & (table["sun_elevation"] >= baselines.MIN_ELEVATION)]
    peak_share = float(train.groupby("station")["target"].quantile(0.99).median())
    kaggle, rated_kW = kaggle_table(peak_share)
    print(f"   HKUST median 99th-percentile output: {peak_share:.3f} of rated "
          f"-> Kaggle capacity proxy {rated_kW:.1f} kW")

    rows = []
    for name, data, rated in [("HKUST test 2023 (seen stations)", table, evaluate.rated_kW(table)),
                              ("Kaggle rooftop 2024-2025 (zero-shot)", kaggle,
                               pd.Series(rated_kW, index=kaggle.index))]:
        preds = {"reduced lgbm": baselines.finish(model.predict(data[columns]), data),
                 "smart_persistence": baselines.smart_persistence(data)}
        bench = baselines.last_value(data)
        mask = evaluate.eval_mask(data, "test", "daytime")
        for model_name, pred in preds.items():
            result = metrics.score(data["target"][mask], pred[mask], bench[mask], rated[mask])
            result.update({"data": name, "model": model_name, "benchmark": "last_value"})
            rows.append(result)

    first = ["data", "model", "benchmark", "n_rows", "n_days"]
    results = pd.DataFrame(rows)
    results = results[first + [c for c in results.columns if c not in first]]
    results["capacity_proxy_kW"] = [np.nan, np.nan, rated_kW, rated_kW]
    results["commit"] = evaluate.git_commit()
    results.to_csv(evaluate.TABLES / "09_kaggle_transfer.csv", index=False, float_format="%.5f")

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    labels = [f"{r.model}\n{r.data.split(' (')[0]}" for r in results.itertuples()]
    errors = [results["skill"] - results["skill_low"], results["skill_high"] - results["skill"]]
    ax.bar(range(4), results["skill"], color=["#2a78d6", "#7a7974", "#2a78d6", "#7a7974"], width=0.6)
    ax.errorbar(range(4), results["skill"], yerr=errors, fmt="none", ecolor="#52514e", capsize=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0, color="#52514e", lw=0.6)
    ax.set_ylabel("skill vs last-value persistence")
    ax.set_title("1-h transfer check: reduced model, daytime (lines = 95 % CI)", fontsize=11)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / "09_kaggle_transfer.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(results[["data", "model", "n_rows", "n_days", "nmae_pct", "rmse_norm", "skill",
                   "skill_low", "skill_high"]].to_string(index=False))


PARTS = {"unseen": part_unseen, "kfold": part_kfold, "kaggle": part_kaggle}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 9: generalisation.")
    parser.add_argument("--part", choices=sorted(PARTS))
    args = parser.parse_args()
    for name in ([args.part] if args.part else ["unseen", "kfold", "kaggle"]):
        print(f"== {name}")
        PARTS[name]()
