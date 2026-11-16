"""Final gradient-boosting models (Stage 4).

    python -m src.train_gbm                  # all horizons
    python -m src.train_gbm --horizon 1h     # one horizon

For each horizon and each model (LightGBM, XGBoost), with the tuned setting
from `configs/gbm_<model>_<horizon>.yaml` (PLAN §3 "Final training"):
    1. fit on TRAIN with the tuned number of trees -> validation scores;
    2. refit on TRAIN + VALIDATION with the same setting and the same
       number of trees -> the final model, scored on TEST once.

Outputs:
    models/gbm_<model>_<horizon>.txt|.json     final models (git-ignored)
    models/features_<horizon>.json             feature names, in model order
    data/processed/pred_gbm_<horizon>.parquet  test predictions (git-ignored)
    results/tables/04_gbm_results.csv          val + test, all hours + daytime, CIs
    results/tables/04_gbm_per_station.csv      test, daytime
    results/tables/04_gbm_per_season.csv       test, daytime
    results/figures/04_gbm_skill.png, 04_gbm_nmae.png
"""
import argparse
import json
import time

import pandas as pd
import yaml

from src import evaluate
from src.models import gbm
from src.models.ridge import feature_columns

CONFIGS = evaluate.REPO / "configs"
MODEL_NAMES = ["lgbm", "xgb"]
RESULTS = evaluate.TABLES / "04_gbm_results.csv"
PER_STATION = evaluate.TABLES / "04_gbm_per_station.csv"
PER_SEASON = evaluate.TABLES / "04_gbm_per_season.csv"


def load_config(model_name, horizon):
    path = CONFIGS / f"gbm_{model_name}_{horizon}.yaml"
    return yaml.safe_load(path.read_text())


def model_path(model_name, horizon):
    ending = "txt" if model_name == "lgbm" else "json"
    return evaluate.MODELS / f"gbm_{model_name}_{horizon}.{ending}"


def save_model(model, model_name, horizon):
    path = model_path(model_name, horizon)
    if model_name == "lgbm":
        model.booster_.save_model(str(path))
    else:
        model.save_model(str(path))
    return path


def run_horizon(horizon):
    """Train, evaluate and save both models of one horizon."""
    table = evaluate.load_table(horizon)
    bench = evaluate.benchmark(table, horizon)
    rated = evaluate.rated_kW(table)

    evaluate.MODELS.mkdir(exist_ok=True)
    features_file = evaluate.MODELS / f"features_{horizon}.json"
    features_file.write_text(json.dumps(feature_columns(table)))

    rows, stations, seasons = [], [], []
    test = table["split"] == "test"
    keep = ["station", "role", "target", "usable", "sun_elevation"]
    preds = table.loc[test, keep].copy()
    preds["benchmark"] = bench[test]

    for name in MODEL_NAMES:
        config = load_config(name, horizon)
        params, n_trees = config["params"], config["n_trees"]

        # 1. train only -> validation
        on_train = gbm.fit_fixed(table, name, params, n_trees, ["train"])
        pred_val = gbm.predict(on_train, table)
        rows += evaluate.score_model(table, pred_val, bench, rated, horizon, name, "val")

        # 2. train + validation -> test (once)
        start = time.time()
        final = gbm.fit_fixed(table, name, params, n_trees, ["train", "val"])
        seconds = time.time() - start
        pred = gbm.predict(final, table)
        test_rows = evaluate.score_model(table, pred, bench, rated, horizon, name, "test")
        for row in test_rows:
            row["fit_seconds"] = round(seconds, 1)      # used by the Stage 6 benchmark
        rows += test_rows

        stations.append(evaluate.score_per_station(table, pred, bench, rated, horizon, name))
        seasons.append(evaluate.score_per_season(table, pred, bench, rated, horizon, name))
        preds[name] = pred[test]

        path = save_model(final, name, horizon)
        day = test_rows[1]
        print(f"   {name}: {n_trees} trees, final fit {seconds:.0f} s, test daytime "
              f"nMAE {day['nmae_pct']:.2f} %, skill {day['skill']:.3f} -> {path.name}")

    preds.to_parquet(evaluate.PROCESSED / f"pred_gbm_{horizon}.parquet")
    return rows, pd.concat(stations), pd.concat(seasons)


def merge_csv(new, path, horizons):
    """Replace the rows of `horizons` in an existing table, keep the others."""
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[~old["horizon"].isin(horizons)], new])
    new.to_csv(path, index=False, float_format="%.5f")
    return new


def plots():
    """GBMs next to the Stage 3 baselines, test set, daytime."""
    both = pd.concat([pd.read_csv(evaluate.TABLES / "03_baselines.csv"), pd.read_csv(RESULTS)])
    title = "2023 test set, daytime, seen stations (lines = 95 % CI)"
    evaluate.plot_test_daytime(both, evaluate.FIGURES / "04_gbm_nmae.png",
                               "nMAE: " + title + ", lower is better")
    evaluate.plot_test_daytime(both, evaluate.FIGURES / "04_gbm_skill.png",
                               "Skill vs benchmark persistence: " + title + ", higher is better",
                               metric="skill", label="skill = 1 - RMSE / RMSE benchmark")


def main(horizons):
    all_rows, all_stations, all_seasons = [], [], []
    for horizon in horizons:
        print(f"== {horizon}")
        rows, stations, seasons = run_horizon(horizon)
        all_rows += rows
        all_stations.append(stations)
        all_seasons.append(seasons)

    results = pd.DataFrame(all_rows)
    first = evaluate.FIRST_COLUMNS
    results = results[first + [c for c in results.columns if c not in first]]
    results["commit"] = evaluate.git_commit()
    results["date"] = pd.Timestamp.today().date().isoformat()
    merge_csv(results, RESULTS, horizons)
    merge_csv(pd.concat(all_stations), PER_STATION, horizons)
    merge_csv(pd.concat(all_seasons), PER_SEASON, horizons)
    print(f"saved {RESULTS.name}, {PER_STATION.name}, {PER_SEASON.name}")
    plots()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Final GBMs: refit, test, save.")
    parser.add_argument("--horizon", choices=evaluate.HORIZONS)
    args = parser.parse_args()
    main([args.horizon] if args.horizon else evaluate.HORIZONS)
