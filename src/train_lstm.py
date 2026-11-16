"""Train and evaluate the LSTM (Stage 5).

    python -m src.train_lstm                 # 15 min and 1 h
    python -m src.train_lstm --horizon 1h

Protocol (the same as for the GBMs, PLAN §3):
    1. for each hidden size in HIDDEN_SIZES: fit on TRAIN with early stopping
       on VALIDATION; the lowest validation RMSE wins -> validation scores;
    2. refit on TRAIN + VALIDATION with that hidden size for the same number
       of epochs -> final model, scored on TEST once.

Outputs:
    models/lstm_<horizon>.pt                    final model (git-ignored)
    data/processed/pred_lstm_<horizon>.parquet  test predictions (git-ignored)
    results/tables/05_lstm_results.csv          val + test, all hours + daytime, CIs
    results/tables/05_lstm_training.csv         RMSE per epoch of every run
    results/tables/05_lstm_per_station.csv      test, daytime
    results/figures/05_lstm_curves.png, 05_models_nmae.png, 05_models_skill.png
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src import evaluate
from src.models import lstm
from src.train_gbm import merge_csv
from src.utils import set_seed

LSTM_HORIZONS = ["15min", "1h"]
HIDDEN_SIZES = [32, 64]
RESULTS = evaluate.TABLES / "05_lstm_results.csv"
TRAINING = evaluate.TABLES / "05_lstm_training.csv"
PER_STATION = evaluate.TABLES / "05_lstm_per_station.csv"


def run_horizon(horizon):
    table = evaluate.load_table(horizon)
    bench = evaluate.benchmark(table, horizon)
    rated = evaluate.rated_kW(table)

    # 1. small search on train -> validation
    runs, histories = {}, []
    for hidden in HIDDEN_SIZES:
        print(f"   hidden {hidden}: train -> val")
        start = time.time()
        net, scaler, history = lstm.train(table, horizon, hidden, ["train"], "val")
        history["hidden"], history["run"] = hidden, "train (early stopping on val)"
        history["seconds"] = time.time() - start
        histories.append(history)
        runs[hidden] = (net, scaler, history)

    best_val = {h: runs[h][2]["val_rmse"].min() for h in runs}
    hidden = min(best_val, key=best_val.get)
    net, scaler, history = runs[hidden]
    epochs = int(history["val_rmse"].idxmin()) + 1
    print(f"   chosen: hidden {hidden}, {epochs} epochs, val RMSE {best_val[hidden]:.5f}")

    is_val = table["split"] == "val"
    pred_val = lstm.predict(net, scaler, table, horizon, is_val)
    rows = evaluate.score_model(table, pred_val, bench, rated, horizon, "lstm", "val")

    # 2. final refit on train + validation, fixed epochs -> test (once)
    print(f"   final: train + val, {epochs} epochs")
    start = time.time()
    final, final_scaler, final_history = lstm.train(
        table, horizon, hidden, ["train", "val"], epochs=epochs)
    seconds = time.time() - start
    final_history["hidden"], final_history["run"] = hidden, "final (train + val)"
    final_history["seconds"] = seconds
    histories.append(final_history)

    is_test = table["split"] == "test"
    pred = lstm.predict(final, final_scaler, table, horizon, is_test)
    test_rows = evaluate.score_model(table, pred, bench, rated, horizon, "lstm", "test")
    for row in test_rows:
        row["fit_seconds"] = round(seconds, 1)
    rows += test_rows
    stations = evaluate.score_per_station(table, pred, bench, rated, horizon, "lstm")

    info = {"horizon": horizon, "hidden": hidden, "epochs": epochs,
            "window": lstm.WINDOW[horizon], "fit_seconds": round(seconds, 1)}
    evaluate.MODELS.mkdir(exist_ok=True)
    lstm.save(final, final_scaler, evaluate.MODELS / f"lstm_{horizon}.pt", info)

    keep = ["station", "role", "target", "usable", "sun_elevation"]
    preds = table.loc[is_test, keep].copy()
    preds["benchmark"] = bench[is_test]
    preds["lstm"] = pred[is_test]
    preds.to_parquet(evaluate.PROCESSED / f"pred_lstm_{horizon}.parquet")

    day = test_rows[1]
    print(f"   lstm: final fit {seconds:.0f} s, test daytime nMAE {day['nmae_pct']:.2f} %, "
          f"skill {day['skill']:.3f}")

    training = pd.concat(histories)
    training.insert(0, "horizon", horizon)
    return rows, training, stations


def plot_curves():
    """Train and validation RMSE per epoch of the search runs."""
    training = pd.read_csv(TRAINING)
    training = training[training["run"].str.startswith("train")]
    horizons = [h for h in LSTM_HORIZONS if h in set(training["horizon"])]

    fig, axes = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 3.6), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    colors = {32: "#2a78d6", 64: "#d6602a"}
    for ax, horizon in zip(axes[0], horizons):
        for hidden, d in training[training["horizon"] == horizon].groupby("hidden"):
            ax.plot(d["epoch"], d["train_rmse"], color=colors.get(hidden, "#52514e"),
                    ls="--", label=f"hidden {hidden}: train")
            ax.plot(d["epoch"], d["val_rmse"], color=colors.get(hidden, "#52514e"),
                    label=f"hidden {hidden}: validation")
        ax.set_facecolor("#fcfcfb")
        ax.set_title(horizon)
        ax.set_xlabel("epoch")
        ax.set_ylabel("RMSE (power / rated), daytime")
        ax.grid(color="#e6e5e1", lw=0.8)
        ax.legend(fontsize=8, frameon=False)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.suptitle("LSTM training curves (early stopping on validation)", fontsize=11)
    plt.tight_layout()
    plt.savefig(evaluate.FIGURES / "05_lstm_curves.png", dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_models():
    """All models of Stages 3-5 next to each other, test set, daytime."""
    tables = ["03_baselines.csv", "04_gbm_results.csv", "05_lstm_results.csv"]
    both = pd.concat([pd.read_csv(evaluate.TABLES / t) for t in tables])
    title = "2023 test set, daytime, seen stations (lines = 95 % CI)"
    evaluate.plot_test_daytime(both, evaluate.FIGURES / "05_models_nmae.png",
                               "nMAE: " + title + ", lower is better", horizons=LSTM_HORIZONS)
    evaluate.plot_test_daytime(both, evaluate.FIGURES / "05_models_skill.png",
                               "Skill: " + title + ", higher is better",
                               metric="skill", label="skill = 1 - RMSE / RMSE benchmark",
                               horizons=LSTM_HORIZONS)


def main(horizons):
    set_seed()
    torch.set_num_threads(4)
    all_rows, all_training, all_stations = [], [], []
    for horizon in horizons:
        print(f"== {horizon}")
        rows, training, stations = run_horizon(horizon)
        all_rows += rows
        all_training.append(training)
        all_stations.append(stations)

    results = pd.DataFrame(all_rows)
    first = evaluate.FIRST_COLUMNS
    results = results[first + [c for c in results.columns if c not in first]]
    results["commit"] = evaluate.git_commit()
    results["date"] = pd.Timestamp.today().date().isoformat()
    merge_csv(results, RESULTS, horizons)
    merge_csv(pd.concat(all_training), TRAINING, horizons)
    merge_csv(pd.concat(all_stations), PER_STATION, horizons)
    plot_curves()
    plot_models()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate the LSTM.")
    parser.add_argument("--horizon", choices=LSTM_HORIZONS)
    args = parser.parse_args()
    main([args.horizon] if args.horizon else LSTM_HORIZONS)
