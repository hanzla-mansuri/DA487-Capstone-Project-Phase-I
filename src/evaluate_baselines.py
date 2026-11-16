"""Evaluate every Stage 3 baseline on validation and test.

    python -m src.evaluate_baselines

For each horizon: all persistence baselines and ridge, scored against the
skill-score benchmark of PLAN §3 (last value for 15 min and 1 h, same hour
on D-1 for day-ahead), on SEEN stations and usable targets only.
Two variants: all hours, and daytime only (sun >= 5 degrees).

Ridge follows the "final training" rule (PLAN §3): alpha is chosen on
validation with a model fitted on train; the validation scores come from
that model. For test, the model is refit on train + validation with the same
alpha, and test is scored once.

Outputs:
    results/tables/03_baselines.csv
    results/figures/03_baselines_nmae.png
"""
import subprocess
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src import metrics
from src.models import baselines, ridge

REPO = Path(__file__).resolve().parents[1]
INVENTORY = REPO / "results" / "tables" / "01_station_inventory.csv"
TABLE = REPO / "results" / "tables" / "03_baselines.csv"
FIGURE = REPO / "results" / "figures" / "03_baselines_nmae.png"

HORIZONS = ["15min", "1h", "dayahead"]
BENCHMARK = {"15min": "last_value", "1h": "last_value", "dayahead": "same_hour_yesterday"}


def git_commit():
    """Short hash of the current commit (PLAN §6: every results table records it)."""
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=REPO, capture_output=True, text=True)
    return result.stdout.strip()


def all_predictions(table):
    """Predictions of every baseline. Ridge gets one column per split."""
    preds = {}
    for name, fn in baselines.BASELINES[table.attrs["horizon"]].items():
        preds[name] = fn(table)

    alpha, _ = ridge.choose_alpha(table)
    on_train = ridge.fit(table, alpha, splits=["train"])
    on_train_val = ridge.fit(table, alpha, splits=["train", "val"])
    preds["ridge"] = ridge.predict(on_train, table)
    preds["ridge_final"] = ridge.predict(on_train_val, table)
    print(f"   ridge alpha = {alpha}")
    return preds


def evaluate_horizon(horizon, rated_kW_of):
    table = pd.read_parquet(REPO / "data" / "processed" / f"features_{horizon}.parquet")
    table.attrs["horizon"] = horizon
    preds = all_predictions(table)
    bench = preds[BENCHMARK[horizon]]
    rated_kW = table["station"].map(rated_kW_of)

    rows = []
    for split in ["val", "test"]:
        for hours in ["all", "daytime"]:
            mask = (table["role"] == "seen") & table["usable"] & (table["split"] == split)
            if hours == "daytime":
                mask = mask & (table["sun_elevation"] >= baselines.MIN_ELEVATION)

            for model in preds:
                # ridge: train-only model on validation, train+val model on test
                if model == "ridge" and split == "test":
                    continue
                if model == "ridge_final" and split == "val":
                    continue

                args = (table["target"][mask], preds[model][mask], bench[mask], rated_kW[mask])
                result = metrics.score(*args)
                result["mean_station_skill"] = metrics.mean_station_skill(
                    *args, table["station"][mask])
                result.update({"horizon": horizon, "split": split, "hours": hours,
                               "model": model.replace("_final", ""),
                               "benchmark": BENCHMARK[horizon]})
                rows.append(result)
    return rows


def plot_test_daytime(results):
    """nMAE with 95 % CI per model, test set, daytime; one panel per horizon."""
    data = results[(results["split"] == "test") & (results["hours"] == "daytime")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    fig.patch.set_facecolor("#fcfcfb")

    for ax, horizon in zip(axes, HORIZONS):
        d = data[data["horizon"] == horizon].sort_values("nmae_pct", ascending=False)
        labels = [m + (" (benchmark)" if m == BENCHMARK[horizon] else "") for m in d["model"]]
        errors = [d["nmae_pct"] - d["nmae_pct_low"], d["nmae_pct_high"] - d["nmae_pct"]]

        ax.set_facecolor("#fcfcfb")
        ax.barh(labels, d["nmae_pct"], height=0.55, color="#2a78d6")
        ax.errorbar(d["nmae_pct"], labels, xerr=errors, fmt="none", ecolor="#52514e", capsize=3, lw=1)
        for y, v, hi in zip(range(len(d)), d["nmae_pct"], d["nmae_pct_high"]):
            ax.text(hi + 0.2, y, f"{v:.1f}", va="center", fontsize=9, color="#0b0b0b")

        ax.set_title(horizon, color="#0b0b0b")
        ax.set_xlabel("nMAE (% of rated)", color="#52514e", fontsize=9)
        ax.set_xlim(0, data["nmae_pct_high"].max() * 1.15)
        ax.tick_params(colors="#52514e", labelsize=9)
        ax.grid(axis="x", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)

    fig.suptitle("Baselines on the 2023 test set, daytime, seen stations "
                 "(lower is better; lines = 95 % CI)", color="#0b0b0b", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURE, dpi=130, facecolor=fig.get_facecolor())
    print(f"saved {FIGURE}")


def main():
    inventory = pd.read_csv(INVENTORY)
    rated_kW_of = inventory.set_index("station")["rated_value"]   # kW for all stations

    rows = []
    for horizon in HORIZONS:
        print(f"== {horizon}")
        rows += evaluate_horizon(horizon, rated_kW_of)

    results = pd.DataFrame(rows)
    first = ["horizon", "split", "hours", "model", "benchmark", "n_rows", "n_days"]
    results = results[first + [c for c in results.columns if c not in first]]
    results["commit"] = git_commit()
    results["date"] = date.today().isoformat()
    results.to_csv(TABLE, index=False, float_format="%.5f")
    print(f"saved {TABLE}: {len(results)} rows")

    plot_test_daytime(results)


if __name__ == "__main__":
    main()
