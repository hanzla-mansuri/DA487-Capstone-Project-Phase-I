"""Shared evaluation helpers for Stages 4-9.

Every model is scored the same way as the Stage 3 baselines
(`src/evaluate_baselines.py`): usable targets only, all hours and daytime,
against the skill-score benchmark of PLAN §3, with block-bootstrap CIs.
"""
import subprocess
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import metrics
from src.models import baselines
from src.splits import season

REPO = Path(__file__).resolve().parents[1]
PROCESSED = REPO / "data" / "processed"
TABLES = REPO / "results" / "tables"
FIGURES = REPO / "results" / "figures"
MODELS = REPO / "models"
INVENTORY = TABLES / "01_station_inventory.csv"

HORIZONS = ["15min", "1h", "dayahead"]
BENCHMARK = {"15min": "last_value", "1h": "last_value", "dayahead": "same_hour_yesterday"}
FIRST_COLUMNS = ["horizon", "split", "hours", "model", "benchmark", "n_rows", "n_days"]


def git_commit():
    """Short hash of the current commit (PLAN §6: every results table records it)."""
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                            cwd=REPO, capture_output=True, text=True)
    return result.stdout.strip()


def load_table(horizon):
    """The Stage 2 feature table of one horizon."""
    return pd.read_parquet(PROCESSED / f"features_{horizon}.parquet")


def rated_kW(table):
    """Rated power (kW) of the station of every row."""
    inventory = pd.read_csv(INVENTORY)
    return table["station"].map(inventory.set_index("station")["rated_value"])


def benchmark(table, horizon):
    """Prediction of the skill-score benchmark for every row."""
    return baselines.BASELINES[horizon][BENCHMARK[horizon]](table)


def eval_mask(table, split, hours, role="seen"):
    """Rows that count in an evaluation."""
    mask = (table["role"] == role) & table["usable"] & (table["split"] == split)
    if hours == "daytime":
        mask = mask & (table["sun_elevation"] >= baselines.MIN_ELEVATION)
    return mask


def score_model(table, pred, bench, rated, horizon, model, split, role="seen"):
    """Two result rows (all hours, daytime) for one model on one split."""
    rows = []
    for hours in ["all", "daytime"]:
        mask = eval_mask(table, split, hours, role)
        args = (table["target"][mask], pred[mask], bench[mask], rated[mask])
        result = metrics.score(*args)
        result["mean_station_skill"] = metrics.mean_station_skill(*args, table["station"][mask])
        result.update({"horizon": horizon, "split": split, "hours": hours,
                       "model": model, "benchmark": BENCHMARK[horizon]})
        rows.append(result)
    return rows


def score_per_station(table, pred, bench, rated, horizon, model, role="seen"):
    """Test-set daytime metrics of every station (no CIs)."""
    mask = eval_mask(table, "test", "daytime", role)
    out = metrics.per_station(table["target"][mask], pred[mask], bench[mask],
                              rated[mask], table["station"][mask]).reset_index()
    out.insert(0, "model", model)
    out.insert(0, "horizon", horizon)
    return out


def score_per_season(table, pred, bench, rated, horizon, model, role="seen"):
    """Test-set daytime metrics of every season (no CIs)."""
    mask = eval_mask(table, "test", "daytime", role)
    seasons = season(table.index)
    rows = []
    for name in ["winter", "spring", "summer", "autumn"]:
        m = mask & (seasons == name).to_numpy()
        result = metrics.score(table["target"][m], pred[m], bench[m], rated[m], n_boot=0)
        result.update({"horizon": horizon, "model": model, "season": name})
        rows.append(result)
    return pd.DataFrame(rows)


def save_results(rows, path):
    """Write result rows in the Stage 3 table format, with commit and date."""
    results = pd.DataFrame(rows)
    first = [c for c in FIRST_COLUMNS if c in results.columns]
    results = results[first + [c for c in results.columns if c not in first]]
    results["commit"] = git_commit()
    results["date"] = date.today().isoformat()
    results.to_csv(path, index=False, float_format="%.5f")
    print(f"saved {path}: {len(results)} rows")
    return results


def plot_test_daytime(results, path, title, metric="nmae_pct",
                      label="nMAE (% of rated)", horizons=HORIZONS):
    """One metric with its 95 % CI per model, test set, daytime; one panel
    per horizon."""
    data = results[(results["split"] == "test") & (results["hours"] == "daytime")]
    fig, axes = plt.subplots(1, len(horizons), figsize=(4.2 * len(horizons), 3.6), squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")

    for ax, horizon in zip(axes[0], horizons):
        d = data[data["horizon"] == horizon].sort_values(metric, ascending=(metric == "skill"))
        labels = [m + (" (benchmark)" if m == BENCHMARK[horizon] else "") for m in d["model"]]
        errors = [d[metric] - d[metric + "_low"], d[metric + "_high"] - d[metric]]

        ax.set_facecolor("#fcfcfb")
        ax.barh(labels, d[metric], height=0.55, color="#2a78d6")
        ax.errorbar(d[metric], labels, xerr=errors, fmt="none", ecolor="#52514e", capsize=3, lw=1)
        span = data[metric + "_high"].max()
        for y, v, hi in zip(range(len(d)), d[metric], d[metric + "_high"]):
            text = f"{v:.2f}" if metric == "skill" else f"{v:.1f}"
            ax.text(hi + 0.02 * span, y, text, va="center", fontsize=9, color="#0b0b0b")

        ax.set_title(horizon, color="#0b0b0b")
        ax.set_xlabel(label, color="#52514e", fontsize=9)
        ax.set_xlim(min(0, data[metric + "_low"].min()), span * 1.18)
        ax.tick_params(colors="#52514e", labelsize=9)
        ax.grid(axis="x", color="#e6e5e1", lw=0.8)
        ax.set_axisbelow(True)
        for side in ["top", "right", "left"]:
            ax.spines[side].set_visible(False)

    fig.suptitle(title, color="#0b0b0b", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {path}")
