"""Tune LightGBM and XGBoost for one horizon (Stage 4).

    python -m src.tune_gbm --horizon 1h --trials 20

For each model: `trials` random settings, each fitted on train with early
stopping on validation, scored by validation RMSE (seen stations, usable
daytime rows, power/rated). The test set is not touched.

Outputs:
    results/tables/04_tuning_log.csv        every setting tried (all horizons)
    configs/gbm_<model>_<horizon>.yaml      the best setting + number of trees
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models import gbm
from src.models.ridge import fit_rows

REPO = Path(__file__).resolve().parents[1]
LOG = REPO / "results" / "tables" / "04_tuning_log.csv"
CONFIGS = REPO / "configs"


def validation_rmse(model, table):
    rows = fit_rows(table, ["val"])
    err = gbm.predict(model, table)[rows] - table.loc[rows, "target"]
    return float(np.sqrt((err ** 2).mean()))


def tune(table, horizon, model_name, trials):
    rows = []
    for i, params in enumerate(gbm.sample_settings(model_name, trials), start=1):
        start = time.time()
        model, n_trees = gbm.fit_early_stopping(table, model_name, params)
        rmse = validation_rmse(model, table)
        seconds = time.time() - start
        print(f"   {model_name} {i:2d}/{trials}: val RMSE {rmse:.5f}, trees {n_trees:4d}, {seconds:5.0f} s  {params}")
        rows.append({"horizon": horizon, "model": model_name, "trial": i,
                     "val_rmse": rmse, "n_trees": n_trees, "fit_seconds": round(seconds, 1),
                     "params": json.dumps(params)})
    return pd.DataFrame(rows)


def save_config(log, horizon, model_name):
    """Write the best setting of one model and horizon to configs/."""
    mine = log[(log["horizon"] == horizon) & (log["model"] == model_name)]
    best = mine.loc[mine["val_rmse"].idxmin()]
    config = {
        "model": model_name,
        "horizon": horizon,
        "params": json.loads(best["params"]),
        "n_trees": int(best["n_trees"]),
        "val_rmse": float(best["val_rmse"]),
        "trial": int(best["trial"]),
        "selected_by": "lowest validation RMSE (seen, usable, daytime); early stopping on validation",
    }
    CONFIGS.mkdir(exist_ok=True)
    path = CONFIGS / f"gbm_{model_name}_{horizon}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"   best {model_name}: trial {config['trial']}, val RMSE {config['val_rmse']:.5f}, "
          f"{config['n_trees']} trees -> {path.name}")


def main(horizon, trials):
    table = pd.read_parquet(REPO / "data" / "processed" / f"features_{horizon}.parquet")
    print(f"== {horizon}")

    new = pd.concat([tune(table, horizon, m, trials) for m in ["lgbm", "xgb"]])

    # keep the other horizons' rows in the log, replace this horizon's
    if LOG.exists():
        old = pd.read_csv(LOG)
        new = pd.concat([old[old["horizon"] != horizon], new])
    new.to_csv(LOG, index=False)

    for m in ["lgbm", "xgb"]:
        save_config(new, horizon, m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune LightGBM and XGBoost for one horizon.")
    parser.add_argument("--horizon", choices=["15min", "1h", "dayahead"], required=True)
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()
    main(args.horizon, args.trials)
