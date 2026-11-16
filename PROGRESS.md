# PROGRESS: autonomous run (started 2026-11-16)

`PLAN.md` stays the source of truth for tasks (checkboxes) and decisions (Change log).
This file is the milestone list and the **resume checkpoint** for the autonomous run:
any model or person can continue from the "Checkpoint" section at the bottom.

Rules for this run (user reply 2026-11-16: "autonomous, push is fine, skip licences"):
- No stage-approval stops; commit and push after every milestone.
- Still in force: no force push / rebase / amend on pushed commits, never commit
  `data/` or `models/`, never fabricate numbers, basic notebook code.
- No demo data is committed (licences unverified); the dashboard bundle is rebuilt by a script.

## Milestones

| # | Milestone (PLAN stage) | Status |
|---|---|---|
| 1 | Stage 4: tune 15 min + day-ahead, final refit, test evaluation, saved models | IN PROGRESS |
| 2 | Stage 5: LSTM (15 min, 1 h) | NOT STARTED |
| 3 | Stage 6: efficiency benchmark, ONNX, int8 quantization | NOT STARTED |
| 4 | Stage 7: SHAP explainability and stability | NOT STARTED |
| 5 | Stage 8: residual-based fault flagging | NOT STARTED |
| 6 | Stage 9: unseen stations, grouped K-fold, Kaggle transfer check | NOT STARTED |
| 7 | Stage 10: Streamlit dashboard + demo-bundle script | NOT STARTED |
| 8 | Stage 11 (repo part only): README reproduce steps, pinned requirements, clean run | NOT STARTED |

Not doable by the agent (left for the user): report template fill-in, presentation video,
licence check in `DATA_SOURCES.md`, final tag `v1.0`.

## Checkpoint

- **Now:** milestone 1. 1-h horizon is tuned (`configs/gbm_*_1h.yaml`).
- **Next command:** `python -m src.tune_gbm --horizon dayahead --trials 20`, then `--horizon 15min`.
- **Then:** write `src/train_gbm.py` (refit on train + val, test once, save models, `04_gbm_results.csv`).
