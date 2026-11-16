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
| 1 | Stage 4: tune 15 min + day-ahead, final refit, test evaluation, saved models (+ irradiance sensor fix) | DONE |
| 2 | Stage 5: LSTM (15 min, 1 h) | DONE |
| 3 | Stage 6: efficiency benchmark, ONNX, int8 quantization | NOT STARTED |
| 4 | Stage 7: SHAP explainability and stability | NOT STARTED |
| 5 | Stage 8: residual-based fault flagging | NOT STARTED |
| 6 | Stage 9: unseen stations, grouped K-fold, Kaggle transfer check | NOT STARTED |
| 7 | Stage 10: Streamlit dashboard + demo-bundle script | NOT STARTED |
| 8 | Stage 11 (repo part only): README reproduce steps, pinned requirements, clean run | NOT STARTED |

Not doable by the agent (left for the user): report template fill-in, presentation video,
licence check in `DATA_SOURCES.md`, final tag `v1.0`.

## Checkpoint

- **Done:** milestones 1-2. Final GBMs and LSTMs are in `models/`, results in `results/tables/04_*`, `05_*`.
- **Now:** milestones 3-6 run as one chain (faults, benchmark, explain, generalise).
- **Then, in order (all scripts are written; run one at a time):**
  `python -m src.benchmark` -> `python -m src.explain` -> `python -m src.faults` ->
  `python -m src.generalise` -> `python -m app.build_demo` -> dashboard smoke test.
- After each: tick `PLAN.md`, add a Change log entry with the key numbers, build the stage
  notebook, commit, push.
