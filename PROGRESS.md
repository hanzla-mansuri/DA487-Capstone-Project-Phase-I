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
| 3 | Stage 6: efficiency benchmark, ONNX, int8 quantization | IN PROGRESS (script written, first full run not yet checked) |
| 4 | Stage 7: SHAP explainability and stability | DONE |
| 5 | Stage 8: residual-based fault flagging | DONE |
| 6 | Stage 9: unseen stations, grouped K-fold, Kaggle transfer check | DONE |
| 7 | Stage 10: Streamlit dashboard + demo-bundle script | IN PROGRESS (code written, never run) |
| 8 | Stage 11 (repo part only): README reproduce steps, pinned requirements, clean run | NOT STARTED |

Not doable by the agent (left for the user): report template fill-in, presentation video,
licence check in `DATA_SOURCES.md`, final tag `v1.0`.

## Checkpoint (2026-11-16, run paused: usage limit reached)

- **Done and pushed:** Stages 4, 5, 7, 8, 9 (milestones 1, 2, 4, 5, 6), each with tables, figures,
  a notebook and a `PLAN.md` Change log entry.
- **Next, in this order (run heavy jobs one at a time):**
  1. Stage 6: `python -m src.benchmark`. A first attempt failed on a bug that is now fixed; the
     re-run was started but its output has NOT been checked. Check `results/tables/06_efficiency.csv`
     (ONNX exports may be recorded as failed in the `note` column), then build the notebook
     (`python notebooks/_make_notebooks.py 06` + `jupyter nbconvert --execute --inplace`), tick
     Stage 6 in `PLAN.md`, add a Change log entry, commit, push.
  2. Stage 10: `python -m app.build_demo`, then `streamlit run app/streamlit_app.py`. The app code
     has never been executed: expect small fixes. Add a smoke test (`streamlit.testing.v1.AppTest`),
     save screenshots as `results/figures/10_dashboard_*.png`, tick `PLAN.md`, commit, push.
  3. Stage 11 (repo part): strip notebook outputs if wanted, re-check README steps, final test run.
- `notebooks/_make_notebooks.py` writes the Stage 4-9 notebooks (run from the repo root).
