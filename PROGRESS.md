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
| 3 | Stage 6: efficiency benchmark, ONNX, int8 quantization | DONE |
| 4 | Stage 7: SHAP explainability and stability | DONE |
| 5 | Stage 8: residual-based fault flagging | DONE |
| 6 | Stage 9: unseen stations, grouped K-fold, Kaggle transfer check | DONE |
| 7 | Stage 10: Streamlit dashboard + demo-bundle script | DONE (screenshots left for the user) |
| 8 | Stage 11 (repo part only): README reproduce steps, pinned requirements, clean run | DONE |

Not doable by the agent (left for the user): report template fill-in, presentation video,
licence check in `DATA_SOURCES.md`, final tag `v1.0`.

## Checkpoint (2026-11-17): autonomous run complete

All eight milestones are done and pushed. Nothing is running.

Left for the user (cannot be done by the agent):
1. Dashboard screenshots -> `results/figures/10_dashboard_*.png` (no browser in the agent environment).
2. Report from the template (not shared yet); every number is in `results/tables/` and the `PLAN.md` Change log.
3. Presentation video; viva answers from each stage's "Viva question".
4. Licence check in `DATA_SOURCES.md` (Kaggle licence, Dryad licence).
5. Final tag `v1.0`.

To use the dashboard: `python -m app.build_demo` (already built in this working copy), then
`streamlit run app/streamlit_app.py`.
