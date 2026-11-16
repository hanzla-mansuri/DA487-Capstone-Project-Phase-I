# DECISIONS: choices made during the autonomous run

Short log of technical choices made without asking (see `PROGRESS.md` for the run rules).
Anything that changes the plan itself is also in the `PLAN.md` Change log.

| Date | Decision | Why |
|---|---|---|
| 2026-11-16 | `PLAN.md` checkboxes stay the task tracker; `PROGRESS.md` only lists milestones and the resume checkpoint. | One source of truth; avoids two task lists drifting apart. |
| 2026-11-16 | `WORKLOG.md` is not updated. | The user dropped it earlier. |
| 2026-11-16 | Dashboard lives at `app/streamlit_app.py`, not `app.py`. | `PLAN.md` Stage 10 names this path. |
| 2026-11-16 | No demo data or models are committed; `app/build_demo.py` rebuilds the bundle from raw data. | Dataset licences are unverified ("skip licences"). |
| 2026-11-16 | Untracked `test.py` in the repo root is never staged. | Not part of the project; commits add files by name. |
