"""Simple missing-data and gap statistics for a time series on a regular grid.

A value counts as *missing* if its timestamp is absent from the file or if the value is NaN.
Nothing is filled or changed here; this only measures.
"""
import pandas as pd


def missing_mask(series, start, end, freq):
    """True for every grid step (start..end inclusive, step `freq`) with no usable value."""
    series = series[~series.index.duplicated()]
    grid = pd.date_range(start, end, freq=freq)
    return series.reindex(grid).isna()


def gap_stats(mask, freq):
    """Summary of a missing mask: totals, number of gaps and the longest gap.

    A gap is a run of consecutive missing steps.
    """
    # label each run of equal values, then keep the runs that are missing
    run_id = (mask != mask.shift()).cumsum()
    gap_lengths = mask.groupby(run_id).sum()
    gap_lengths = gap_lengths[gap_lengths > 0]

    step_hours = pd.Timedelta(freq).total_seconds() / 3600
    longest = int(gap_lengths.max()) if len(gap_lengths) else 0
    return {
        "expected_steps": len(mask),
        "missing_steps": int(mask.sum()),
        "pct_missing": round(100 * mask.mean(), 2),
        "n_gaps": len(gap_lengths),
        "longest_gap_steps": longest,
        "longest_gap_hours": round(longest * step_hours, 2),
    }


def longest_gaps(mask, freq, top=3):
    """Start time and length (hours) of the `top` longest gaps."""
    run_id = (mask != mask.shift()).cumsum()
    lengths = mask.groupby(run_id).sum()
    starts = mask.index.to_series().groupby(run_id).first()
    gaps = pd.DataFrame({"start": starts, "steps": lengths})
    gaps = gaps[gaps["steps"] > 0].sort_values("steps", ascending=False).head(top)
    gaps["hours"] = (gaps["steps"] * pd.Timedelta(freq).total_seconds() / 3600).round(2)
    return gaps.reset_index(drop=True)
