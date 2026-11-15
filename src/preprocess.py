"""Cleaning steps for the site-level power files (Stage 2).

Each step is a small function so it can be explained and tested on its own.
"""
import pandas as pd


def resolve_duplicates(df):
    """Sort by time and make every timestamp appear once.

    Rule (decided from the Stage 2 duplicate check):
    - if all copies of a timestamp have the same power, keep one copy;
    - if the copies disagree, we cannot know which is right, so the
      timestamp is kept but its values are set to missing (NaN).

    Returns the cleaned table and a small dict of counts.
    """
    df = df.sort_index(kind="stable")
    is_dup = df.index.duplicated(keep=False)

    # how many different power values does each duplicated timestamp have?
    n_values = df[is_dup].groupby(level=0)["power(W)"].nunique(dropna=False)
    disagree = n_values[n_values > 1].index

    out = df[~df.index.duplicated(keep="first")].copy()
    out.loc[disagree, :] = float("nan")

    counts = {
        "rows_in": len(df),
        "duplicate_timestamps": int(n_values.shape[0]),
        "disagreeing_timestamps": int(len(disagree)),
        "rows_out": len(out),
    }
    return out, counts
