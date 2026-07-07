"""Single source of truth for per-series P90 outlier capping on historical crowd.

The dataset notebook saves RAW data; this function is applied on every historical
load, so the raw files stay usable uncapped in other situations. It clamps the
real crowd column above ``k * P{percentile}`` per series (daytime positive obs
only), which strips spurious Bayesian-counter spikes while preserving genuine
peaks. Only the ground-truth crowd column is touched — model predictions must
never be passed here.

Import as a sibling (`from crowd_outliers import cap_outliers`) from the
prediction scripts; the new_training_pipeline adds this dir to sys.path.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_K = 1.5
DEFAULT_PERCENTILE = 90
DAY_START, DAY_END = 8, 20


def series_p90_thresholds(df: pd.DataFrame, y_col: str = "y",
                          k: float = DEFAULT_K, percentile: int = DEFAULT_PERCENTILE,
                          hour_col: str = "hour",
                          day_start: int = DAY_START, day_end: int = DAY_END) -> pd.Series:
    """Per-series cap threshold ``k * P{percentile}`` on daytime positive crowd."""
    hours = df[hour_col] if hour_col in df.columns else pd.to_datetime(df["ds"]).dt.hour
    y = pd.to_numeric(df[y_col], errors="coerce")
    day = df[(hours >= day_start) & (hours <= day_end) & (y > 0)]
    return k * day.groupby("unique_id")[y_col].quantile(percentile / 100.0)


def cap_outliers(df: pd.DataFrame, y_col: str = "y",
                 k: float | None = DEFAULT_K, percentile: int = DEFAULT_PERCENTILE,
                 hour_col: str = "hour", day_start: int = DAY_START, day_end: int = DAY_END,
                 thresholds: pd.Series | None = None, verbose: bool = True) -> tuple[pd.DataFrame, int]:
    """Clamp the real crowd ``y_col`` above ``k * P{percentile}`` per series.

    Mutates and returns the same frame (caller owns it); returns ``(df, n_capped)``.
    ``k`` of 0/None is a no-op. Pass ``thresholds`` (a per-series Series) to share
    one ceiling across split files — series missing from it fall back to their own
    P90. Never pass a prediction column as ``y_col``.
    """
    if not k or k <= 0:
        return df, 0
    local = series_p90_thresholds(df, y_col, k, percentile, hour_col, day_start, day_end)
    thr = thresholds.combine_first(local) if thresholds is not None else local
    y = pd.to_numeric(df[y_col], errors="coerce")
    thr_row = df["unique_id"].map(thr)
    mask = thr_row.notna() & (y > thr_row)
    df.loc[mask, y_col] = thr_row[mask]
    n = int(mask.sum())
    if verbose:
        n_pos = int((y > 0).sum())
        print(f"[cap] {y_col} > {k}*P{percentile} per series: clamped {n:,} rows "
              f"({100 * n / max(n_pos, 1):.2f}% of positive) across "
              f"{df.loc[mask, 'unique_id'].nunique()} series")
    return df, n
