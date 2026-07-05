#!/usr/bin/env python3
"""
Methodologically-rigorous retraining + evaluation for TFT, LSTM and XGBoost.

Methodology (read before running)
=================================
1. Two validation protocols, both selectable via --protocol:
   (A) cross_year  : train = 2022 rows only          ; test = frozen 2025 window.
   (B) expanding   : train = ALL rows strictly before ; test = same window.
                     the test-window start (2022 + 2025-up-to-cutoff)
   Each (model, horizon) is trained ONCE, then scored by rolling forecast
   origins (weekly Mondays) across the test window: at every origin the model
   sees only real history ending before that origin (no future leakage).
2. Leakage guards (each enforced + commented in code):
   - per-series P90 capacity uses TRAIN rows only, floored cap = max(P90, 1);
   - HP tuning objective is an INNER validation = a temporal tail of TRAIN,
     never the test window;
   - in expanding mode the training history strictly precedes each test origin.
3. HP search: reduced informed spaces (see audit §7), equal Optuna budget for
   every family (TPE, seed 42, ~15 startup trials, --trials each).
4. Multi-seed: the best config is refit over --seeds seeds; metrics reported as
   mean +/- SD per (model, horizon, protocol), XGB included.
5. Identical test rows: the three families are inner-joined on
   (unique_id, ds) — the absolute target timestamp — so metrics AND the DM test
   use the EXACT same (series, lead) rows for every model. (XGB has no forecast
   origin, so the join cannot key on issue_date.)
6. Metrics: primary = per-series relMAE = MAE_series / P90_series, MEAN over
   series (the thesis compute_rel_mae). Cross-checks: ratio-of-means (pooled),
   mean-normalised, raw MAE (users), per-beach R2. Buckets: overall / season
   (months 4-9) / summer (6-8).
7. Cross-family Diebold-Mariano on the identical rows, per horizon: loss =
   absolute error, Newey-West HAC variance within (series) blocks +
   Harvey-Leybourne-Newbold correction + Wilcoxon signed-rank.

NF plumbing (model construction, fit, rolling predict) is copied from the
proven cross-year scripts that already work with neuralforecast 3.1.5.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# OpenMP guard — prevents XGBoost segfaulting after PyTorch spawns threads.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

# Canonical per-series P90 outlier cap (single source of truth, lives with the
# data in beachcamweb so every historical-load site reuses the same function).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "beachcamweb" / "apps" / "prediction" / "scripts"))
from crowd_outliers import cap_outliers as _cap_outliers  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Shared protocol constants (copied from run_cross_year_unified.py)
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
HOURS_PER_DAY = 13  # operational window 8:00-20:00 inclusive = 13 buckets/day
HORIZON_HOURS = {3: 39, 10: 130, 15: 195}  # days -> daytime steps at 13 buckets/day
SEASON_MONTHS = {4, 5, 6, 7, 8, 9}
SUMMER_MONTHS = {6, 7, 8}

NF_TRIAL_MAX_STEPS = 200   # quick training during HP search
NF_FINAL_MAX_STEPS = 500   # full training for the refit
NF_BATCH_SIZE = 64         # NN minibatch; lower it (e.g. 8/16) to fit 15d on small RAM

# Outlier cap on the REAL crowd y: clamp values above k*P90 (per series) to the
# threshold. Removes spurious CV-counter spikes (~1.3% of positive rows at k=1.5)
# while preserving genuine peaks. Applied to ground-truth y only — never to
# model predictions. Set to 0/None to disable.
DEFAULT_CAP_K = 1.5
CAP_PERCENTILE = 90

# TFT/LSTM three-input typology (NeuralForecast) — same schema both families.
STAT_EXOG = ["stat_mean_y", "stat_cv"]
TFT_FUTR = ["hour", "day_of_week", "month", "is_weekend",
            "om_temperature_2m", "om_apparent_temperature", "om_cloud_cover"]
TFT_HIST = ["om_shortwave_radiation", "om_vapour_pressure_deficit"]

# Castelle 2025 XGB schema.
CASTELLE_WEATHER = [
    "om_temperature_2m", "om_precipitation", "om_wind_speed_10m",
    "om_wind_direction_10m", "om_shortwave_radiation",
]
CALENDAR = ["hour", "day_of_week", "month", "is_weekend"]

# Default panel: the clean backup panel (audit "v2", minmax-scaler-favoured,
# 22 series). run_cross_year_unified.py's literal argparse default points at the
# 210526 panel instead; the prompt and audit call the *backup* the clean panel,
# so it is the default here. Override with --panel for the 210526 panel.
# VERIFY: confirm this is the intended panel for the thesis comparison.
DEFAULT_PANEL = (
    "beachcamweb/apps/prediction/pipeline_workspace/"
    "clean_dataset_backup/all_clean.csv"
)


# ─────────────────────────────────────────────────────────────────────────────
# Panel preparation (copied verbatim from run_cross_year_unified.py)
# ─────────────────────────────────────────────────────────────────────────────

# Manual aliases — 1:1 matches that slugification alone misses.
_UID_ALIASES = {
    "marina":                                  "platja-marina",
    "camp-de-mar":                             "golf-camp-de-mar",
    "port-palma":                              "port",
    "platja-d-or-can-pastilla-desde-bonaona":  "can-pastilla-bonaona",
    "platja-d-or-can-pastilla":                "can-pastilla-mallorca-pipeline",
}


def _slugify_uid(name: str) -> str:
    """Normalise unique_id so 2022 display names match 2025 production slugs."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = re.sub(r"[-_]\d+$", "", s)
    return _UID_ALIASES.get(s, s)


def cap_real_outliers(df: pd.DataFrame, k: float | None,
                      percentile: int = CAP_PERCENTILE,
                      day_start: int = 8, day_end: int = 20) -> pd.DataFrame:
    """Thin wrapper over the canonical ``crowd_outliers.cap_outliers`` (single
    source of truth). Clamps the REAL crowd ``y`` above ``k*P{percentile}`` per
    series; predictions never pass here; the capacity denominator uses ``y_raw``.
    """
    df, _ = _cap_outliers(df, y_col="y", k=k, percentile=percentile,
                          day_start=day_start, day_end=day_end)
    return df


def load_panel(panel_csv: Path, cap_k: float | None = DEFAULT_CAP_K) -> pd.DataFrame:
    """Read the unified panel and reindex each series to a continuous hourly grid.

    NeuralForecast with freq="h" enforces strict hourly continuity at predict
    time. The raw panel is daytime-only with sporadic gaps, so we reindex
    per-series, forward/back-fill exogenous features, and fill missing y with 0
    (night/offline = zero occupancy). A daytime mask is kept for the metric.
    """
    df = pd.read_csv(panel_csv, low_memory=False)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.dropna(subset=["unique_id"]).copy()

    n_before = df["unique_id"].nunique()
    df["unique_id"] = df["unique_id"].map(_slugify_uid)
    n_after = df["unique_id"].nunique()
    print(f"[info] slugified unique_ids: {n_before} -> {n_after}")

    df = df.drop_duplicates(subset=["unique_id", "ds"], keep="first")

    weather_cols = [c for c in df.columns if c.startswith("om_")]
    pieces = []
    for uid, g in df.groupby("unique_id", sort=False):
        g = g.sort_values("ds").drop_duplicates("ds")
        full_idx = pd.date_range(g["ds"].min(), g["ds"].max(), freq="h")
        g = g.set_index("ds").reindex(full_idx).rename_axis("ds").reset_index()
        g["unique_id"] = uid
        if weather_cols:
            g[weather_cols] = g[weather_cols].ffill().bfill()
        g["is_padded"] = g["y"].isna()      # reindex-filled hour, no real observation
        g["y"] = g["y"].fillna(0)
        pieces.append(g)
    df = pd.concat(pieces, ignore_index=True)

    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_daytime"] = df["hour"].between(8, 20).astype(int)

    df["y_raw"] = df["y"]          # pre-cap crowd, kept for the capacity denominator
    df = cap_real_outliers(df, cap_k)

    print(f"[info] panel: {len(df):,} rows ({int(df['is_daytime'].sum()):,} daytime), "
          f"{df['unique_id'].nunique()} series, "
          f"ds {df['ds'].min().date()} -> {df['ds'].max().date()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Splits — cross_year and expanding (both leakage-checked)
# ─────────────────────────────────────────────────────────────────────────────

def split_train_test(df: pd.DataFrame, protocol: str,
                     test_start: str, test_end: str):
    """Return (train_df, test_df) for the chosen protocol.

    cache2022  : train = 2022 (cache era) rows only — learn general patterns; test
                 = ALL beaches present in the 2025-26 window, INCLUDING those never
                 seen in 2022 (the global model predicts them zero-shot from their
                 own pre-window history + static covariates). No beach intersection
                 is required. This is the "more data" comparison protocol.
    cross_year : train = 2022 rows only; test restricted to beaches also in 2022.
    expanding  : train = ALL rows strictly before test_start.
    The train set always ends strictly before the test window (no temporal leak).
    """
    test_mask = (df["ds"] >= test_start) & (df["ds"] <= test_end)
    if protocol in ("cross_year", "cache2022"):
        train_mask = df["ds"].dt.year == 2022
    elif protocol == "expanding":
        # Strictly-before guard: every train timestamp precedes the window start.
        train_mask = df["ds"] < pd.Timestamp(test_start)
    else:
        raise ValueError(f"unknown protocol: {protocol}")

    train_df = df.loc[train_mask].copy().reset_index(drop=True)
    test_df = df.loc[test_mask].copy().reset_index(drop=True)

    # Leakage assertion: no overlap between train and test timestamps.
    assert train_df["ds"].max() < pd.Timestamp(test_start), \
        "leakage: train extends into the test window"

    train_beaches = set(train_df["unique_id"].unique())
    test_beaches = set(test_df["unique_id"].unique())
    zero_shot = sorted(test_beaches - train_beaches)
    print(f"[info] overlap(train&test)={len(train_beaches & test_beaches)}  "
          f"test-only(zero-shot)={len(zero_shot)}")
    if protocol == "cross_year" and zero_shot:
        # cross_year requires the beach to exist in both eras.
        print(f"[info] cross_year drops {len(zero_shot)} test-only beaches: {zero_shot}")
        test_df = test_df[test_df["unique_id"].isin(train_beaches)].reset_index(drop=True)
    elif zero_shot:
        print(f"[info] {protocol} keeps {len(zero_shot)} zero-shot test beaches: {zero_shot}")

    print(f"[info] [{protocol}] train: {len(train_df):,} rows "
          f"({train_df['ds'].min().date()}..{train_df['ds'].max().date()}) "
          f"from {train_df['unique_id'].nunique()} series")
    print(f"[info] [{protocol}] test:  {len(test_df):,} rows "
          f"({test_start}..{test_end}) from {test_df['unique_id'].nunique()} series")

    if test_df.empty:
        raise SystemExit("[fatal] no overlapping beaches between train and test window.")
    return train_df, test_df


def make_walkforward_folds(panel: pd.DataFrame, n_folds: int,
                           fold_start: str, fold_end: str):
    """Expanding walk-forward CV folds (in-distribution, no look-ahead).

    Split [fold_start, fold_end] into n_folds contiguous time windows. Fold i =
    (train: every panel row with ds < window_i start — all history before it;
    test: window_i). Each fold trains strictly on its past and validates the next
    window, so the per-series-P90 relMAE pooled over folds is an honest
    in-distribution estimate. Returns a list of (train_df, test_df)."""
    fs, fe = pd.Timestamp(fold_start), pd.Timestamp(fold_end)
    edges = pd.date_range(fs, fe, periods=n_folds + 1)
    folds = []
    for i in range(n_folds):
        w0, w1 = edges[i], edges[i + 1]
        train = panel.loc[panel["ds"] < w0].copy().reset_index(drop=True)
        test = panel.loc[(panel["ds"] >= w0) & (panel["ds"] <= w1)].copy().reset_index(drop=True)
        real = test[~test["is_padded"]] if "is_padded" in test.columns else test
        if train.empty or real.empty:
            print(f"[walkforward] fold {i} skipped (train empty or no real test rows)")
            continue
        folds.append((train, test))
    return folds


def compute_capacity(train_df: pd.DataFrame) -> dict[str, float]:
    """P90 capacity per series, TRAIN daytime rows only, floored at 1.

    Leakage guard: the denominator must never see the test window. Restricting
    to operational hours (8-20) matches the production normalisation; night
    zero-padding would otherwise halve the P90.
    """
    src = train_df[~train_df["is_padded"]] if "is_padded" in train_df.columns else train_df
    day = src[src["hour"].between(8, 20)] if "hour" in src.columns else src
    # Denominator on the UNCAPPED crowd (y_raw): the outlier cap must clean the
    # targets, never shrink the capacity. They differ only when the per-series
    # train P90 exceeds the panel-wide cap threshold (1.5*P90); using y_raw keeps
    # the denominator at the true typical-high occupancy regardless.
    col = "y_raw" if "y_raw" in day.columns else "y"
    raw = day.groupby("unique_id")[col].quantile(0.9).to_dict()
    return {uid: max(float(v), 1.0) for uid, v in raw.items()}


def build_static_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-beach static features from REAL (non-padded) daytime rows only."""
    s0 = train_df[~train_df["is_padded"]] if "is_padded" in train_df.columns else train_df
    src = s0[s0["hour"].between(8, 20)] if "hour" in s0.columns else s0
    return (
        src.groupby("unique_id")["y"]
        .agg(stat_mean_y="mean",
             stat_cv=lambda s: s.std() / (s.mean() + 1e-8))
        .reset_index()
    )


def issuance_dates(test_df: pd.DataFrame, weekday: int = 0) -> list[pd.Timestamp]:
    """Mondays in the test window, used as rolling forecast origins."""
    if test_df.empty or test_df["ds"].dropna().empty:
        return []
    ts_min = test_df["ds"].min().normalize()
    ts_max = test_df["ds"].max().normalize()
    out, cur = [], ts_min
    while cur <= ts_max:
        if cur.weekday() == weekday:
            out.append(cur)
        cur += pd.Timedelta(days=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Inner validation for HP tuning (temporal tail of TRAIN — never the test window)
# ─────────────────────────────────────────────────────────────────────────────

def inner_val_cutoff(train_df: pd.DataFrame, frac: float = 0.18) -> pd.Timestamp:
    """Timestamp splitting TRAIN into inner-train / inner-val by the last `frac`.

    The HP objective is scored on this tail. It is a slice of TRAIN, so it
    precedes the test window by construction — the test set never informs HP.
    """
    ds_sorted = train_df["ds"].sort_values()
    return ds_sorted.iloc[int(len(ds_sorted) * (1.0 - frac))]


def nf_inner_split(train_df: pd.DataFrame, cutoff: pd.Timestamp,
                  futr: list[str], hist: list[str], input_size: int,
                  horizon_hours: int):
    """Inner-train rows (ds < cutoff) and per-beach inner-val origins.

    Returns (inner_train_df, list_of (beach, ctx_df, val_window_df)). Each
    origin uses the input_size rows ending just before `cutoff` as context and
    the next horizon_hours rows (>= cutoff) as the validation target.
    """
    inner_train = train_df[train_df["ds"] < cutoff].copy()
    train_cols = ["unique_id", "ds", "y"] + list(dict.fromkeys(futr + hist))
    origins = []
    for beach, g in train_df.groupby("unique_id"):
        ctx = g[g["ds"] < cutoff].sort_values("ds").tail(input_size)
        window = g[g["ds"] >= cutoff].sort_values("ds").head(horizon_hours)
        if len(ctx) < input_size or len(window) < horizon_hours:
            continue
        origins.append((beach, ctx[train_cols].copy(),
                        window[["unique_id", "ds", "y"] + futr].copy()))
    return inner_train, origins


# ─────────────────────────────────────────────────────────────────────────────
# Metrics — primary per-series relMAE + cross-checks (thesis compute_rel_mae)
# ─────────────────────────────────────────────────────────────────────────────

def _restrict_daytime(preds: pd.DataFrame) -> pd.DataFrame:
    df = preds.dropna(subset=["y_true", "y_pred"]).copy()
    if df.empty:
        return df
    if "is_padded" in df.columns:        # score only REAL observations, not reindex-padding
        df = df[~df["is_padded"].fillna(False)].copy()
    df["hour"] = pd.to_datetime(df["ds"]).dt.hour
    df = df[df["hour"].between(8, 20)].copy()
    df["month"] = pd.to_datetime(df["ds"]).dt.month
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    return df


def per_series_rel_mae(df: pd.DataFrame, capacity: dict[str, float],
                      min_rows: int = 5) -> tuple[float, pd.DataFrame]:
    """Thesis primary metric: per-series relMAE = MAE_series / P90_series,
    then the MEAN over series. P90 is the TRAIN capacity, floored at 1."""
    rows = []
    for uid, sub in df.groupby("unique_id"):
        if len(sub) < min_rows:
            continue
        mae = float(np.mean(np.abs(sub["y_true"].values - sub["y_pred"].values)))
        rmse = float(np.sqrt(np.mean((sub["y_true"].values - sub["y_pred"].values) ** 2)))
        try:
            r2 = float(r2_score(sub["y_true"], sub["y_pred"]))
        except Exception:
            r2 = float("nan")
        if uid not in capacity:
            continue                      # no train-derived capacity -> skip; never fall back to test-window P90 (I3)
        p90 = max(float(capacity[uid]), 1.0)
        mean_y = float(sub["y_true"].mean())
        rows.append({"unique_id": uid, "n_rows": len(sub), "P90": p90,
                     "MAE": mae, "RMSE": rmse, "R2": r2,
                     "relMAE": mae / p90,
                     "mae_over_mean": mae / mean_y if mean_y > 0 else float("nan")})
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return float("nan"), rdf
    return float(rdf["relMAE"].mean()), rdf.sort_values("relMAE")


def _ratio_of_means(df: pd.DataFrame, capacity: dict[str, float]) -> float:
    """Pooled cross-check: sum(abs_err) / sum(P90 over the matched rows)."""
    if df.empty:
        return float("nan")
    cap = df["unique_id"].map(lambda u: max(float(capacity.get(u, 1.0)), 1.0))
    denom = float(cap.sum())
    return float(df["abs_err"].sum() / denom) if denom > 0 else float("nan")


def _mean_normalised(df: pd.DataFrame) -> float:
    """Cross-check: MAE / mean(y) on the pooled matched rows."""
    if df.empty:
        return float("nan")
    mean_y = float(df["y_true"].mean())
    return float(df["abs_err"].mean() / mean_y) if mean_y > 0 else float("nan")


def compute_all_metrics(preds: pd.DataFrame, capacity: dict[str, float],
                       model_name: str, horizon_days: int,
                       protocol: str, seed: int) -> tuple[dict, pd.DataFrame]:
    """All metrics for one (model, horizon, protocol, seed) on its matched rows."""
    df = _restrict_daytime(preds)
    base = {"model": model_name, "horizon_days": horizon_days,
            "protocol": protocol, "seed": seed,
            "n_rows": int(len(df)), "n_beaches": int(df["unique_id"].nunique()) if len(df) else 0}
    if df.empty:
        nan_keys = ["relMAE_all", "relMAE_season", "relMAE_summer",
                    "ratioMeans_all", "meanNorm_all", "mae_users",
                    "r2_median", "r2_mean"]
        return {**base, **{k: float("nan") for k in nan_keys}}, pd.DataFrame()

    df_season = df[df["month"].isin(SEASON_MONTHS)]
    df_summer = df[df["month"].isin(SUMMER_MONTHS)]
    rel_all, per_beach = per_series_rel_mae(df, capacity)
    summary = {
        **base,
        "relMAE_all":     rel_all,
        "relMAE_season":  per_series_rel_mae(df_season, capacity)[0],
        "relMAE_summer":  per_series_rel_mae(df_summer, capacity)[0],
        "ratioMeans_all": _ratio_of_means(df, capacity),
        "meanNorm_all":   _mean_normalised(df),
        "mae_users":      float(df["abs_err"].mean()),
        "r2_median":      float(per_beach["R2"].median()) if len(per_beach) else float("nan"),
        "r2_mean":        float(per_beach["R2"].mean()) if len(per_beach) else float("nan"),
    }
    return summary, per_beach


# ─────────────────────────────────────────────────────────────────────────────
# NeuralForecast — model construction (reduced spaces per audit §7)
# ─────────────────────────────────────────────────────────────────────────────

def build_nf_model(family: str, horizon_hours: int, hp: dict,
                  futr: list[str], hist: list[str], max_steps: int):
    """Construct a TFT or LSTM with the proven nf 3.1.5 kwargs.

    FIXED axes (genuinely negligible per the 200-trial audit): n_head=4,
    batch_size=64 (TFT), early_stop_patience_steps=45. SEARCHED (incl. the
    boundary-reopened input_size {24,36,48}, attn_dropout, scaler+standard) come
    from `hp`.
    """
    from neuralforecast.losses.pytorch import MAE

    if family == "tft":
        from neuralforecast.models import TFT
        model = TFT(
            h=horizon_hours,
            input_size=hp.get("input_size", 48),  # SEARCH {24,36,48} (optimum below 48)
            hidden_size=hp["hidden_size"],       # SEARCH {96,192,256}
            n_head=4,                            # FIX (negligible, audit)
            batch_size=NF_BATCH_SIZE,            # FIX (negligible, audit); --batch-size lowers it

            learning_rate=hp["lr"],              # SEARCH log[2e-5,1e-3]
            dropout=hp["dropout"],               # SEARCH [0,0.4]
            attn_dropout=hp.get("attn_dropout", 0.15),  # SEARCH [0,0.4] (edge signal at 15d)
            max_steps=max_steps,
            early_stop_patience_steps=45,        # FIX (with narrowed lr)
            scaler_type=hp["scaler"],            # SEARCH {robust,minmax,standard}
            loss=MAE(),
            futr_exog_list=futr or None,
            hist_exog_list=hist or None,
            stat_exog_list=STAT_EXOG,
            val_check_steps=50,
            random_seed=hp.get("seed", SEED),
            start_padding_enabled=True,
            enable_progress_bar=False,
        )
        return model, "TFT"

    from neuralforecast.models import LSTM
    model = LSTM(
        h=horizon_hours,
        input_size=hp.get("input_size", 48),     # SEARCH {24,36,48}
        encoder_n_layers=hp["encoder_n_layers"], # SEARCH {1,2,3}
        encoder_hidden_size=hp["encoder_hidden_size"],  # SEARCH {64,128,256}
        decoder_hidden_size=64,                  # FIX (negligible per audit)
        decoder_layers=2,                        # FIX
        encoder_dropout=hp["encoder_dropout"],   # SEARCH [0,0.4]
        learning_rate=hp["lr"],                  # SEARCH log[2e-5,1e-3]
        batch_size=NF_BATCH_SIZE,                # FIX (negligible per audit); --batch-size lowers it
        max_steps=max_steps,
        early_stop_patience_steps=45,            # FIX
        scaler_type=hp["scaler"],                # SEARCH {robust,minmax,standard}
        loss=MAE(),
        futr_exog_list=futr or None,
        hist_exog_list=hist or None,
        stat_exog_list=STAT_EXOG,
        val_check_steps=50,
        random_seed=hp.get("seed", SEED),
        start_padding_enabled=True,
        enable_progress_bar=False,
    )
    return model, "LSTM"


def nf_search_space(trial, family: str) -> dict:
    """Reduced-but-boundary-safe Optuna space per family.

    Reopened vs the audit §7 reduction, based on the 200-trial campaign:
    - input_size {24,36,48}: input=48 won at the LOWER edge in 6/6 TFT cells with
      a monotone trend, and the earlier Phase-2 sweep favoured 24 — so the optimum
      likely sits below 48 (never tested). Smaller also cuts memory/time.
    - scaler includes 'standard': it won the backup-15d cell (was missing from TFT).
    - attn_dropout [0,0.4]: long-horizon backup optima hugged the 0.3 edge.
    Genuinely negligible axes stay fixed in build_nf_model (n_head=4, batch=64,
    early_stop_patience=45) to keep the space at ~6 dims.
    """
    if family == "tft":
        return {
            "hidden_size":  trial.suggest_categorical("hidden_size", [96, 192, 256]),
            "lr":           trial.suggest_float("lr", 2e-5, 1e-3, log=True),
            "dropout":      trial.suggest_float("dropout", 0.0, 0.4),
            "scaler":       trial.suggest_categorical("scaler", ["robust", "minmax", "standard"]),
            "input_size":   trial.suggest_categorical("input_size", [24, 36, 48]),
            "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.4),
        }
    return {
        "encoder_n_layers":    trial.suggest_categorical("encoder_n_layers", [1, 2, 3]),
        "lr":                  trial.suggest_float("lr", 2e-5, 1e-3, log=True),
        "encoder_dropout":     trial.suggest_float("encoder_dropout", 0.0, 0.4),
        "encoder_hidden_size": trial.suggest_categorical("encoder_hidden_size", [64, 128, 256]),
        "scaler":              trial.suggest_categorical("scaler", ["robust", "minmax", "standard"]),
        "input_size":          trial.suggest_categorical("input_size", [24, 36, 48]),
    }


def _nf_predict_origins(nf, col: str, origins, futr: list[str],
                       static_df: pd.DataFrame) -> pd.DataFrame:
    """Predict each (beach, ctx, window) origin and stack into long form.

    `ctx` is the real-history context ending before the origin; `futr_df` is the
    known-future window. With freq="h" and real timestamps this is the proven
    cross-year hindcast call (df=ctx, futr_df=window).
    """
    preds = []
    for beach, ctx_df, window in origins:
        futr_df = window[["unique_id", "ds"] + futr].copy()
        try:
            out = nf.predict(df=ctx_df, futr_df=futr_df, static_df=static_df)
        except Exception as e:
            print(f"  [warn] predict failed for {beach}: {e}")
            continue
        if col not in out.columns:
            cands = [c for c in out.columns if c not in {"unique_id", "ds"}]
            if not cands:
                continue
            out = out.rename(columns={cands[0]: col})
        out = out[["unique_id", "ds", col]].rename(columns={col: "y_pred"})
        out["y_pred"] = out["y_pred"].clip(lower=0)
        out = out.merge(window[["unique_id", "ds", "y"]], on=["unique_id", "ds"], how="left")
        out = out.rename(columns={"y": "y_true"}).dropna(subset=["y_true"])
        preds.append(out)
    if not preds:
        return pd.DataFrame(columns=["unique_id", "ds", "y_pred", "y_true"])
    return pd.concat(preds, ignore_index=True)


def nf_objective(trial, family, horizon_hours, train_df, inner_cutoff,
                futr, hist, capacity) -> float:
    """Inner-validation relMAE for one HP trial. Train on TRAIN-tail-excluded
    rows, score on the inner-val tail (never the test window)."""
    from neuralforecast import NeuralForecast

    hp = nf_search_space(trial, family)
    inner_train, origins = nf_inner_split(
        train_df, inner_cutoff, futr, hist, input_size=hp.get("input_size", 48),
        horizon_hours=horizon_hours)
    if not origins:
        return 1e6
    model, col = build_nf_model(family, horizon_hours, hp, futr, hist,
                               NF_TRIAL_MAX_STEPS)
    nf = NeuralForecast(models=[model], freq="h")
    train_cols = ["unique_id", "ds", "y"] + list(dict.fromkeys(futr + hist))
    static_df = build_static_df(inner_train)
    try:
        # val_size>0 required (early_stop_patience_steps is set); the early-stop
        # tail is carved from inner_train, separate from the inner-val origins
        # (ds >= inner_cutoff) used for scoring — no leakage.
        nf.fit(df=inner_train[train_cols], static_df=static_df, val_size=horizon_hours)
    except Exception as e:
        print(f"  [trial fail] {e}")
        return 1e6
    pred = _nf_predict_origins(nf, col, origins, futr, static_df)
    if pred.empty:
        return 1e6
    df = _restrict_daytime(pred)
    rel, _ = per_series_rel_mae(df, capacity)
    return float(rel) if not np.isnan(rel) else 1e6


def nf_fit_and_eval(family: str, horizon_hours: int, hp: dict,
                   train_df: pd.DataFrame, test_df: pd.DataFrame,
                   capacity: dict, panel_df: pd.DataFrame,
                   static_src_df: pd.DataFrame = None) -> pd.DataFrame:
    """Refit a NF model on FULL train with `hp`, then roll over test origins.

    Real history for each origin is sourced from the full panel up to (but not
    including) that origin — for cross_year that history is still only the rows
    the model was trained to expect; the rolling forecast feeds genuine context,
    never future values.
    """
    from neuralforecast import NeuralForecast

    futr = [c for c in TFT_FUTR if c in train_df.columns]
    hist = [c for c in TFT_HIST if c in train_df.columns]
    # Static covariates + capacity cover ALL series (incl. zero-shot test beaches)
    # when static_src_df is given (cache2022); else just the train series.
    static_df = build_static_df(static_src_df if static_src_df is not None else train_df)
    model, col = build_nf_model(family, horizon_hours, hp, futr, hist,
                               NF_FINAL_MAX_STEPS)
    nf = NeuralForecast(models=[model], freq="h")
    train_cols = ["unique_id", "ds", "y"] + list(dict.fromkeys(futr + hist))
    # VERIFY (needs GPU run): freq="h" with the reindexed-continuous panel is the
    # proven cross-year path from run_cross_year_unified.py. The recurrent LSTM
    # predict(df=ctx, futr_df=window) with futr_exog also works in the companion
    # script; confirm both families emit predictions before a long run.
    nf.fit(df=train_df[train_cols], static_df=static_df, val_size=horizon_hours)

    # Build rolling origins from the test window; context comes from the panel.
    # Context length must match the model's searched input_size (not a fixed 48).
    input_size = hp.get("input_size", 48)
    issuances = issuance_dates(test_df)
    origins = []
    for beach, beach_test in test_df.groupby("unique_id"):
        beach_hist = panel_df[panel_df["unique_id"] == beach]
        for issue in issuances:
            ctx = beach_hist[beach_hist["ds"] < issue].sort_values("ds").tail(input_size)
            window = beach_test[beach_test["ds"] >= issue].sort_values("ds").head(horizon_hours)
            if len(ctx) < input_size or len(window) < horizon_hours:
                continue
            wcols = ["unique_id", "ds", "y"] + (["is_padded"] if "is_padded" in window.columns else []) + futr
            w = window[wcols].copy()
            w["issue_date"] = issue.date().isoformat()
            origins.append((beach, ctx[train_cols].copy(), w))

    preds = []
    for beach, ctx_df, window in origins:
        issue = window["issue_date"].iloc[0]
        futr_df = window[["unique_id", "ds"] + futr].copy()
        try:
            out = nf.predict(df=ctx_df, futr_df=futr_df, static_df=static_df)
        except Exception as e:
            print(f"  [warn] predict failed for {beach} {issue}: {e}")
            continue
        if col not in out.columns:
            cands = [c for c in out.columns if c not in {"unique_id", "ds"}]
            if not cands:
                continue
            out = out.rename(columns={cands[0]: col})
        out = out[["unique_id", "ds", col]].rename(columns={col: "y_pred"})
        out["y_pred"] = out["y_pred"].clip(lower=0)
        ycols = ["unique_id", "ds", "y"] + (["is_padded"] if "is_padded" in window.columns else [])
        out = out.merge(window[ycols], on=["unique_id", "ds"], how="left")
        out = out.rename(columns={"y": "y_true"}).dropna(subset=["y_true"])
        out["issue_date"] = issue
        preds.append(out)
    if not preds:
        return pd.DataFrame(columns=["unique_id", "ds", "y_pred", "y_true", "issue_date"])
    return pd.concat(preds, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# XGBoost — Castelle schema (8-param space kept; fANOVA computed here too)
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb_features(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Castelle-style shifted-feature matrix: target is y[t+h]."""
    df = df.copy().sort_values(["unique_id", "ds"]).reset_index(drop=True)
    df["y_lag1"] = df.groupby("unique_id")["y"].shift(1)
    df["y_roll12"] = df.groupby("unique_id")["y"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    df["y_roll36"] = df.groupby("unique_id")["y"].transform(
        lambda s: s.shift(1).rolling(36, min_periods=1).mean())

    ahead = [c for c in (CASTELLE_WEATHER + CALENDAR) if c in df.columns]
    df_fut = df.groupby("unique_id")[ahead].shift(-horizon_hours)
    df_fut.columns = [f"{c}_t_plus_h" for c in ahead]

    df["y_target"] = df.groupby("unique_id")["y"].shift(-horizon_hours)
    df["ds_target"] = df.groupby("unique_id")["ds"].shift(-horizon_hours)

    out = pd.concat([
        df[["unique_id", "ds", "ds_target", "y_lag1", "y_roll12", "y_roll36"]],
        df_fut,
        df[["y_target"]],
    ], axis=1)
    return out.dropna(subset=["y_target", "y_lag1"]).reset_index(drop=True)


def xgb_search_space(trial) -> dict:
    """Existing 8-param XGB space (cheap, kept at the same --trials)."""
    return {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
        "max_depth":        trial.suggest_int("max_depth", 3, 12),
        "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
        "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
    }


def xgb_search(train_feat: pd.DataFrame, feat_cols: list[str],
              inner_cutoff: pd.Timestamp, trials: int, n_startup: int):
    """Tune XGB on an inner temporal tail of TRAIN; return (study, X_tr, y_tr).

    The inner split is by ds (not a positional cut) so the validation tail is a
    genuine temporal hold-out of TRAIN, never the test window.
    """
    import optuna
    from xgboost import XGBRegressor

    inner_tr = train_feat[train_feat["ds"] < inner_cutoff]
    inner_va = train_feat[train_feat["ds"] >= inner_cutoff]
    X_a, y_a = inner_tr[feat_cols].values, inner_tr["y_target"].values
    X_b, y_b = inner_va[feat_cols].values, inner_va["y_target"].values
    if len(X_a) < 50 or len(X_b) < 20:  # fallback to 90/10 positional split
        cut = int(len(train_feat) * 0.85)
        X_a, y_a = train_feat[feat_cols].iloc[:cut].values, train_feat["y_target"].iloc[:cut].values
        X_b, y_b = train_feat[feat_cols].iloc[cut:].values, train_feat["y_target"].iloc[cut:].values

    def objective(trial):
        hp = xgb_search_space(trial)
        m = XGBRegressor(**hp, tree_method="hist", n_jobs=-1,
                        random_state=SEED, verbosity=0)
        m.fit(X_a, y_a, verbose=False)
        return float(np.mean(np.abs(m.predict(X_b) - y_b)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=n_startup))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    X_tr = train_feat[feat_cols].values
    y_tr = train_feat["y_target"].values
    return study, X_tr, y_tr


def xgb_fit_predict(best_hp: dict, X_tr, y_tr, test_feat: pd.DataFrame,
                   feat_cols: list[str], seed: int) -> pd.DataFrame:
    """Refit XGB on full train with best_hp + a given seed, predict the test."""
    from xgboost import XGBRegressor
    model = XGBRegressor(**best_hp, tree_method="hist", n_jobs=-1,
                        random_state=seed, verbosity=0)
    model.fit(X_tr, y_tr, verbose=False)
    pred = model.predict(test_feat[feat_cols].values).clip(0, None)
    out = test_feat[["unique_id", "ds", "ds_target"]].rename(
        columns={"ds": "origin_ds", "ds_target": "ds"}).copy()
    out["y_pred"] = pred
    out["y_true"] = test_feat["y_target"].values
    # XGB has no weekly issuance concept; record the feature origin for traceability.
    # The identical-rows join keys on (unique_id, ds) only, so this is informational.
    out["issue_date"] = pd.to_datetime(out["origin_ds"]).dt.date.astype(str)
    return out, model


# ─────────────────────────────────────────────────────────────────────────────
# Diebold-Mariano (HAC within blocks + HLN), reused from dm_tft_vs_lstm.py
# ─────────────────────────────────────────────────────────────────────────────

def hac_var_blocks(d, blocks, lag):
    """Newey-West long-run variance of mean(d); autocovariances within blocks
    only (Bartlett weights), then pooled. Returns var(mean). Verbatim from
    07_model_evaluation_and_validation/dm_tft_vs_lstm.py."""
    d = np.asarray(d, float)
    n = len(d)
    dc = d - d.mean()
    g0 = np.sum(dc * dc) / n
    gammas = [g0]
    for k in range(1, lag + 1):
        s = 0.0
        for idx in blocks:
            if len(idx) > k:
                a = dc[idx[k:]]
                b = dc[idx[:-k]]
                s += np.sum(a * b)
        gammas.append(s / n)
    lrv = gammas[0] + 2.0 * sum((1 - k / (lag + 1)) * gammas[k] for k in range(1, lag + 1))
    lrv = max(lrv, 1e-12)
    return lrv / n


def dm_pair(matched: pd.DataFrame, col_a: str, col_b: str,
           horizon_hours: int) -> dict | None:
    """DM test of model A vs B on identical (series, origin, lead) rows.

    `matched` has columns y_true, <col_a>, <col_b>, unique_id (already inner-
    joined). Loss = absolute error; HAC blocks = per series; HLN small-sample
    correction; Wilcoxon as the rank-based check. Logic mirrors
    dm_tft_vs_lstm.py / dm_summer_from_reeval.py.
    """
    sub = matched.dropna(subset=[col_a, col_b, "y_true"]).copy()
    _sort_keys = ["unique_id"] + [c for c in ("issue_date", "ds") if c in sub.columns]
    sub = sub.sort_values(_sort_keys, kind="stable").reset_index(drop=True)  # stable + temporal: HAC autocov needs time order within each series block
    n = len(sub)
    if n < 10:
        return None
    ea = (sub[col_a] - sub["y_true"]).abs()
    eb = (sub[col_b] - sub["y_true"]).abs()
    blocks, start = [], 0
    for _, g in sub.groupby("unique_id", sort=False):
        blocks.append(np.arange(start, start + len(g)))
        start += len(g)
    lag = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))  # Newey-West 1994 bandwidth
    d = (ea - eb).to_numpy()
    var = hac_var_blocks(d, blocks, lag)
    dm = d.mean() / np.sqrt(var)
    corr = np.sqrt(max((n + 1 - 2 * horizon_hours + horizon_hours * (horizon_hours - 1) / n) / n, 1e-6))
    dm_hln = dm * corr
    p_hln = 2 * stats.t.sf(abs(dm_hln), df=n - 1)
    try:
        w_p = float(stats.wilcoxon(ea, eb).pvalue)
    except Exception:
        w_p = float("nan")
    p90 = max(float(sub["y_true"].quantile(0.90)), 1.0)
    return {
        "model_A": col_a, "model_B": col_b, "n_pairs": n,
        "n_beaches": int(sub["unique_id"].nunique()), "lag_NW": lag,
        "relMAE_A": float(ea.mean() / p90), "relMAE_B": float(eb.mean() / p90),
        "mean_loss_diff": float(d.mean()),
        "DM_HLN": float(dm_hln), "p_HLN": float(p_hln),
        "wilcoxon_p": w_p,
        "winner": col_a if d.mean() < 0 else col_b,
    }


# ─────────────────────────────────────────────────────────────────────────────
# fANOVA importance + combined figure
# ─────────────────────────────────────────────────────────────────────────────

def param_importance(study) -> pd.DataFrame:
    """fANOVA hyperparameter importance for one Optuna study."""
    import optuna
    try:
        imp = optuna.importance.get_param_importances(
            study, evaluator=optuna.importance.FanovaImportanceEvaluator(seed=SEED))
    except Exception as e:
        print(f"  [warn] fANOVA failed: {e}")
        imp = {}
    if not imp:
        return pd.DataFrame(columns=["param", "importance"])
    return pd.DataFrame(
        [{"param": k, "importance": float(v)} for k, v in imp.items()]
    ).sort_values("importance", ascending=False)


def plot_importance_grid(imp_by_cell: dict[str, pd.DataFrame], out_path: Path):
    """One bar panel per (family, horizon) cell. Fonts >= 10pt."""
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 12, "axes.labelsize": 10})
    cells = [k for k, v in imp_by_cell.items() if v is not None and len(v)]
    if not cells:
        return
    ncol = 3
    nrow = int(np.ceil(len(cells) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, key in enumerate(cells):
        ax = axes[i // ncol][i % ncol]
        ax.axis("on")
        d = imp_by_cell[key].head(8)
        ax.barh(d["param"][::-1], d["importance"][::-1], color="#3a7ca5")
        ax.set_title(key, fontsize=12)
        ax.set_xlabel("fANOVA importance", fontsize=10)
        ax.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] fANOVA figure -> {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Prior-best warm path: reuse the 200-trial campaign optima (skip the search)
# ─────────────────────────────────────────────────────────────────────────────

PRIOR_OPTUNA_DB = Path(__file__).resolve().parent / "optuna" / "cross_year.db"
PRIOR_CAMPAIGN = "cy_capped_backup_20260623_115630"  # cross-year capped 150-trial campaign (TFT/LSTM/XGB x 3/10/15d), bundled in scripts/optuna/cross_year.db


def load_prior_best(family: str, horizon_days: int, campaign: str = None):
    """best_params of the matching 200-trial study, or None if unavailable."""
    campaign = campaign or PRIOR_CAMPAIGN   # read global at call time (CLI override)
    if not PRIOR_OPTUNA_DB.exists():
        print(f"  [warn] prior optuna db not found: {PRIOR_OPTUNA_DB}")
        return None
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    name = f"{campaign}__{family}__{horizon_days}d"
    try:
        st = optuna.load_study(study_name=name, storage=f"sqlite:///{PRIOR_OPTUNA_DB}")
        return dict(st.best_params)
    except Exception as e:
        print(f"  [warn] prior best not loaded for {name}: {e}")
        return None


def prior_best_for_build(family: str, raw: dict) -> dict:
    """Map a 200-trial study's best_params onto the names build_nf_model reads.
    No clamping: the refit accepts any value (the categorical choices only
    constrain the *search* space, not the model construction)."""
    if family == "tft":
        return {k: raw[k] for k in
                ("hidden_size", "lr", "dropout", "scaler", "input_size", "attn_dropout")
                if k in raw}
    if family == "lstm":
        out = {}
        for k in ("encoder_n_layers", "lr", "scaler", "input_size"):
            if k in raw:
                out[k] = raw[k]
        if "dropout" in raw:      out["encoder_dropout"] = raw["dropout"]       # name differs
        if "hidden_size" in raw:  out["encoder_hidden_size"] = raw["hidden_size"]  # name differs
        return out
    return dict(raw)  # xgb: XGBRegressor accepts every key


# ─────────────────────────────────────────────────────────────────────────────
# Per-family driver: HP search -> multi-seed refit -> predictions + metrics
# ─────────────────────────────────────────────────────────────────────────────

def run_family(family: str, horizon_days: int, protocol: str,
              train_df: pd.DataFrame, test_df: pd.DataFrame,
              panel_df: pd.DataFrame, capacity: dict,
              trials: int, n_startup: int, seeds: list[int],
              out_dir: Path, use_prior_best: bool = False,
              static_src_df: pd.DataFrame = None) -> dict:
    """Returns dict with: best_params, importance_df, seed_summaries (list),
    and the per-row predictions of the FIRST seed (used for the identical-rows
    DM join — one deterministic prediction set per family)."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    horizon_hours = HORIZON_HOURS[horizon_days]
    label = f"{family}_{horizon_days}d"
    if use_prior_best:
        print(f"\n[{label}/{protocol}] prior-best refit (no search) ...")
    else:
        print(f"\n[{label}/{protocol}] HP search ({trials} trials, startup {n_startup}) ...")

    if family in ("tft", "lstm"):
        futr = [c for c in TFT_FUTR if c in train_df.columns]
        hist = [c for c in TFT_HIST if c in train_df.columns]
        study = None
        raw = load_prior_best(family, horizon_days) if use_prior_best else None
        if raw:
            best = prior_best_for_build(family, raw)
            print(f"[{label}/{protocol}] PRIOR-BEST ({PRIOR_CAMPAIGN}) hp={best}")
        else:
            inner_cut = inner_val_cutoff(train_df)
            study = optuna.create_study(
                direction="minimize",
                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=n_startup))
            study.optimize(
                lambda t: nf_objective(t, family, horizon_hours, train_df, inner_cut,
                                      futr, hist, capacity),
                n_trials=trials, show_progress_bar=False)
            best = dict(study.best_params)
            print(f"[{label}/{protocol}] best inner relMAE={study.best_value:.4f} hp={best}")

        seed_summaries, first_preds = [], None
        for sd in seeds:
            hp = {**best, "seed": sd}
            preds = nf_fit_and_eval(family, horizon_hours, hp, train_df, test_df,
                                   capacity, panel_df, static_src_df=static_src_df)
            s, pb = compute_all_metrics(preds, capacity, family.upper(),
                                       horizon_days, protocol, sd)
            seed_summaries.append(s)
            if first_preds is None:
                first_preds = preds
                preds.to_csv(out_dir / f"per_row_predictions_{family}_{horizon_days}d.csv", index=False)
                pb.to_csv(out_dir / f"per_beach_{family}_{horizon_days}d.csv", index=False)
            print(f"[{label}/{protocol}] seed {sd}: relMAE_summer={s['relMAE_summer']:.4f}")
        imp = param_importance(study) if study is not None else pd.DataFrame(columns=["param", "importance"])

    else:  # xgb
        train_feat = build_xgb_features(train_df, horizon_hours)
        test_feat_full = build_xgb_features(panel_df, horizon_hours)
        # Test rows: targets landing inside the window AND origin consistent with
        # the protocol's train/test split for the XGB shifted features.
        ws, we = test_df["ds"].min(), test_df["ds"].max()
        tmask = test_feat_full["ds_target"].between(ws, we)
        if protocol in ("cross_year", "cache2022"):
            # feature origin must be in the 2025 era (lags from real 2025 history),
            # never a 2022 training row.
            tmask &= test_feat_full["ds"].dt.year != 2022
        else:
            tmask &= test_feat_full["ds"] >= ws  # origin within/after window start
        test_feat = test_feat_full.loc[tmask].copy()
        feat_cols = [c for c in train_feat.columns
                     if c not in {"unique_id", "ds", "ds_target", "y_target"}]
        study = None
        raw = load_prior_best("xgb", horizon_days) if use_prior_best else None
        if raw:
            best = dict(raw)
            X_tr = train_feat[feat_cols].values
            y_tr = train_feat["y_target"].values
            print(f"[{label}/{protocol}] PRIOR-BEST ({PRIOR_CAMPAIGN}) hp={best}")
        else:
            inner_cut = inner_val_cutoff(train_df)
            study, X_tr, y_tr = xgb_search(train_feat, feat_cols, inner_cut, trials, n_startup)
            best = dict(study.best_params)
            print(f"[{label}/{protocol}] best inner MAE={study.best_value:.4f} hp={best}")

        seed_summaries, first_preds = [], None
        for sd in seeds:
            preds, _ = xgb_fit_predict(best, X_tr, y_tr, test_feat, feat_cols, sd)
            s, pb = compute_all_metrics(preds, capacity, "XGB",
                                       horizon_days, protocol, sd)
            seed_summaries.append(s)
            if first_preds is None:
                first_preds = preds
                preds.to_csv(out_dir / f"per_row_predictions_xgb_{horizon_days}d.csv", index=False)
                pb.to_csv(out_dir / f"per_beach_xgb_{horizon_days}d.csv", index=False)
            print(f"[{label}/{protocol}] seed {sd}: relMAE_summer={s['relMAE_summer']:.4f}")
        imp = param_importance(study) if study is not None else pd.DataFrame(columns=["param", "importance"])

    (out_dir / f"best_params_{family}_{horizon_days}d.json").write_text(
        json.dumps(best, indent=2))
    if len(imp):
        imp.to_csv(out_dir / f"param_importance_{family}_{horizon_days}d.csv", index=False)
    return {"best_params": best, "importance": imp,
            "seed_summaries": seed_summaries, "first_preds": first_preds}


# ─────────────────────────────────────────────────────────────────────────────
# Identical-rows join + seed aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_seeds(seed_summaries: list[dict]) -> dict:
    """mean +/- SD over seeds for every numeric metric."""
    df = pd.DataFrame(seed_summaries)
    metric_cols = ["relMAE_all", "relMAE_season", "relMAE_summer",
                   "ratioMeans_all", "meanNorm_all", "mae_users",
                   "r2_median", "r2_mean"]
    base = {k: df[k].iloc[0] for k in ["model", "horizon_days", "protocol"]}
    base["n_seeds"] = len(df)
    base["n_rows"] = int(df["n_rows"].iloc[0])
    base["n_beaches"] = int(df["n_beaches"].iloc[0])
    for c in metric_cols:
        base[f"{c}_mean"] = float(df[c].mean())
        base[f"{c}_sd"] = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
    return base


def identical_rows(preds_by_model: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Inner-join the families on (unique_id, ds) so the DM test and the matched
    metrics use the EXACT same (series, target-timestamp) rows for every model.

    The join key is (unique_id, ds) — the absolute target timestamp — NOT the
    origin: the NF models carry a weekly issue_date but XGB has no origin concept
    (Castelle shifted features), so joining on issue_date would yield zero
    cross-family rows. NF can predict the same ds from overlapping Monday
    origins, so each (unique_id, ds) is collapsed to its first prediction before
    the join (same dedup as dm_summer_from_reeval.py). Returns a wide frame:
    unique_id, ds, y_true, <MODEL> per family.
    """
    keys = ["unique_id", "ds"]
    merged = None
    for name, p in preds_by_model.items():
        if p is None or p.empty:
            continue
        d = p.copy()
        d["ds"] = pd.to_datetime(d["ds"]).astype(str)
        d = (d[keys + ["y_true", "y_pred"]]
             .drop_duplicates(subset=keys, keep="first")  # collapse overlapping origins
             .rename(columns={"y_pred": name}))
        if merged is None:
            merged = d
        else:
            # y_true carried from the first model; assert targets agree on the key.
            chk = merged.merge(d[keys + ["y_true"]], on=keys, how="inner",
                              suffixes=("", "_b"))
            if len(chk) and (chk["y_true"] - chk["y_true_b"]).abs().max() > 1e-6:
                print("[warn] y_true mismatch across families on matched keys")
            merged = merged.merge(d.drop(columns=["y_true"]), on=keys, how="inner")
    if merged is None:
        return pd.DataFrame()
    return merged.reset_index(drop=True)


def matched_metrics(matched: pd.DataFrame, families: list[str],
                   capacity: dict, horizon_days: int, protocol: str) -> list[dict]:
    """Score every family on the IDENTICAL matched rows — the comparison-grade
    table (same row set per model, unlike the per-model full-row summary). Audit
    §2 warns that full-row summaries are not cross-family comparable; this is.
    """
    rows = []
    for fam in families:
        col = fam.upper()
        if col not in matched.columns:
            continue
        sub = matched[["unique_id", "ds", "y_true", col]].rename(columns={col: "y_pred"})
        s, _ = compute_all_metrics(sub, capacity, col, horizon_days, protocol, seed=-1)
        s["row_set"] = "identical_matched"
        rows.append(s)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Protocol record + ranking
# ─────────────────────────────────────────────────────────────────────────────

def write_protocol(out_dir: Path, args, panel_path: Path, protocol: str,
                  train_df: pd.DataFrame, test_df: pd.DataFrame,
                  capacity: dict, nf_version: str):
    record = {
        "protocol": protocol,
        "panel_path": str(panel_path),
        "nf_version": nf_version,
        "seed_base": SEED,
        "seeds": args.seeds,
        "trials": args.trials,
        "n_startup_trials": args.startup,
        "horizons_days": args.horizons,
        "horizon_hours": {str(d): HORIZON_HOURS[d] for d in args.horizons},
        "families": args.families,
        "cap_k": args.cap_k,
        "cap_percentile": CAP_PERCENTILE,
        "test_window": [args.test_start, args.test_end],
        "train_window": [str(train_df["ds"].min().date()), str(train_df["ds"].max().date())],
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_beaches": int(test_df["unique_id"].nunique()),
        "issuance": "weekly Mondays (rolling forecast origins)",
        "daytime_window": "08:00-20:00",
        "season_months": sorted(SEASON_MONTHS),
        "summer_months": sorted(SUMMER_MONTHS),
        "capacity_def": "per-series P90 on TRAIN daytime rows, floored at max(P90,1)",
        "primary_metric": "per-series relMAE = MAE_series / P90_series, mean over series",
        "cross_check_metrics": ["ratio-of-means (pooled)", "mean-normalised (MAE/mean_y)",
                                "raw MAE (users)", "per-beach R2"],
        "inner_val": "temporal tail of TRAIN (last ~18%), never the test window",
        "dm": "abs-error loss, Newey-West HAC within series blocks + HLN + Wilcoxon, identical rows",
        "search_spaces": {
            "tft": {"FIX": {"n_head": 4, "batch_size": 64,
                            "early_stop_patience_steps": 45},
                    "SEARCH": {"hidden_size": [96, 192, 256],
                               "lr": "loguniform[2e-5,1e-3]", "dropout": [0.0, 0.4],
                               "scaler": ["robust", "minmax", "standard"],
                               "input_size": [24, 36, 48],
                               "attn_dropout": [0.0, 0.4]}},
            "lstm": {"FIX": {"batch_size": 64, "decoder_hidden_size": 64,
                             "decoder_layers": 2, "early_stop_patience_steps": 45},
                     "SEARCH": {"encoder_n_layers": [1, 2, 3],
                                "lr": "loguniform[2e-5,1e-3]", "encoder_dropout": [0.0, 0.4],
                                "encoder_hidden_size": [64, 128, 256],
                                "scaler": ["robust", "minmax", "standard"],
                                "input_size": [24, 36, 48]}},
            "xgb": {"SEARCH": "8-param Castelle space (n_estimators, max_depth, lr, "
                              "subsample, colsample_bytree, min_child_weight, "
                              "reg_alpha, reg_lambda)"},
        },
        "nf_max_steps": {"trial": NF_TRIAL_MAX_STEPS, "final": NF_FINAL_MAX_STEPS},
        "generated_at": datetime.now().isoformat(),
    }
    (out_dir / "protocol.json").write_text(json.dumps(record, indent=2))
    print(f"[save] protocol.json -> {out_dir / 'protocol.json'}")


def write_ranking(out_dir: Path, agg_rows: list[dict], protocol: str):
    df = pd.DataFrame(agg_rows)
    if df.empty:
        return
    df = df.sort_values(["horizon_days", "relMAE_summer_mean"]).reset_index(drop=True)
    cols = ["model", "horizon_days", "n_seeds", "n_rows",
            "relMAE_all_mean", "relMAE_all_sd",
            "relMAE_season_mean", "relMAE_summer_mean", "relMAE_summer_sd",
            "mae_users_mean", "r2_median_mean"]
    cols = [c for c in cols if c in df.columns]
    md = [f"# Ranking — protocol: {protocol}", "",
          "Primary metric: per-series relMAE (mean over series), lower is better. "
          "Values are mean +/- SD over seeds.", "",
          df[cols].to_markdown(index=False, floatfmt=".4f")]
    (out_dir / "ranking.md").write_text("\n".join(md))
    print(f"[save] ranking.md -> {out_dir / 'ranking.md'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_protocol(protocol: str, panel: pd.DataFrame, panel_path: Path,
                args, nf_version: str):
    out_dir = Path(args.out) / protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\nPROTOCOL: {protocol}  ->  {out_dir}\n{'='*72}")

    train_df, test_df = split_train_test(panel, protocol, args.test_start, args.test_end)
    # Capacity + static covariates: for cache2022 they must cover ALL test beaches
    # (incl. zero-shot ones), so use each series' REAL rows BEFORE the test window
    # (no leakage). For cross_year/expanding the train set already covers the
    # tested beaches, so train_df is the source.
    if protocol == "cache2022":
        static_src = panel[panel["ds"] < pd.Timestamp(args.test_start)].copy()
    else:
        static_src = train_df
    capacity = compute_capacity(static_src)
    write_protocol(out_dir, args, panel_path, protocol, train_df, test_df,
                  capacity, nf_version)

    agg_rows, dm_rows, matched_rows = [], [], []
    importance_by_cell = {}

    for h in args.horizons:
        preds_by_model: dict[str, pd.DataFrame] = {}
        for family in args.families:
            res = run_family(family, h, protocol, train_df, test_df, panel,
                            capacity, args.trials, args.startup,
                            list(range(SEED, SEED + args.seeds)), out_dir,
                            use_prior_best=args.use_prior_best, static_src_df=static_src)
            agg_rows.append(aggregate_seeds(res["seed_summaries"]))
            if len(res["importance"]):
                importance_by_cell[f"{family.upper()} {h}d"] = res["importance"]
            preds_by_model[family.upper()] = res["first_preds"]

        # Identical test rows across the families, then matched metrics + DM.
        matched = identical_rows(preds_by_model)
        if matched.empty:
            print(f"[warn] no identical rows for horizon {h}d — skipping DM")
            continue
        matched.to_csv(out_dir / f"identical_rows_{h}d.csv", index=False)
        matched_rows.extend(matched_metrics(matched, args.families, capacity, h, protocol))
        hh = HORIZON_HOURS[h]
        # DM on DAYTIME rows only — consistent with the metrics and the thesis
        # 8-20 convention (night rows are reindex-padded zeros and would dilute it).
        matched_day = matched[pd.to_datetime(matched["ds"]).dt.hour.between(8, 20)]
        for a, b in [("TFT", "LSTM"), ("TFT", "XGB")]:
            if a in matched.columns and b in matched.columns:
                r = dm_pair(matched_day, a, b, hh)
                if r:
                    dm_rows.append({"protocol": protocol, "horizon_days": h, **r})

    pd.DataFrame(agg_rows).to_csv(out_dir / "metrics_summary.csv", index=False)
    if matched_rows:
        pd.DataFrame(matched_rows).to_csv(out_dir / "matched_metrics.csv", index=False)
        print(f"[save] matched_metrics.csv (identical rows) -> {out_dir / 'matched_metrics.csv'}")
    if dm_rows:
        pd.DataFrame(dm_rows).to_csv(out_dir / "dm_results.csv", index=False)
        print(f"[save] dm_results.csv -> {out_dir / 'dm_results.csv'}")
    plot_importance_grid(importance_by_cell, out_dir / "param_importance_fanova.png")
    write_ranking(out_dir, agg_rows, protocol)
    print(f"[done] protocol {protocol} -> {out_dir}")


def run_walkforward(panel: pd.DataFrame, panel_path: Path, args, nf_version: str):
    """Comparison 2 — in-distribution walk-forward CV. N expanding folds; each fold
    trains on its past and validates the window. Pools the per-fold predictions and
    runs the SAME downstream as run_protocol (identical rows across families +
    per-series-P90 relMAE + Diebold-Mariano), so it is on the exact same evaluation
    basis as the cross_year protocol (Comparison 1). HP = prior-best (per-fold search
    is infeasible); the deployable final model is trained separately by
    cross_year_train_3_models (the servable trainer)."""
    django_only = getattr(args, "django_only", False)
    if django_only:
        # Pure in-distribution CV: drop the 2022 cache entirely so BOTH training
        # and testing live in the django era (>=2025). Uses every django beach
        # (~16), with no cross-era beach matching.
        n0, b0 = len(panel), panel["unique_id"].nunique()
        panel = panel[panel["ds"].dt.year >= 2025].copy()
        print(f"[django-only] panel {n0:,} rows/{b0} beaches "
              f"-> {len(panel):,} rows/{panel['unique_id'].nunique()} beaches (>=2025)")
    proto = "walkforward_django" if django_only else "walkforward"
    out_dir = Path(args.out) / proto
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*72}\nPROTOCOL: {proto}  ->  {out_dir}\n{'='*72}")

    folds = make_walkforward_folds(panel, args.n_folds, args.test_start, args.test_end)
    if not folds:
        print("[fatal] no walk-forward folds produced"); return
    print(f"[info] {len(folds)} expanding folds over {args.test_start}..{args.test_end}")
    # Single per-series P90 capacity (full-panel real daytime) shared across folds,
    # so the denominator is identical everywhere — consistent with Comparison 1.
    capacity = compute_capacity(panel)
    write_protocol(out_dir, args, panel_path, proto, folds[0][0], folds[-1][1],
                  capacity, nf_version)

    matched_rows, dm_rows = [], []
    for h in args.horizons:
        preds_by_model: dict[str, pd.DataFrame] = {}
        for family in args.families:
            fam_preds = []
            for fi, (tr, te) in enumerate(folds):
                res = run_family(family, h, proto, tr, te, panel, capacity,
                                args.trials, args.startup,
                                list(range(SEED, SEED + args.seeds)), out_dir,
                                use_prior_best=True, static_src_df=tr)
                p = res.get("first_preds")
                if p is not None and len(p):
                    p = p.copy(); p["fold"] = fi
                    fam_preds.append(p)
            if fam_preds:
                preds_by_model[family.upper()] = pd.concat(fam_preds, ignore_index=True)

        matched = identical_rows(preds_by_model)
        if matched.empty:
            print(f"[warn] no identical rows for horizon {h}d — skipping")
            continue
        matched.to_csv(out_dir / f"identical_rows_{h}d.csv", index=False)
        matched_rows.extend(matched_metrics(matched, args.families, capacity, h, proto))
        hh = HORIZON_HOURS[h]
        matched_day = matched[pd.to_datetime(matched["ds"]).dt.hour.between(8, 20)]
        for a, b in [("TFT", "LSTM"), ("TFT", "XGB")]:
            if a in matched.columns and b in matched.columns:
                r = dm_pair(matched_day, a, b, hh)
                if r:
                    dm_rows.append({"protocol": proto, "horizon_days": h, **r})

    if matched_rows:
        mdf = pd.DataFrame(matched_rows)
        mdf.to_csv(out_dir / "matched_metrics.csv", index=False)
        rk = mdf.sort_values(["horizon_days", "relMAE_summer"]).reset_index(drop=True)
        cols = [c for c in ["model", "horizon_days", "n_rows", "n_beaches",
                            "relMAE_all", "relMAE_season", "relMAE_summer"] if c in rk.columns]
        (out_dir / "ranking.md").write_text(
            f"# Walk-forward CV ranking — {len(folds)} expanding folds\n\n"
            "Per-series P90 relMAE on identical rows (same basis as cross_year).\n\n"
            + rk[cols].to_markdown(index=False, floatfmt=".4f"))
        print(f"[save] matched_metrics.csv + ranking.md -> {out_dir}")
    if dm_rows:
        pd.DataFrame(dm_rows).to_csv(out_dir / "dm_results.csv", index=False)
    print(f"[done] protocol {proto} -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # cache2022 (default): train on 2022 cache to learn general patterns; validate on
    #   ALL 2025-26 django beaches (zero-shot for those not in 2022) — predict vs real,
    #   no beach intersection required (more test data).
    # cross_year: train 2022, but restrict the test to beaches also present in 2022.
    # (expanding dropped: the dual-era 810-day gap zero-fill contaminates its training.)
    ap.add_argument("--protocol", choices=["cache2022", "cross_year", "walkforward"], default="cache2022",
                    help="cross_year/cache2022 = Comparison 1 (train 2022 -> validate django). "
                         "walkforward = Comparison 2 (in-distribution N-fold walk-forward CV, "
                         "same identical-rows + P90/series + DM downstream).")
    ap.add_argument("--n-folds", dest="n_folds", type=int, default=4,
                    help="walkforward: number of expanding CV folds over the test window.")
    ap.add_argument("--django-only", dest="django_only", action="store_true",
                    help="walkforward: restrict to the django era (>=2025) — drop the "
                         "2022 cache from train AND test. All ~16 django beaches, pure "
                         "in-distribution CV, no cross-era matching. Writes to "
                         "validated_run/walkforward_django/.")
    ap.add_argument("--horizons", type=int, nargs="*", default=[3, 10, 15],
                    help="Horizons in DAYS (3 10 15 -> 36 120 180 h).")
    ap.add_argument("--families", nargs="*", default=["tft", "lstm", "xgb"],
                    choices=["tft", "lstm", "xgb"])
    ap.add_argument("--trials", type=int, default=80,
                    help="Optuna trials per family (equal budget; ~6-dim TFT space).")
    ap.add_argument("--startup", type=int, default=20,
                    help="TPE startup (random) trials.")
    ap.add_argument("--seeds", type=int, default=5,
                    help="Number of seeds for the multi-seed refit.")
    ap.add_argument("--panel", default=DEFAULT_PANEL,
                    help="Panel CSV (default: clean backup panel).")
    ap.add_argument("--test-start", dest="test_start", default="2025-06-01",
                    help="Apr-May 2025 stays as warm-up (static + first context).")
    ap.add_argument("--test-end", dest="test_end", default="2026-02-28",
                    help="Default covers all 2025-26 django data (more test data).")
    ap.add_argument("--out", default="new_training_pipeline/validated_run")
    ap.add_argument("--cap-k", dest="cap_k", type=float, default=DEFAULT_CAP_K,
                    help="Outlier cap: clamp real y above k*P90 (per series). "
                         "Applied to ground-truth only, never predictions. "
                         "0 disables.")
    ap.add_argument("--use-prior-best", dest="use_prior_best", action="store_true",
                    help="Skip the HP search; refit each (family,horizon) with the "
                         "200-trial campaign's best params from optuna/cross_year.db.")
    ap.add_argument("--prior-campaign", dest="prior_campaign", default=None,
                    help="Override the prior-best study campaign prefix.")
    ap.add_argument("--quick", action="store_true",
                    help="Partial look: --use-prior-best with seeds=1 (no search).")
    ap.add_argument("--batch-size", dest="batch_size", type=int, default=NF_BATCH_SIZE,
                    help="NN minibatch size. Lower (8/16) to fit 15d (H=195) on small RAM.")
    args = ap.parse_args()

    if args.quick:                       # fast partial run with best-known params
        args.use_prior_best = True
        args.seeds = 1
    globals()["NF_BATCH_SIZE"] = args.batch_size
    if args.prior_campaign:
        globals()["PRIOR_CAMPAIGN"] = args.prior_campaign

    # Resolve panel path relative to the repo root if needed.
    panel_path = Path(args.panel)
    if not panel_path.is_absolute():
        # Anchor at beach_counting/ (this script lives in new_training_pipeline/).
        repo_root = Path(__file__).resolve().parent.parent
        cand = repo_root / args.panel
        panel_path = cand if cand.exists() else panel_path
    panel_path = panel_path.resolve()
    if not panel_path.exists():
        sys.exit(f"[fatal] panel csv not found: {panel_path}")

    try:
        import neuralforecast
        nf_version = neuralforecast.__version__
    except Exception:
        nf_version = "unknown"

    print(f"[info] panel: {panel_path}")
    print(f"[info] nf version: {nf_version}")
    panel = load_panel(panel_path, cap_k=args.cap_k)

    t0 = time.time()
    if args.protocol == "walkforward":
        run_walkforward(panel, panel_path, args, nf_version)   # Comparison 2 (in-distribution CV)
    else:
        run_protocol(args.protocol, panel, panel_path, args, nf_version)  # Comparison 1 (cross_year) / cache2022
    print(f"\n[all done] protocol={args.protocol} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
