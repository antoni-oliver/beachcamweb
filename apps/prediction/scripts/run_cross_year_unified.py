#!/usr/bin/env python3
"""
Unified cross-year training & ranking — TFT, LSTM, XGBoost side-by-side.

Goal
====
Reproduce the exact same training and evaluation protocol across three model
families and emit a single ranking table at the end. All three models share:
    - the same panel (clean_dataset_backup),
    - the same train window (2022 rows only — cross-year regime),
    - the same test window (Jun-Aug 2025, scenario S1),
    - the same eligibility filter on webcams,
    - the same metric definition (P90-normalised relMAE on all/season/summer
      buckets + per-beach R^2).

Cross-year means: train once on 2022 data only; at inference time, feed each
2025-summer issuance window through the trained model and aggregate the
forecasts back into a single ranking table.

For TFT and LSTM (NeuralForecast), this is done by:
  1. Fitting the model on the 2022 train rows.
  2. Iterating over each (beach, issuance_date) in the 2025 test window.
  3. Building a futr_df for the next H daytime hours starting at issuance.
  4. Calling nf.predict(futr_df=window) and collecting predictions.

For XGBoost (Castelle 2025 schema), it's the standard shifted-feature setup:
  build features at time t whose target is y[t+h], train on 2022 rows, predict
  on Jun-Aug 2025 rows.

Outputs
-------
    cross_year_results/
        predictions_TFT.csv      long-form predictions per (beach, ds)
        predictions_LSTM.csv     idem
        predictions_XGB.csv      idem
        summary_per_model.csv    one row per model with all metrics
        ranking.csv              sorted by summer relMAE
        per_beach_*.csv          per-beach metrics per model
        ranking.md               human-readable headline table

Usage
-----
    cd apps/prediction
    source .venv/bin/activate
    python scripts/run_cross_year_unified.py \\
        --panel_csv /path/to/clean_dataset_backup/all_clean.csv \\
        --test_start 2025-06-01 --test_end 2025-08-31 \\
        --horizon 3d \\
        --trials 50 \\
        --output_dir cross_year_results/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# OpenMP guard — prevents XGBoost segfaulting after PyTorch has spawned threads
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from crowd_outliers import cap_outliers  # canonical per-series P90 outlier cap

warnings.filterwarnings("ignore")

# ─── Django setup — resolves models roots from settings, regardless of cwd ──
HERE = Path(__file__).resolve().parent.parent  # apps/prediction
PROJ = HERE.parent.parent                       # beachcamweb root
sys.path.insert(0, str(PROJ))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
try:
    import django
    django.setup()
    from django.conf import settings as dj_settings
    DEFAULT_NF_ROOT = Path(dj_settings.TFT_MODELS_DIR).resolve()
    DEFAULT_XGB_ROOT = DEFAULT_NF_ROOT.parent / "xgb_models"
except Exception as _e:
    print(f"[warn] Django not available ({_e}); --nf_models_root / --xgb_models_root must be passed explicitly")
    DEFAULT_NF_ROOT = None
    DEFAULT_XGB_ROOT = None


# ─────────────────────────────────────────────────────────────────────────────
# Constants — shared protocol
# ─────────────────────────────────────────────────────────────────────────────

HOURS_PER_DAY = 12  # 8:00-20:00 daytime filter
HORIZONS = {"3d": 36, "10d": 120, "15d": 180}
SEASON_MONTHS = {4, 5, 6, 7, 8, 9}
SUMMER_MONTHS = {6, 7, 8}

# Castelle 2025 weather features (humidity omitted)
CASTELLE_WEATHER = [
    "om_temperature_2m", "om_precipitation", "om_wind_speed_10m",
    "om_wind_direction_10m", "om_shortwave_radiation",
]

# Calendar features shared by all three models
CALENDAR = ["hour", "day_of_week", "month", "is_weekend"]

# TFT/LSTM 3-input typology (NeuralForecast)
TFT_STAT = ["stat_mean_y", "stat_cv"]
TFT_FUTR = ["hour", "day_of_week", "month", "is_weekend",
            "om_temperature_2m", "om_apparent_temperature", "om_cloud_cover"]
TFT_HIST = ["om_shortwave_radiation", "om_vapour_pressure_deficit"]

# FIXED-HP triple from Phase 3 of the deployed TFT (Section 4.7 of thesis)
FIXED_HP = {
    "hidden_size": 64,
    "n_head": 4,
    "dropout": 0.05,
    "lr": 1e-3,
    "input_size": 24,
    "max_steps": 300,
}


# ─────────────────────────────────────────────────────────────────────────────
# Panel preparation
# ─────────────────────────────────────────────────────────────────────────────

# Manual aliases — clear 1:1 matches that slugification alone misses because
# the production slug uses a different lexical form than the display name.
_UID_ALIASES = {
    "marina":                                  "platja-marina",
    "camp-de-mar":                             "golf-camp-de-mar",
    "port-palma":                              "port",
    "platja-d-or-can-pastilla-desde-bonaona":  "can-pastilla-bonaona",
    "platja-d-or-can-pastilla":                "can-pastilla-mallorca-pipeline",
}


def _slugify_uid(name: str) -> str:
    """Normalise unique_id so 2022 display names match 2025 production slugs.

    Examples
    --------
    "Cala Vedella"           → "cala-vedella"
    "cala-vedella_1"         → "cala-vedella"      (drops _N disambiguator)
    "Son Serra de Marina"    → "son-serra-de-marina"
    "Muro-Sector I"          → "muro-sector-i"
    "Punta de Sa Galera"     → "punta-de-sa-galera"
    "Marina"                 → "platja-marina"     (alias)
    """
    import re, unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = re.sub(r"[-_]\d+$", "", s)
    return _UID_ALIASES.get(s, s)


def load_panel(panel_csv: Path) -> pd.DataFrame:
    """Read the unified panel and reindex each series to continuous hourly grid.

    NeuralForecast with freq="h" enforces strict hourly continuity at predict
    time. The raw panel is daytime-only (operational hours 8-20) with sporadic
    intra-day gaps, so we reindex per-series, forward-fill exogenous features,
    and fill missing y with 0 (treating night/offline as zero occupancy). A
    daytime mask is preserved so metrics can be restricted later.

    The raw panel mixes 2022 display names with 2025 production slugs; both are
    slugified into a canonical form so cross-year cameras line up.
    """
    df = pd.read_csv(panel_csv, low_memory=False)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.dropna(subset=["unique_id"]).copy()

    n_before = df["unique_id"].nunique()
    df["unique_id"] = df["unique_id"].map(_slugify_uid)
    n_after = df["unique_id"].nunique()
    print(f"[info] slugified unique_ids: {n_before} → {n_after} after normalisation")

    # After slugification, multiple raw ids can collapse to the same uid (e.g.
    # cala-vedella + cala-vedella_1). Dedup on (uid, ds) keeping the first row.
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
        g["y"] = g["y"].fillna(0)
        pieces.append(g)
    df = pd.concat(pieces, ignore_index=True)

    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_daytime"] = df["hour"].between(8, 20).astype(int)

    df, _ = cap_outliers(df)          # per-series P90 outlier cap on the real y

    print(f"[info] panel: {len(df):,} rows ({int(df['is_daytime'].sum()):,} daytime), "
          f"{df['unique_id'].nunique()} series, "
          f"ds range {df['ds'].min().date()} → {df['ds'].max().date()}")
    return df


def split_cross_year(df: pd.DataFrame, test_start: str, test_end: str):
    """Cross-year split: train on 2022 rows only, test on the given window.

    Test beaches are intersected with the train-beach set — NeuralForecast
    cannot predict for series it never saw during fit.
    """
    test_mask = (df["ds"] >= test_start) & (df["ds"] <= test_end)
    train_mask = df["ds"].dt.year == 2022
    train_df = df.loc[train_mask].copy().reset_index(drop=True)
    test_df = df.loc[test_mask].copy().reset_index(drop=True)

    train_beaches = set(train_df["unique_id"].unique())
    test_beaches = set(test_df["unique_id"].unique())
    overlap = sorted(train_beaches & test_beaches)
    print(f"[info] beach overlap (train ∩ test): {len(overlap)} → {overlap}")
    dropped = sorted(test_beaches - train_beaches)
    if dropped:
        print(f"[info] dropping {len(dropped)} test-only beaches (unseen in 2022 train): {dropped}")
        test_df = test_df[test_df["unique_id"].isin(train_beaches)].reset_index(drop=True)
    train_only = sorted(train_beaches - test_beaches)
    if train_only:
        print(f"[info] keeping {len(train_only)} train-only beaches for model context: {train_only}")

    print(f"[info] train (2022): {len(train_df):,} rows from {train_df['unique_id'].nunique()} series")
    print(f"[info] test  ({test_start}..{test_end}): {len(test_df):,} rows from {test_df['unique_id'].nunique()} series")

    if test_df.empty:
        raise SystemExit(
            "[fatal] no overlapping beaches between 2022 train and test window. "
            "Either widen the train window via --train_years, choose a test window "
            "with shared cameras, or supply a beach-id mapping. Exiting cleanly."
        )
    return train_df, test_df


def compute_capacity(df: pd.DataFrame) -> dict[str, float]:
    """P90 capacity per series on daytime rows only.

    The panel is reindexed to continuous hourly grid with night y=0 padding,
    so a full-panel P90 deflates by ~half. We restrict to operational hours
    (8-20) so the denominator matches the production normalisation.
    """
    day = df[df["hour"].between(8, 20)] if "hour" in df.columns else df
    return day.groupby("unique_id")["y"].quantile(0.9).to_dict()


def build_static_df(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-beach static features computed on the train window, daytime only.

    Using daytime rows keeps the static signal aligned with the metric: night
    zero-padding would otherwise pull stat_mean_y and stat_cv toward the wrong
    scale.
    """
    src = train_df[train_df["hour"].between(8, 20)] if "hour" in train_df.columns else train_df
    return (
        src.groupby("unique_id")["y"]
        .agg(stat_mean_y="mean",
             stat_cv=lambda s: s.std() / (s.mean() + 1e-8))
        .reset_index()
    )


def issuance_dates(test_df: pd.DataFrame, weekday: int = 0) -> list[pd.Timestamp]:
    """Mondays in the test window, used as forecast issuance dates."""
    if test_df.empty or test_df["ds"].dropna().empty:
        return []
    ts_min = test_df["ds"].min().normalize()
    ts_max = test_df["ds"].max().normalize()
    out = []
    cur = ts_min
    while cur <= ts_max:
        if cur.weekday() == weekday:
            out.append(cur)
        cur += pd.Timedelta(days=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Model persistence — register trained models for auto-discovery
# ─────────────────────────────────────────────────────────────────────────────

def save_nf_model_set(nf, model_class: str, horizon_hours: int,
                      input_size: int, futr: list[str], hist: list[str],
                      static_df: pd.DataFrame, horizon_label: str,
                      set_root: Path, set_name: str) -> Path:
    """Save a NeuralForecast model in the layout tft_service auto-discovers.

    Layout matches `_discover_model_sets()` requirements:
        <set_root>/<set_name>/<model>_<horizon_label>/
            config.json
            nf_model/                  (NeuralForecast.save artefact)
            static_features.csv
    """
    set_dir = set_root / set_name / f"{model_class.lower()}_{horizon_label}"
    if set_dir.exists():
        shutil.rmtree(set_dir)
    set_dir.mkdir(parents=True, exist_ok=True)

    nf.save(path=str(set_dir / "nf_model"), overwrite=True,
            save_dataset=False)

    config = {
        "model_type": model_class,
        "horizon": horizon_hours,
        "input_size": input_size,
        "futr_exog": futr,
        "hist_exog": hist,
        "stat_exog": list(static_df.columns.drop("unique_id")),
        "trained_at": datetime.now().isoformat(),
        "regime": "cross_year_train_2022",
    }
    (set_dir / "config.json").write_text(json.dumps(config, indent=2))
    static_df.to_csv(set_dir / "static_features.csv", index=False)
    print(f"[save] {model_class} → {set_dir}")
    return set_dir


def save_xgb_model_set(model, best_params: dict, horizon_hours: int,
                       feat_cols: list[str], horizon_label: str,
                       set_root: Path, set_name: str) -> Path:
    """Save XGBoost into a parallel registry (xgb_models/).

    Layout chosen so a future `xgb_service` can auto-discover by mirroring the
    tft_service scan pattern:
        <set_root>/<set_name>/xgb_<horizon_label>/
            config.json
            model.joblib
            best_params.json
    """
    set_dir = set_root / set_name / f"xgb_{horizon_label}"
    if set_dir.exists():
        shutil.rmtree(set_dir)
    set_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, set_dir / "model.joblib")
    (set_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    config = {
        "model_type": "XGBRegressor",
        "horizon": horizon_hours,
        "features": feat_cols,
        "trained_at": datetime.now().isoformat(),
        "regime": "cross_year_train_2022",
        "schema": "castelle_2025",
    }
    (set_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[save] XGB → {set_dir}")
    return set_dir


# ─────────────────────────────────────────────────────────────────────────────
# Common driver for NeuralForecast models (TFT / LSTM)
# ─────────────────────────────────────────────────────────────────────────────

def _nf_train_and_iterate(model, train_df: pd.DataFrame, test_df: pd.DataFrame,
                          horizon_hours: int, static_df: pd.DataFrame,
                          futr_cols: list[str], hist_cols: list[str],
                          model_col: str,
                          save_root: Path | None = None,
                          set_name: str | None = None,
                          horizon_label: str | None = None,
                          panel_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Fit a NeuralForecast model once on train, then issue per-window forecasts.

    For arbitrary issuance dates (well after the training cutoff), NeuralForecast
    needs an explicit `df=ctx` context window so it knows where to start
    predicting from. Without it, `predict(futr_df=...)` tries to forecast from
    the last training row and rejects our 2025 timestamps as missing.

    `model_col` is the prediction-column name (TFT, LSTM, …).
    `panel_df` (if provided) is the full panel — used to source context rows for
    issuance dates whose history isn't in test_df alone.
    """
    from neuralforecast import NeuralForecast

    nf = NeuralForecast(models=[model], freq="h")
    train_cols = ["unique_id", "ds", "y"] + futr_cols + hist_cols
    nf.fit(df=train_df[train_cols], static_df=static_df, val_size=0)

    if save_root is not None and set_name and horizon_label:
        save_nf_model_set(
            nf, model_class=model_col, horizon_hours=horizon_hours,
            input_size=FIXED_HP["input_size"], futr=futr_cols, hist=hist_cols,
            static_df=static_df, horizon_label=horizon_label,
            set_root=Path(save_root), set_name=set_name,
        )

    history_source = panel_df if panel_df is not None else pd.concat(
        [train_df, test_df], ignore_index=True)

    issuances = issuance_dates(test_df)
    input_size = FIXED_HP["input_size"]
    all_preds = []
    for beach, beach_test in test_df.groupby("unique_id"):
        beach_hist = history_source[history_source["unique_id"] == beach]
        for issue in issuances:
            # Context: input_size rows ending just before the issuance.
            ctx_mask = beach_hist["ds"] < issue
            ctx = beach_hist.loc[ctx_mask].sort_values("ds").tail(input_size)
            if len(ctx) < input_size:
                continue

            # Future: horizon_hours rows starting at the issuance.
            window = beach_test[beach_test["ds"] >= issue].sort_values("ds").head(horizon_hours)
            if len(window) < horizon_hours:
                continue

            ctx_df = ctx[train_cols].copy()
            futr_df = window[["unique_id", "ds"] + futr_cols].copy()
            try:
                preds = nf.predict(df=ctx_df, futr_df=futr_df,
                                   static_df=static_df)
            except Exception as e:
                print(f"  [warn] predict failed for {beach} {issue.date()}: {e}")
                continue

            if model_col not in preds.columns:
                # NeuralForecast sometimes uses model alias as column name
                candidates = [c for c in preds.columns
                              if c not in {"unique_id", "ds"}]
                if not candidates:
                    continue
                preds = preds.rename(columns={candidates[0]: model_col})

            out = preds[["unique_id", "ds", model_col]].rename(
                columns={model_col: "y_pred"})
            out = out.merge(window[["unique_id", "ds", "y"]],
                            on=["unique_id", "ds"], how="left")
            out = out.rename(columns={"y": "y_true"}).dropna(subset=["y_true"])
            out["issue_date"] = issue.date().isoformat()
            all_preds.append(out)

    if not all_preds:
        return pd.DataFrame(columns=["unique_id", "ds", "y_pred", "y_true", "issue_date"])
    return pd.concat(all_preds, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# Model 1 — TFT (Temporal Fusion Transformer)
# ─────────────────────────────────────────────────────────────────────────────

def train_predict_tft(train_df: pd.DataFrame, test_df: pd.DataFrame,
                       horizon_hours: int, static_df: pd.DataFrame,
                       save_root: Path | None = None,
                       set_name: str | None = None,
                       horizon_label: str | None = None,
                       panel_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Train TFT with FIXED-HP on 2022, hindcast each issuance in 2025-summer."""
    from neuralforecast.models import TFT
    from neuralforecast.losses.pytorch import MAE

    futr = [c for c in TFT_FUTR if c in train_df.columns]
    hist = [c for c in TFT_HIST if c in train_df.columns]

    model = TFT(
        h=horizon_hours,
        input_size=FIXED_HP["input_size"],
        hidden_size=FIXED_HP["hidden_size"],
        n_head=FIXED_HP["n_head"],
        dropout=FIXED_HP["dropout"],
        learning_rate=FIXED_HP["lr"],
        max_steps=FIXED_HP["max_steps"],
        stat_exog_list=TFT_STAT,
        futr_exog_list=futr,
        hist_exog_list=hist,
        loss=MAE(),
        scaler_type="robust",
        random_seed=42,
        enable_progress_bar=False,
    )
    return _nf_train_and_iterate(
        model, train_df, test_df, horizon_hours, static_df,
        futr, hist, model_col="TFT",
        save_root=save_root, set_name=set_name, horizon_label=horizon_label,
        panel_df=panel_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model 2 — LSTM (recurrent baseline matched to TFT)
# ─────────────────────────────────────────────────────────────────────────────

def train_predict_lstm(train_df: pd.DataFrame, test_df: pd.DataFrame,
                        horizon_hours: int, static_df: pd.DataFrame,
                        save_root: Path | None = None,
                        set_name: str | None = None,
                        horizon_label: str | None = None,
                        panel_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """LSTM cross-year baseline.

    Architecture matches the TFT's FIXED-HP hidden_size for an apples-to-apples
    comparison: same hidden width, same learning rate, same input window.
    The NeuralForecast LSTM accepts the same three-input typology
    (static / known-future / historical-observed) so the feature schema is
    identical to TFT's. Differences vs. TFT: no Variable Selection Networks,
    no attention block, no static-context injection at every layer — the
    static features are concatenated once at the input.
    """
    from neuralforecast.models import LSTM
    from neuralforecast.losses.pytorch import MAE

    futr = [c for c in TFT_FUTR if c in train_df.columns]
    hist = [c for c in TFT_HIST if c in train_df.columns]

    model = LSTM(
        h=horizon_hours,
        input_size=FIXED_HP["input_size"],
        encoder_n_layers=2,
        encoder_hidden_size=FIXED_HP["hidden_size"],
        decoder_hidden_size=FIXED_HP["hidden_size"],
        decoder_layers=2,
        encoder_dropout=FIXED_HP["dropout"],
        learning_rate=FIXED_HP["lr"],
        max_steps=FIXED_HP["max_steps"],
        stat_exog_list=TFT_STAT,
        futr_exog_list=futr,
        hist_exog_list=hist,
        loss=MAE(),
        scaler_type="robust",
        random_seed=42,
        enable_progress_bar=False,
    )
    return _nf_train_and_iterate(
        model, train_df, test_df, horizon_hours, static_df,
        futr, hist, model_col="LSTM",
        save_root=save_root, set_name=set_name, horizon_label=horizon_label,
        panel_df=panel_df,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Model 3 — XGBoost (Castelle 2025 schema)
# ─────────────────────────────────────────────────────────────────────────────

def build_xgb_features(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """Castelle-style shifted-feature matrix: target is y[t+h]."""
    df = df.copy().sort_values(["unique_id", "ds"]).reset_index(drop=True)
    df["y_lag1"] = df.groupby("unique_id")["y"].shift(1)
    df["y_roll12"] = df.groupby("unique_id")["y"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    df["y_roll36"] = df.groupby("unique_id")["y"].transform(
        lambda s: s.shift(1).rolling(36, min_periods=1).mean())

    cols_ahead = [c for c in (CASTELLE_WEATHER + CALENDAR) if c in df.columns]
    df_future = df.groupby("unique_id")[cols_ahead].shift(-horizon_hours)
    df_future.columns = [f"{c}_t_plus_h" for c in cols_ahead]

    df["y_target"] = df.groupby("unique_id")["y"].shift(-horizon_hours)
    df["ds_target"] = df.groupby("unique_id")["ds"].shift(-horizon_hours)

    out = pd.concat([
        df[["unique_id", "ds", "ds_target", "y_lag1", "y_roll12", "y_roll36"]],
        df_future,
        df[["y_target"]],
    ], axis=1)
    return out.dropna(subset=["y_target", "y_lag1"]).reset_index(drop=True)


def train_predict_xgb(panel_df: pd.DataFrame, test_start: str, test_end: str,
                      horizon_hours: int, trials: int,
                      save_root: Path | None = None,
                      set_name: str | None = None,
                      horizon_label: str | None = None) -> pd.DataFrame:
    """XGBoost Castelle cross-year: train on 2022 rows, predict on Jun-Aug 2025."""
    import optuna
    from xgboost import XGBRegressor

    features = build_xgb_features(panel_df, horizon_hours)
    train_mask = features["ds"].dt.year == 2022
    test_mask = features["ds_target"].between(pd.Timestamp(test_start),
                                               pd.Timestamp(test_end))
    train_feat = features.loc[train_mask].copy()
    test_feat = features.loc[test_mask & (~train_mask)].copy()

    if len(train_feat) < 100 or len(test_feat) < 50:
        return pd.DataFrame(columns=["unique_id", "ds", "y_pred", "y_true"])

    feat_cols = [c for c in train_feat.columns
                 if c not in {"unique_id", "ds", "ds_target", "y_target"}]
    cut = int(len(train_feat) * 0.9)
    X_tr, y_tr = train_feat[feat_cols].iloc[:cut].values, train_feat["y_target"].iloc[:cut].values
    X_va, y_va = train_feat[feat_cols].iloc[cut:].values, train_feat["y_target"].iloc[cut:].values
    X_te, y_te = test_feat[feat_cols].values, test_feat["y_target"].values

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "tree_method":      "hist", "n_jobs": -1, "random_state": 42, "verbosity": 0,
        }
        m = XGBRegressor(**params)
        m.fit(X_tr, y_tr, verbose=False)
        return float(np.mean(np.abs(m.predict(X_va) - y_va)))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    final = XGBRegressor(**study.best_params, tree_method="hist",
                         n_jobs=-1, random_state=42, verbosity=0)
    final.fit(np.concatenate([X_tr, X_va]), np.concatenate([y_tr, y_va]),
              verbose=False)
    pred = final.predict(X_te)

    if save_root is not None and set_name and horizon_label:
        save_xgb_model_set(
            final, best_params=study.best_params,
            horizon_hours=horizon_hours, feat_cols=feat_cols,
            horizon_label=horizon_label,
            set_root=Path(save_root), set_name=set_name,
        )

    out = test_feat[["unique_id", "ds_target"]].rename(columns={"ds_target": "ds"}).copy()
    out["y_pred"] = pred
    out["y_true"] = y_te
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Shared metric computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(preds: pd.DataFrame, capacity: dict[str, float],
                     model_name: str) -> tuple[dict, pd.DataFrame]:
    """P90-normalised relMAE on all/season/summer + per-beach R².

    Restricts to daytime hours (8-20) so reindex-padded night rows don't dilute
    the metric.
    """
    df = preds.dropna(subset=["y_true", "y_pred"]).copy()
    if len(df) == 0:
        return {"model": model_name, "n_rows": 0,
                 "relMAE_all": float("nan"), "relMAE_season": float("nan"),
                 "relMAE_summer": float("nan"), "mae_users": float("nan"),
                 "r2_median": float("nan"), "r2_mean": float("nan")}, pd.DataFrame()
    df["hour"] = df["ds"].dt.hour
    df = df[df["hour"].between(8, 20)].copy()
    if len(df) == 0:
        return {"model": model_name, "n_rows": 0,
                 "relMAE_all": float("nan"), "relMAE_season": float("nan"),
                 "relMAE_summer": float("nan"), "mae_users": float("nan"),
                 "r2_median": float("nan"), "r2_mean": float("nan")}, pd.DataFrame()
    df["month"] = df["ds"].dt.month
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    df["capacity"] = df["unique_id"].map(capacity)
    missing = df["capacity"].isna() | (df["capacity"] <= 0)
    if missing.any():
        df.loc[missing, "capacity"] = df.loc[missing].groupby("unique_id")["y_true"].transform(
            lambda s: s.quantile(0.9))

    def _rel(g):
        denom = g["capacity"].mean()
        return float(g["abs_err"].mean() / denom) if denom > 0 else float("nan")

    summary = {
        "model":         model_name,
        "n_rows":        len(df),
        "relMAE_all":    _rel(df),
        "relMAE_season": _rel(df[df["month"].isin(SEASON_MONTHS)]),
        "relMAE_summer": _rel(df[df["month"].isin(SUMMER_MONTHS)]),
        "mae_users":     float(df["abs_err"].mean()),
    }
    rows = []
    for beach, g in df.groupby("unique_id"):
        if len(g) < 10:
            continue
        try:
            r2 = float(r2_score(g["y_true"], g["y_pred"]))
        except Exception:
            r2 = float("nan")
        rows.append({"unique_id": beach, "n_rows": len(g),
                     "relMAE": _rel(g), "r2": r2,
                     "mae_users": float(g["abs_err"].mean())})
    per_beach = pd.DataFrame(rows).sort_values("relMAE")
    summary["r2_median"] = float(per_beach["r2"].median()) if len(per_beach) else float("nan")
    summary["r2_mean"]   = float(per_beach["r2"].mean())   if len(per_beach) else float("nan")
    return summary, per_beach


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_csv",
                    default="apps/prediction/pipeline_workspace/clean_datasets_210526/all_clean.csv",
                    help="Path to clean_dataset_backup/all_clean.csv")
    ap.add_argument("--test_start", default="2025-06-01")
    ap.add_argument("--test_end",   default="2025-08-31")
    ap.add_argument("--horizon", choices=list(HORIZONS), default="3d",
                    help="Which horizon to evaluate. 3d=36h, 10d=120h, 15d=180h.")
    ap.add_argument("--trials", type=int, default=50,
                    help="Optuna trials for XGBoost.")
    ap.add_argument("--output_dir", default="cross_year_results")
    ap.add_argument("--skip", default="",
                    help="Comma-separated list of models to skip: TFT,LSTM,XGB")
    ap.add_argument("--nf_models_root",
                    default=str(DEFAULT_NF_ROOT) if DEFAULT_NF_ROOT else None,
                    help="Root of tft_models/ — TFT and LSTM are saved here for "
                         "tft_service auto-discovery. Defaults to "
                         "settings.TFT_MODELS_DIR.")
    ap.add_argument("--xgb_models_root",
                    default=str(DEFAULT_XGB_ROOT) if DEFAULT_XGB_ROOT else None,
                    help="Root of xgb_models/ — XGB models are saved here. "
                         "Defaults to <TFT_MODELS_DIR>/../xgb_models.")
    ap.add_argument("--no_save", action="store_true",
                    help="Disable model persistence (predictions still saved).")
    ap.add_argument("--set_name", default=None,
                    help="Set name under each models root (default: "
                         "cross_year_<horizon>_<timestamp>).")
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    # Tee stdout/stderr to a log file alongside the outputs. The proxy must
    # forward attribute access (encoding, fileno, isatty…) to the primary
    # stream — libraries like Ray Tune inspect these and crash otherwise.
    log_path = out / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_fh = open(log_path, "w", buffering=1)

    class _Tee:
        def __init__(self, primary, secondary):
            self._primary = primary
            self._secondary = secondary
        def write(self, data):
            try: self._primary.write(data)
            except Exception: pass
            try: self._secondary.write(data)
            except Exception: pass
            return len(data) if isinstance(data, str) else 0
        def flush(self):
            for s in (self._primary, self._secondary):
                try: s.flush()
                except Exception: pass
        def isatty(self):
            try: return self._primary.isatty()
            except Exception: return False
        def __getattr__(self, name):
            return getattr(self._primary, name)

    sys.stdout = _Tee(sys.__stdout__, log_fh)
    sys.stderr = _Tee(sys.__stderr__, log_fh)
    print(f"[info] log: {log_path.resolve()}")
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}
    horizon_hours = HORIZONS[args.horizon]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_name = args.set_name or f"cross_year_{args.horizon}_{timestamp}"
    if args.no_save:
        nf_root = xgb_root = None
        print("[info] model persistence disabled (--no_save)")
    else:
        nf_root = Path(args.nf_models_root).resolve() if args.nf_models_root else None
        xgb_root = Path(args.xgb_models_root).resolve() if args.xgb_models_root else None
        if nf_root:
            nf_root.mkdir(parents=True, exist_ok=True)
            print(f"[info] NF models root: {nf_root}")
        if xgb_root:
            xgb_root.mkdir(parents=True, exist_ok=True)
            print(f"[info] XGB models root: {xgb_root}")
        if nf_root or xgb_root:
            print(f"[info] persisting trained models under set '{set_name}'")

    print(f"\n{'='*72}\nUnified cross-year ranking — horizon {args.horizon} ({horizon_hours} h)\n{'='*72}")
    panel_path = Path(args.panel_csv)
    if not panel_path.is_absolute():
        panel_path = PROJ / panel_path
    panel_path = panel_path.resolve()
    if not panel_path.exists():
        sys.exit(f"[fatal] panel csv not found: {panel_path}")
    print(f"[info] panel csv: {panel_path}")
    panel = load_panel(panel_path)
    train_df, test_df = split_cross_year(panel, args.test_start, args.test_end)
    static_df = build_static_df(train_df)
    capacity = compute_capacity(panel)

    summaries = []

    # ── TFT ───────────────────────────────────────────────────────────────
    if "TFT" not in skip:
        print("\n[TFT] training …")
        preds = train_predict_tft(train_df, test_df, horizon_hours, static_df,
                                   save_root=nf_root, set_name=set_name,
                                   horizon_label=args.horizon, panel_df=panel)
        preds.to_csv(out / "predictions_TFT.csv", index=False)
        s, pb = compute_metrics(preds, capacity, "TFT")
        pb.to_csv(out / "per_beach_TFT.csv", index=False)
        summaries.append(s)
        print(f"[TFT] relMAE_summer={s['relMAE_summer']:.4f}  R²_med={s['r2_median']:.3f}")

    # ── LSTM ──────────────────────────────────────────────────────────────
    if "LSTM" not in skip:
        print("\n[LSTM] training …")
        preds = train_predict_lstm(train_df, test_df, horizon_hours, static_df,
                                    save_root=nf_root, set_name=set_name,
                                    horizon_label=args.horizon, panel_df=panel)
        preds.to_csv(out / "predictions_LSTM.csv", index=False)
        s, pb = compute_metrics(preds, capacity, "LSTM")
        pb.to_csv(out / "per_beach_LSTM.csv", index=False)
        summaries.append(s)
        print(f"[LSTM] relMAE_summer={s['relMAE_summer']:.4f}  R²_med={s['r2_median']:.3f}")

    # ── XGBoost ───────────────────────────────────────────────────────────
    if "XGB" not in skip:
        print(f"\n[XGB] training with {args.trials} Optuna trials …")
        preds = train_predict_xgb(panel, args.test_start, args.test_end,
                                   horizon_hours, args.trials,
                                   save_root=xgb_root, set_name=set_name,
                                   horizon_label=args.horizon)
        preds.to_csv(out / "predictions_XGB.csv", index=False)
        s, pb = compute_metrics(preds, capacity, "XGB")
        pb.to_csv(out / "per_beach_XGB.csv", index=False)
        summaries.append(s)
        print(f"[XGB] relMAE_summer={s['relMAE_summer']:.4f}  R²_med={s['r2_median']:.3f}")

    if not summaries:
        print("[fatal] no models ran"); return

    # ── Headline ranking ──────────────────────────────────────────────────
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(out / "summary_per_model.csv", index=False)
    ranking = summary_df.sort_values("relMAE_summer")[
        ["model", "n_rows", "relMAE_all", "relMAE_season", "relMAE_summer",
         "mae_users", "r2_median", "r2_mean"]
    ].reset_index(drop=True)
    ranking.to_csv(out / "ranking.csv", index=False)

    print("\n" + "=" * 72)
    print("Cross-year ranking (sorted by summer relMAE, lower is better)")
    print("=" * 72)
    print(ranking.to_string(index=False))

    md = ["# Cross-year ranking", "",
          f"- Horizon: **{args.horizon}** ({horizon_hours} h)",
          f"- Train: 2022 rows only",
          f"- Test:  {args.test_start} → {args.test_end}",
          f"- Normalisation: P90 per series", "",
          ranking.to_markdown(index=False, floatfmt=".4f"),
    ]
    (out / "ranking.md").write_text("\n".join(md))
    print(f"\n[done] outputs in {out}/")


if __name__ == "__main__":
    main()
