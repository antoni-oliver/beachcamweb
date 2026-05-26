#!/usr/bin/env python3
"""
Cross-Year Training — 2022 train, 2025 summer test, all three model families.

FAIRNESS PROTOCOL (this is a TFM methodology requirement — every invariant
below MUST hold for every model and dataset compared by this script):

    1. Same dataset → same train/test split.
       TFT, LSTM and XGB all see the *same* cache_2022 panel as training data
       and the *same* django_2025 daytime window as test data, on the same
       beaches (the intersection of unique_ids that appear in both eras).
    2. Same horizon set (3d, 10d, 15d daytime hours).
    3. Same Optuna search budget (n trials per model × horizon, default 50).
    4. Same metric: P90-normalised relMAE on daytime rows.
    5. Same normalisation, same primary significance test (Diebold-Mariano).
    6. Same seed (SEED = 42) for the Optuna sampler and the model init.

The invariants for each run are written to results/protocol.json so the
methodology is auditable after the fact. The same script can be pointed at
multiple datasets (--data-dirs) and runs the entire protocol independently
on each, then emits a combined cross-dataset ranking.

Output structure (per dataset):
    <out-root>/<prefix>_<dataset_tag>_<timestamp>/
        results/
            protocol.json     — fairness invariants snapshot
            summary.csv       — per (model, horizon) metrics
            ranking.csv .md   — sorted ranking
        tft_3d/ tft_10d/ tft_15d/      — TFT artifacts (nf_model/ + config + per_beach)
        lstm_3d/ lstm_10d/ lstm_15d/   — LSTM artifacts (idem)
    <apps/prediction>/xgb_models/<prefix>_<dataset_tag>_<timestamp>/
        xgb_3d/ xgb_10d/ xgb_15d/      — XGB artifacts (model.joblib + config + per_beach)
    <apps/prediction>/cross_year_results/<timestamp>/
        combined_ranking.csv .md       — cross-dataset combined ranking

Each NF model dir is tft_service-compatible (config.json + nf_model/ +
static_features.csv) and is auto-discovered.

Usage:
    # default — both datasets if available, all models, all horizons, 50 trials
    python scripts/cross_year_train_3_models.py

    # one dataset, smaller budget
    python scripts/cross_year_train_3_models.py \\
        --data-dirs pipeline_workspace/clean_datasets_210526 \\
        --models tft lstm xgb --horizons 3d 10d 15d --trials 30

    # both datasets explicit (new auto-pipeline + super-clean backup)
    python scripts/cross_year_train_3_models.py \\
        --data-dirs pipeline_workspace/clean_datasets_210526 \\
                    pipeline_workspace/clean_dataset_backup \\
        --trials 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

# OpenMP guard — prevent XGBoost segfaulting after PyTorch threads spawn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

SEED = 42
HOURS_PER_DAY = 12
SEASON_MONTHS = {4, 5, 6, 7, 8, 9}
SUMMER_MONTHS = {6, 7, 8}

TARGET_HORIZONS = {"3d": 36, "10d": 120, "15d": 180}

# Shared NF feature schema — matches tft_train_3_models conventions
STAT_EXOG = ["stat_mean_y", "stat_cv"]
TEMPORAL_FUTR = ["hour", "day_of_week", "month", "is_weekend", "is_summer"]
FUTR_WEATHER = ["om_temperature_2m", "om_apparent_temperature", "om_cloud_cover"]
DEFAULT_HIST = ["om_cloud_cover_low", "om_shortwave_radiation",
                "om_vapour_pressure_deficit"]
SELECTED_FUTR = TEMPORAL_FUTR + FUTR_WEATHER

INPUT_SIZE = 48                   # default; can be overridden per trial
NF_TRIAL_MAX_STEPS = 500          # enough training for HP behavior analysis, not just ranking
NF_FINAL_MAX_STEPS = 1500         # full training for the best HP
INPUT_SIZE_POOL = [48, 72, 96, 120]   # daytime-hour context windows
BATCH_SIZE_POOL = [16, 32, 64, 128]    # A6000 has 48 GiB; 128 fits TFT@h=180
OPTUNA_STARTUP_TRIALS = 30        # random trials before TPE kicks in — broader search-space coverage

_OPTUNA_STORAGE_URL: str | None = None
_OPTUNA_RUN_TAG: str = ""


def _set_optuna_storage(path: Path, tag: str) -> str:
    global _OPTUNA_STORAGE_URL, _OPTUNA_RUN_TAG
    path.parent.mkdir(parents=True, exist_ok=True)
    _OPTUNA_STORAGE_URL = f"sqlite:///{path}"
    _OPTUNA_RUN_TAG = tag
    return _OPTUNA_STORAGE_URL


def _study_name(model: str, label: str) -> str:
    return f"{_OPTUNA_RUN_TAG}__{model}__{label}"

# Castelle XGB schema — keep in sync with run_cross_year_unified / xgb_train.
CASTELLE_WEATHER = [
    "om_temperature_2m", "om_precipitation", "om_wind_speed_10m",
    "om_wind_direction_10m", "om_shortwave_radiation",
]
CALENDAR = ["hour", "day_of_week", "month", "is_weekend"]
XGB_FALLBACK_HP = {
    "n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
    "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 3,
    "reg_alpha": 0.01, "reg_lambda": 0.1,
}


def log(msg, level="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] [{level}] {msg}", flush=True)


def make_run_dir(prefix: str, root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run = root / f"{prefix}_{ts}"
    run.mkdir(parents=True, exist_ok=True)
    log(f"Run directory: {run}")
    return run


# ── Data loading ────────────────────────────────────────────────────────────

_UID_ALIASES = {
    "marina":                                  "platja-marina",
    "camp-de-mar":                             "golf-camp-de-mar",
    "port-palma":                              "port",
    "platja-d-or-can-pastilla-desde-bonaona":  "can-pastilla-bonaona",
    "platja-d-or-can-pastilla":                "can-pastilla-mallorca-pipeline",
}


def _slugify_uid(name: str) -> str:
    """Normalise unique_id so 2022 display names match 2025 production slugs."""
    import re, unicodedata
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    s = re.sub(r"[-_]\d+$", "", s)
    return _UID_ALIASES.get(s, s)


def load_csvs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = pd.read_csv(data_dir / "cache_2022_clean.csv", parse_dates=["ds"], low_memory=False)
    django = pd.read_csv(data_dir / "django_2025_clean.csv", parse_dates=["ds"], low_memory=False)

    cache["unique_id"] = cache["unique_id"].map(_slugify_uid)
    django["unique_id"] = django["unique_id"].map(_slugify_uid)
    cache = cache.drop_duplicates(subset=["unique_id", "ds"], keep="first")
    django = django.drop_duplicates(subset=["unique_id", "ds"], keep="first")

    all_clean_path = data_dir / "all_clean.csv"
    if all_clean_path.exists():
        valid = pd.read_csv(all_clean_path)["unique_id"].map(_slugify_uid).unique().tolist()
        cache = cache[cache["unique_id"].isin(valid)]
        django = django[django["unique_id"].isin(valid)]

    log(f"Cache 2022:  {len(cache):,} rows, {cache['unique_id'].nunique()} beaches")
    log(f"Django 2025: {len(django):,} rows, {django['unique_id'].nunique()} beaches")
    return cache, django


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    return df


def build_panels(cache: pd.DataFrame, django: pd.DataFrame,
                  test_start: str, test_end: str):
    """Train = full 2022 cache; Test = django window. Daytime 8-20 on both."""
    train = _calendar(cache.copy())
    train = train.loc[train["hour"].between(8, 20)].dropna(subset=["y"]).copy()
    train["original_id"] = train["unique_id"]
    train["unique_id"] = train["unique_id"].astype(str)
    train["period"] = "cache_2022"

    test = _calendar(django.copy())
    mask = (test["ds"] >= test_start) & (test["ds"] <= test_end) \
            & (test["hour"].between(8, 20))
    test = test.loc[mask].dropna(subset=["y"]).copy()
    test["original_id"] = test["unique_id"]
    test["unique_id"] = test["unique_id"].astype(str)
    test["period"] = "django_2025"

    # Intersect by unique_id so each test beach has training context
    overlap = sorted(set(train["unique_id"]) & set(test["unique_id"]))
    log(f"Beach overlap (train ∩ test): {len(overlap)} → {overlap}")
    if not overlap:
        log("No overlap — cannot evaluate", "ERROR")
        sys.exit(2)
    train = train[train["unique_id"].isin(overlap)].reset_index(drop=True)
    test = test[test["unique_id"].isin(overlap)].reset_index(drop=True)

    # NF needs continuous index — use cumcount integer ds, freq=1 (same trick
    # as tft_train_3_models.py to sidestep the daytime-gap freq="h" problem).
    for d in (train, test):
        d.sort_values(["unique_id", "ds"], inplace=True)
        d["ds_real"] = d["ds"]
        d["ds"] = d.groupby("unique_id").cumcount()
        d.reset_index(drop=True, inplace=True)

    stats = train.groupby("unique_id").agg(
        stat_mean_y=("y", "mean"),
        stat_cv=("y", lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
    ).reset_index()
    static_df = stats[["unique_id"] + STAT_EXOG]

    return train, test, static_df


def compute_capacity(panel: pd.DataFrame) -> dict:
    """P90 per beach on daytime rows (already filtered here)."""
    return panel.groupby("unique_id")["y"].quantile(0.9).to_dict()


# ── Metric ──────────────────────────────────────────────────────────────────

def relmae_per_beach(pred: pd.DataFrame, capacity: dict) -> tuple[float, pd.DataFrame]:
    rows = []
    for uid, sub in pred.groupby("unique_id"):
        if len(sub) < 5:
            continue
        mae = float(np.mean(np.abs(sub["y_true"].values - sub["y_pred"].values)))
        rmse = float(np.sqrt(np.mean((sub["y_true"].values - sub["y_pred"].values) ** 2)))
        try:
            r2 = float(r2_score(sub["y_true"], sub["y_pred"]))
        except Exception:
            r2 = float("nan")
        p90 = max(float(capacity.get(uid, sub["y_true"].quantile(0.9))), 1.0)
        rows.append({"unique_id": uid, "P90": p90, "MAE": mae,
                      "RMSE": rmse, "R2": r2, "relMAE": mae / p90})
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        return float("nan"), rdf
    return float(rdf["relMAE"].mean()), rdf.sort_values("relMAE")


# ── NeuralForecast models (TFT / LSTM) ──────────────────────────────────────

_NF_TRAINER_KWARGS = {
    # Force single-GPU. Lightning otherwise picks up all visible CUDA devices
    # and tries DDP, which crashes when NVML is broken on the host.
    "devices": 1,
    "num_nodes": 1,
    "enable_model_summary": False,
    "enable_progress_bar": False,
    "logger": False,
}


def _valid_n_heads(hidden_size: int, choices=(1, 2, 4, 8)) -> list[int]:
    """Return n_head choices that divide hidden_size (TFT asserts h % n == 0)."""
    return [n for n in choices if hidden_size % n == 0] or [1]


def _build_nf_model(model_name: str, horizon: int, hp: dict,
                    futr: list[str], hist: list[str], max_steps: int):
    from neuralforecast.losses.pytorch import MAE
    input_size = int(hp.get("input_size", INPUT_SIZE))
    if model_name == "tft":
        from neuralforecast.models import TFT
        # Safety: if a stale Optuna trial bypasses the divisibility constraint,
        # snap n_head to the nearest valid divisor of hidden_size.
        hs, nh = int(hp["hidden_size"]), int(hp["n_head"])
        if hs % nh != 0:
            nh = max([n for n in _valid_n_heads(hs) if n <= nh] or [1])
        return TFT(
            h=horizon, input_size=input_size,
            hidden_size=hs, n_head=nh,
            learning_rate=hp["lr"], batch_size=hp.get("batch_size", 32),
            max_steps=max_steps,
            early_stop_patience_steps=hp.get("early_stop_patience", 30),
            scaler_type=hp.get("scaler", "minmax"), loss=MAE(),
            futr_exog_list=futr, hist_exog_list=hist, stat_exog_list=STAT_EXOG,
            dropout=hp["dropout"], attn_dropout=hp.get("attn_dropout", hp["dropout"]),
            val_check_steps=50, random_seed=SEED, start_padding_enabled=True,
            **_NF_TRAINER_KWARGS,
        ), "TFT"
    from neuralforecast.models import LSTM
    return LSTM(
        h=horizon, input_size=input_size,
        encoder_n_layers=hp.get("encoder_n_layers", 2),
        encoder_hidden_size=hp["hidden_size"],
        decoder_hidden_size=hp.get("decoder_hidden_size", hp["hidden_size"]),
        decoder_layers=hp.get("decoder_layers", 2),
        encoder_dropout=hp["dropout"],
        learning_rate=hp["lr"], batch_size=hp.get("batch_size", 32),
        max_steps=max_steps,
        early_stop_patience_steps=hp.get("early_stop_patience", 30),
        scaler_type=hp.get("scaler", "minmax"), loss=MAE(),
        futr_exog_list=futr, hist_exog_list=hist, stat_exog_list=STAT_EXOG,
        val_check_steps=50, random_seed=SEED, start_padding_enabled=True,
        **_NF_TRAINER_KWARGS,
    ), "LSTM"


def _nf_predict_per_beach(nf, col: str, horizon: int, futr: list[str], hist: list[str],
                          context_df: pd.DataFrame, target_df: pd.DataFrame,
                          static_df: pd.DataFrame) -> pd.DataFrame:
    """Per-beach hindcast: context_df tail → futr_df = target_df head."""
    train_cols = ["unique_id", "ds", "y"] + list(dict.fromkeys(futr + hist))
    preds = []
    for uid, beach_ctx in context_df.groupby("unique_id"):
        beach_tgt = target_df[target_df["unique_id"] == uid].head(horizon)
        if len(beach_tgt) < horizon:
            continue
        ctx = beach_ctx.tail(INPUT_SIZE)[train_cols].copy()
        ctx_max_ds = int(ctx["ds"].max())
        futr_df = beach_tgt[["unique_id"] + futr].copy()
        futr_df["ds"] = list(range(ctx_max_ds + 1, ctx_max_ds + 1 + len(futr_df)))
        try:
            out = nf.predict(df=ctx, futr_df=futr_df, static_df=static_df)
        except Exception as e:
            log(f"  [warn] predict failed for {uid}: {e}", "WARN")
            continue
        if col not in out.columns:
            cands = [c for c in out.columns if c not in {"unique_id", "ds"}]
            if not cands:
                continue
            out = out.rename(columns={cands[0]: col})
        out = out[["unique_id", col]].reset_index(drop=True)
        out["y_pred"] = out[col].clip(lower=0)
        out["y_true"] = beach_tgt["y"].values[: len(out)]
        out["ds_real"] = beach_tgt["ds_real"].values[: len(out)]
        preds.append(out[["unique_id", "ds_real", "y_pred", "y_true"]])
    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(
        columns=["unique_id", "ds_real", "y_pred", "y_true"])


def _nf_split_train(train_df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last `horizon` rows of each beach as inner-val."""
    inner_train, inner_val = [], []
    for uid, g in train_df.groupby("unique_id"):
        if len(g) < INPUT_SIZE + horizon + 5:
            continue
        inner_train.append(g.iloc[:-horizon].copy())
        inner_val.append(g.iloc[-(INPUT_SIZE + horizon):].copy())
    if not inner_train:
        return train_df, train_df.tail(0)
    return (pd.concat(inner_train, ignore_index=True),
            pd.concat(inner_val, ignore_index=True))


_HIDDEN_SIZES = [32, 64, 96, 128, 192, 256, 384, 512]
# Common divisors of every entry in _HIDDEN_SIZES — sampled as a flat dimension
# so n_head shows up as a single axis in the Optuna dashboard hyperparameter
# importance chart. Trade-off: 12 and 24 are dropped (they don't divide 32/64/128/256).
_HEAD_POOL = [1, 2, 4, 8, 16, 32]
_SCALERS = ["robust", "standard", "minmax"]


def _nf_objective(trial, model_name, horizon, train_df, val_df, static_df,
                  futr, hist, capacity):
    import optuna
    from neuralforecast import NeuralForecast
    hp = {
        "hidden_size": trial.suggest_categorical("hidden_size", _HIDDEN_SIZES),
        "dropout":     trial.suggest_float("dropout", 0.0, 0.5),
        "lr":          trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        "batch_size":  trial.suggest_categorical("batch_size", BATCH_SIZE_POOL),
        "input_size":  trial.suggest_categorical("input_size", INPUT_SIZE_POOL),
        "scaler":      trial.suggest_categorical("scaler", _SCALERS),
        "early_stop_patience": trial.suggest_int("early_stop_patience", 20, 50),
    }
    if model_name == "tft":
        # Flat single-axis n_head from common divisors of every hidden_size.
        # This keeps n_head visible in the dashboard's parameter-importance plot
        # (instead of being split across n_head_hs64 / n_head_hs128 / ...).
        hp["n_head"] = trial.suggest_categorical("n_head", _HEAD_POOL)
        hp["attn_dropout"] = trial.suggest_float("attn_dropout", 0.0, 0.3)
    else:
        hp["encoder_n_layers"] = trial.suggest_int("encoder_n_layers", 1, 4)
        hp["decoder_layers"]   = trial.suggest_int("decoder_layers", 1, 4)
        # Allow asymmetric encoder/decoder widths (LSTM-specific)
        hp["decoder_hidden_size"] = trial.suggest_categorical(
            "decoder_hidden_size", _HIDDEN_SIZES)
    model, col = _build_nf_model(model_name, horizon, hp, futr, hist, NF_TRIAL_MAX_STEPS)
    nf = NeuralForecast(models=[model], freq=1)
    train_cols = ["unique_id", "ds", "y"] + list(dict.fromkeys(futr + hist))
    try:
        nf.fit(df=train_df[train_cols], static_df=static_df, val_size=horizon)
    except Exception as e:
        log(f"  [trial fail] {e}", "WARN")
        raise optuna.TrialPruned(f"fit failed: {e}")
    pred = _nf_predict_per_beach(nf, col, horizon, futr, hist, train_df, val_df, static_df)
    if pred.empty:
        raise optuna.TrialPruned("empty predictions on inner-val")
    rel, _ = relmae_per_beach(pred, capacity)
    if np.isnan(rel):
        raise optuna.TrialPruned("nan relMAE")
    return float(rel)


def _nf_train_predict(model_name: str, horizon: int, train_df: pd.DataFrame,
                      test_df: pd.DataFrame, static_df: pd.DataFrame,
                      futr: list[str], hist: list[str]) -> pd.DataFrame:
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import MAE

    if model_name == "tft":
        from neuralforecast.models import TFT as ModelCls
        model = ModelCls(
            h=horizon, input_size=INPUT_SIZE,
            hidden_size=FIXED_HP["hidden_size"], n_head=FIXED_HP["n_head"],
            learning_rate=FIXED_HP["lr"], batch_size=32, max_steps=500,
            early_stop_patience_steps=30, scaler_type="minmax", loss=MAE(),
            futr_exog_list=futr or None, hist_exog_list=hist or None,
            stat_exog_list=STAT_EXOG,
            dropout=FIXED_HP["dropout"], attn_dropout=FIXED_HP["dropout"],
            val_check_steps=50, random_seed=SEED, start_padding_enabled=True,
        )
        col = "TFT"
    else:
        from neuralforecast.models import LSTM as ModelCls
        model = ModelCls(
            h=horizon, input_size=INPUT_SIZE,
            encoder_n_layers=2,
            encoder_hidden_size=FIXED_HP["hidden_size"],
            decoder_hidden_size=FIXED_HP["hidden_size"],
            decoder_layers=2,
            encoder_dropout=FIXED_HP["dropout"],
            learning_rate=FIXED_HP["lr"], batch_size=32, max_steps=500,
            early_stop_patience_steps=30, scaler_type="minmax", loss=MAE(),
            futr_exog_list=futr or None, hist_exog_list=hist or None,
            stat_exog_list=STAT_EXOG,
            val_check_steps=50, random_seed=SEED, start_padding_enabled=True,
        )
        col = "LSTM"

    nf = NeuralForecast(models=[model], freq=1)
    train_cols = ["unique_id", "ds", "y"] + list(dict.fromkeys(futr + hist))
    nf.fit(df=train_df[train_cols], static_df=static_df, val_size=horizon)

    # For each beach, predict the first `horizon` test rows using the tail of
    # train as context — train and test share the cumcount-integer index per
    # series, so feeding df=train_tail + futr_df=test_head works.
    preds = []
    for uid, beach_train in train_df.groupby("unique_id"):
        beach_test = test_df[test_df["unique_id"] == uid].head(horizon)
        if len(beach_test) < horizon:
            continue
        ctx = beach_train.tail(INPUT_SIZE)[train_cols]
        # NF expects test ds to continue from train ds; remap test ds onto the
        # train timeline by appending input_size + 1..h.
        ctx_max_ds = int(ctx["ds"].max())
        futr_df = beach_test[["unique_id", "ds"] + futr].copy()
        futr_df["ds"] = list(range(ctx_max_ds + 1, ctx_max_ds + 1 + len(futr_df)))
        try:
            out = nf.predict(df=ctx, futr_df=futr_df, static_df=static_df)
        except Exception as e:
            log(f"  [warn] predict failed for {uid}: {e}", "WARN")
            continue
        if col not in out.columns:
            cands = [c for c in out.columns if c not in {"unique_id", "ds"}]
            if not cands:
                continue
            out = out.rename(columns={cands[0]: col})
        out = out[["unique_id", col]].reset_index(drop=True)
        out["y_pred"] = out[col].clip(lower=0)
        out["y_true"] = beach_test["y"].values[: len(out)]
        out["ds_real"] = beach_test["ds_real"].values[: len(out)]
        preds.append(out[["unique_id", "ds_real", "y_pred", "y_true"]])

    return pd.concat(preds, ignore_index=True) if preds else pd.DataFrame(
        columns=["unique_id", "ds_real", "y_pred", "y_true"])


def save_nf_artifacts(nf_save_path: Path, horizon: int, futr: list[str],
                       hist: list[str], static_df: pd.DataFrame,
                       model_type: str, season_rel: float, overall_rel: float,
                       best_hp: dict, study_name: str):
    nf_save_path.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type":   model_type,
        "horizon":      int(horizon),
        "horizon_days": int(horizon // HOURS_PER_DAY),
        "hours_per_day": HOURS_PER_DAY,
        "input_size":   INPUT_SIZE,
        "futr_exog":    futr,
        "hist_exog":    hist,
        "stat_exog":    STAT_EXOG,
        "hp":           best_hp,
        "optuna_study": study_name,
        "regime":       "cross_year_train_2022",
        "season_relMAE": float(season_rel) if not np.isnan(season_rel) else None,
        "overall_relMAE": float(overall_rel) if not np.isnan(overall_rel) else None,
        "trained_at":   datetime.now().isoformat(),
    }
    (nf_save_path / "config.json").write_text(json.dumps(config, indent=2))
    (nf_save_path / "best_params.json").write_text(json.dumps(best_hp, indent=2))
    static_df.to_csv(nf_save_path / "static_features.csv", index=False)


def run_nf_model(model_name: str, label: str, horizon: int,
                  train_df: pd.DataFrame, test_df: pd.DataFrame,
                  static_df: pd.DataFrame, capacity: dict, run_dir: Path,
                  trials: int) -> dict | None:
    import optuna
    from neuralforecast import NeuralForecast
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    log(f"[{model_name}] H={horizon} ({label})  trials={trials}")

    futr = [c for c in SELECTED_FUTR if c in train_df.columns]
    hist = [c for c in DEFAULT_HIST if c in train_df.columns]

    inner_train, inner_val = _nf_split_train(train_df, horizon)
    if inner_val.empty:
        log(f"[{model_name}] not enough rows per beach for inner val", "ERROR")
        return None

    study_name = _study_name(model_name, label)
    study = optuna.create_study(
        study_name=study_name, storage=_OPTUNA_STORAGE_URL,
        direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=OPTUNA_STARTUP_TRIALS),
    )
    study.optimize(
        lambda t: _nf_objective(t, model_name, horizon, inner_train, inner_val,
                                 static_df, futr, hist, capacity),
        n_trials=trials, show_progress_bar=False,
    )
    best_hp = study.best_params
    log(f"[{model_name}] best inner relMAE={study.best_value:.4f}  hp={best_hp}")

    # Refit on full train with best HP, full max_steps, then predict on test
    model, col = _build_nf_model(model_name, horizon, best_hp, futr, hist,
                                  NF_FINAL_MAX_STEPS)
    nf = NeuralForecast(models=[model], freq=1)
    train_cols = ["unique_id", "ds", "y"] + list(dict.fromkeys(futr + hist))
    nf.fit(df=train_df[train_cols], static_df=static_df, val_size=horizon)

    pred_df = _nf_predict_per_beach(nf, col, horizon, futr, hist, train_df,
                                     test_df, static_df)
    if pred_df.empty:
        log(f"[{model_name}] no predictions produced", "ERROR")
        return None

    overall_rel, pb = relmae_per_beach(pred_df, capacity)
    pred_df["month"] = pd.to_datetime(pred_df["ds_real"]).dt.month
    season_rel = relmae_per_beach(pred_df[pred_df["month"].isin(SEASON_MONTHS)],
                                    capacity)[0]
    summer_rel = relmae_per_beach(pred_df[pred_df["month"].isin(SUMMER_MONTHS)],
                                    capacity)[0]
    mae = float(mean_absolute_error(pred_df["y_true"], pred_df["y_pred"]))
    log(f"[{model_name}] overall={overall_rel:.4f}  season={season_rel:.4f}  "
        f"summer={summer_rel:.4f}  MAE={mae:.1f}  n={len(pred_df)}")

    model_dir = run_dir / f"{model_name}_{label}"
    nf.save(str(model_dir / "nf_model"), overwrite=True, save_dataset=False)
    save_nf_artifacts(model_dir, horizon, futr, hist, static_df,
                       "TFT" if model_name == "tft" else "LSTM",
                       season_rel, overall_rel, best_hp, study_name)
    pred_df.to_csv(model_dir / "cv_predictions.csv", index=False)
    pb.to_csv(model_dir / "per_beach.csv", index=False)
    pd.DataFrame([{**t.params, "value": t.value, "state": str(t.state)}
                   for t in study.trials]).to_csv(
        model_dir / "optuna_trials.csv", index=False)
    return {"model": model_name, "horizon": label, "n_rows": len(pred_df),
             "relMAE_overall": overall_rel, "relMAE_season": season_rel,
             "relMAE_summer": summer_rel, "MAE": mae}


# ── XGBoost ─────────────────────────────────────────────────────────────────

def build_xgb_features(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    df = df.copy().sort_values(["unique_id", "ds_real"]).reset_index(drop=True)
    df["y_lag1"] = df.groupby("unique_id")["y"].shift(1)
    df["y_roll12"] = df.groupby("unique_id")["y"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    df["y_roll36"] = df.groupby("unique_id")["y"].transform(
        lambda s: s.shift(1).rolling(36, min_periods=1).mean())

    ahead = [c for c in (CASTELLE_WEATHER + CALENDAR) if c in df.columns]
    df_fut = df.groupby("unique_id")[ahead].shift(-horizon)
    df_fut.columns = [f"{c}_t_plus_h" for c in ahead]

    df["y_target"] = df.groupby("unique_id")["y"].shift(-horizon)
    df["ds_target"] = df.groupby("unique_id")["ds_real"].shift(-horizon)

    out = pd.concat([
        df[["unique_id", "ds_real", "ds_target", "y_lag1", "y_roll12", "y_roll36"]],
        df_fut,
        df[["y_target"]],
    ], axis=1)
    return out.dropna(subset=["y_target", "y_lag1"]).reset_index(drop=True)


def run_xgb(label: str, horizon: int, train_df: pd.DataFrame,
             test_df: pd.DataFrame, capacity: dict, run_dir: Path,
             trials: int) -> dict | None:
    log(f"[xgb] H={horizon} ({label})  trials={trials}")
    import optuna
    from xgboost import XGBRegressor
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    train_feat = build_xgb_features(train_df, horizon)
    test_feat = build_xgb_features(test_df, horizon)

    feat_cols = [c for c in train_feat.columns
                  if c not in {"unique_id", "ds_real", "ds_target", "y_target"}]
    if not len(train_feat) or not len(test_feat):
        log("[xgb] empty feature matrix", "ERROR")
        return None
    X_tr = train_feat[feat_cols].values
    y_tr = train_feat["y_target"].values
    X_te = test_feat[feat_cols].values
    y_te = test_feat["y_target"].values

    # 90/10 split within train for HP search — keep unique_id so we can score
    # with relmae_per_beach (same metric as TFT/LSTM) instead of raw MAE.
    cut = int(len(train_feat) * 0.9)
    val_uid = train_feat["unique_id"].values[cut:]
    X_a, X_b = X_tr[:cut], X_tr[cut:]
    y_a, y_b = y_tr[:cut], y_tr[cut:]

    def objective(trial):
        hp = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 2000, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 15),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.5, log=True),
            "subsample":        trial.suggest_float("subsample", 0.4, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma":            trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        m = XGBRegressor(**hp, tree_method="hist", n_jobs=-1,
                          random_state=SEED, verbosity=0)
        try:
            m.fit(X_a, y_a, verbose=False)
        except Exception as e:
            log(f"  [trial fail] {e}", "WARN")
            raise optuna.TrialPruned(f"xgb fit failed: {e}")
        pred = m.predict(X_b).clip(0, None)
        # Same metric as TFT/LSTM objective: P90-normalised relMAE per beach.
        val_pred_df = pd.DataFrame({
            "unique_id": val_uid, "y_true": y_b, "y_pred": pred,
        })
        rel, _ = relmae_per_beach(val_pred_df, capacity)
        if np.isnan(rel):
            raise optuna.TrialPruned("nan relMAE")
        return float(rel)

    study_name = _study_name("xgb", label)
    study = optuna.create_study(
        study_name=study_name, storage=_OPTUNA_STORAGE_URL,
        direction="minimize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=OPTUNA_STARTUP_TRIALS),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_hp = study.best_params
    log(f"[xgb] best inner relMAE={study.best_value:.4f}  hp={best_hp}")

    final = XGBRegressor(**best_hp, tree_method="hist", n_jobs=-1,
                          random_state=SEED, verbosity=0)
    final.fit(X_tr, y_tr, verbose=False)
    pred = final.predict(X_te).clip(0, None)

    pred_df = pd.DataFrame({
        "unique_id": test_feat["unique_id"].values,
        "ds_real":   test_feat["ds_target"].values,
        "y_pred":    pred,
        "y_true":    y_te,
    })
    overall_rel, pb = relmae_per_beach(pred_df, capacity)
    pred_df["month"] = pd.to_datetime(pred_df["ds_real"]).dt.month
    season_rel = relmae_per_beach(pred_df[pred_df["month"].isin(SEASON_MONTHS)],
                                    capacity)[0]
    summer_rel = relmae_per_beach(pred_df[pred_df["month"].isin(SUMMER_MONTHS)],
                                    capacity)[0]
    mae = float(mean_absolute_error(pred_df["y_true"], pred_df["y_pred"]))
    log(f"[xgb] overall={overall_rel:.4f}  season={season_rel:.4f}  "
        f"summer={summer_rel:.4f}  MAE={mae:.1f}  n={len(pred_df)}")

    model_dir = run_dir / f"xgb_{label}"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, model_dir / "model.joblib")
    (model_dir / "best_params.json").write_text(json.dumps(best_hp, indent=2))
    (model_dir / "config.json").write_text(json.dumps({
        "model_type":     "XGBRegressor",
        "horizon":        int(horizon),
        "horizon_days":   int(horizon // HOURS_PER_DAY),
        "features":       feat_cols,
        "regime":         "cross_year_train_2022",
        "schema":         "castelle_2025",
        "hp":             best_hp,
        "optuna_study":   study_name,
        "overall_relMAE": float(overall_rel) if not np.isnan(overall_rel) else None,
        "season_relMAE":  float(season_rel) if not np.isnan(season_rel) else None,
        "summer_relMAE":  float(summer_rel) if not np.isnan(summer_rel) else None,
        "trained_at":     datetime.now().isoformat(),
    }, indent=2))
    pred_df.to_csv(model_dir / "cv_predictions.csv", index=False)
    pb.to_csv(model_dir / "per_beach.csv", index=False)
    pd.DataFrame([{**t.params, "value": t.value, "state": str(t.state)}
                   for t in study.trials]).to_csv(
        model_dir / "optuna_trials.csv", index=False)

    return {"model": "xgb", "horizon": label, "n_rows": len(pred_df),
             "relMAE_overall": overall_rel, "relMAE_season": season_rel,
             "relMAE_summer": summer_rel, "MAE": mae}


# ── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Cross-Year Training — 2022 train, 2025 test")
    # Backwards-compatible single dir; new --data-dirs accepts many.
    p.add_argument("--data-dir", default=None,
                    help="Single data dir (alias of --data-dirs with one entry).")
    p.add_argument("--data-dirs", nargs="+", default=None,
                    help="One or more data dirs to train and validate against. "
                          "Each dir produces its own run with the SAME fairness "
                          "protocol (same models, horizons, HP budget, validation "
                          "window). Default: pipeline_workspace/clean_datasets_210526 "
                          "+ pipeline_workspace/clean_dataset_backup if both exist.")
    p.add_argument("--models", nargs="*", default=["tft", "lstm", "xgb"],
                    choices=["tft", "lstm", "xgb"])
    p.add_argument("--horizons", nargs="*", default=list(TARGET_HORIZONS),
                    choices=list(TARGET_HORIZONS))
    p.add_argument("--test-start", default="2025-06-01")
    p.add_argument("--test-end",   default="2025-08-31")
    p.add_argument("--trials", type=int, default=50,
                    help="Optuna trials per (model, horizon). Default 50.")
    p.add_argument("--prefix", default="cross_year",
                    help="Run directory prefix (default: cross_year)")
    p.add_argument("--out-root", default=None,
                    help="Parent directory for the run dir. Defaults to "
                          "apps/prediction/tft_models so tft_service auto-discovers TFT/LSTM. "
                          "XGB lands in the sibling xgb_models/ regardless.")
    p.add_argument("--storage-url", default=None,
                    help="Optuna storage URL (default: sqlite at apps/prediction/optuna/cross_year.db). "
                          "Use optuna-dashboard <url> to visualize.")
    return p.parse_args()


def _resolve_dataset_list(args) -> list[Path]:
    """Combine --data-dir + --data-dirs into a deduplicated list of existing dirs."""
    candidates: list[str] = []
    if args.data_dirs:
        candidates.extend(args.data_dirs)
    if args.data_dir:
        candidates.append(args.data_dir)
    if not candidates:
        candidates = ["pipeline_workspace/clean_datasets_210526",
                       "pipeline_workspace/clean_dataset_backup"]
    resolved: list[Path] = []
    seen: set[str] = set()
    for c in candidates:
        p = _resolve_data_dir(c)
        if p.exists() and str(p) not in seen:
            resolved.append(p)
            seen.add(str(p))
    if not resolved:
        sys.exit(f"[fatal] no data dirs found among: {candidates}")
    return resolved


def _dataset_tag(dataset_path: Path) -> str:
    """Short tag for the dataset folder, used in run-name + study-name."""
    return dataset_path.name.replace("clean_datasets_", "").replace("clean_dataset_", "")


def _resolve_data_dir(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return pp.resolve()
    here = Path(__file__).resolve().parent.parent  # apps/prediction
    for c in (pp, here / pp, here.parent / pp):
        if c.exists():
            return c.resolve()
    return (here / pp).resolve()


def _write_protocol_json(path: Path, args, data_dir: Path, dataset_tag: str,
                          train_df, test_df, capacity: dict, storage_url: str,
                          set_name: str):
    """Record the fairness invariants for this run, as an audit trail.

    Every model family inside this run was trained and evaluated under the
    *same* protocol — same dataset, same beach overlap, same train/test rows,
    same horizons, same Optuna budget. This file makes that explicit so the
    methodology is reviewable.
    """
    invariants = {
        "set_name": set_name,
        "dataset_tag": dataset_tag,
        "dataset_path": str(data_dir),
        "models": args.models,
        "horizons": args.horizons,
        "horizon_hours": {k: TARGET_HORIZONS[k] for k in args.horizons},
        "regime": "train on cache_2022 (full year, daytime 8-20); "
                   "test on django_2025 window",
        "test_window": {"start": args.test_start, "end": args.test_end},
        "n_train_rows": int(len(train_df)),
        "n_test_rows": int(len(test_df)),
        "n_beaches": int(train_df["unique_id"].nunique()),
        "beaches": sorted(train_df["unique_id"].unique().tolist()),
        "capacity_p90_per_beach": {k: float(v) for k, v in capacity.items()},
        "normalisation": "P90 per series (daytime only)",
        "seed": SEED,
        "input_size_default": INPUT_SIZE,
        "input_size_search_pool": INPUT_SIZE_POOL,
        "batch_size_search_pool": BATCH_SIZE_POOL,
        "hidden_size_search_pool": _HIDDEN_SIZES,
        "n_head_search_pool": _HEAD_POOL,
        "scaler_search_pool": _SCALERS,
        "lr_search_range": [1e-5, 1e-2],
        "dropout_search_range": [0.0, 0.5],
        "optuna_trials_per_model_horizon": args.trials,
        "nf_trial_max_steps": NF_TRIAL_MAX_STEPS,
        "nf_final_max_steps": NF_FINAL_MAX_STEPS,
        "optuna_storage": storage_url,
        "primary_significance_test": "Diebold-Mariano (Wilcoxon as sanity check)",
        "started_at": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(invariants, indent=2))


def run_one_dataset(args, data_dir: Path, here: Path, nf_root: Path,
                     xgb_root: Path) -> tuple[pd.DataFrame, Path]:
    """Train + validate all (model, horizon) combos on a single dataset.

    The fairness invariants for this run are written to protocol.json.
    Returns (summary_df, nf_run_dir).
    """
    dataset_tag = _dataset_tag(data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    set_name = f"{args.prefix}_{dataset_tag}_{timestamp}"
    nf_run = nf_root / set_name
    xgb_run = xgb_root / set_name
    nf_run.mkdir(parents=True, exist_ok=True)
    xgb_run.mkdir(parents=True, exist_ok=True)
    results_dir = nf_run / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Optuna storage: same DB across datasets; study name carries dataset_tag
    # so dashboard shows one study per (dataset, model, horizon) — every cell
    # of the comparison gets its own searchable history.
    if args.storage_url:
        global _OPTUNA_STORAGE_URL, _OPTUNA_RUN_TAG
        _OPTUNA_STORAGE_URL = args.storage_url
        _OPTUNA_RUN_TAG = set_name
        storage_url = args.storage_url
    else:
        db_path = here / "optuna" / "cross_year.db"
        storage_url = _set_optuna_storage(db_path, set_name)

    log("=" * 72)
    log(f"DATASET RUN — {dataset_tag}")
    log("=" * 72)
    log(f"Data dir:  {data_dir}")
    log(f"Models:    {args.models}    Horizons: {args.horizons}")
    log(f"Test:      {args.test_start} → {args.test_end}")
    log(f"Trials:    {args.trials} per (model, horizon)")
    log(f"NF set:    {nf_run}")
    log(f"XGB set:   {xgb_run}")
    log(f"Optuna:    {storage_url}")

    cache_df, django_df = load_csvs(data_dir)
    train_df, test_df, static_df = build_panels(
        cache_df, django_df, args.test_start, args.test_end)
    capacity = compute_capacity(train_df)
    log(f"Train rows: {len(train_df):,}  Test rows: {len(test_df):,}  "
        f"Beaches: {train_df['unique_id'].nunique()}")

    # Fairness audit: same train/test split, same beach overlap, same HP budget
    # applies to ALL models below. Recorded so methodology is reviewable.
    _write_protocol_json(results_dir / "protocol.json", args, data_dir,
                          dataset_tag, train_df, test_df, capacity,
                          storage_url, set_name)
    log(f"Protocol:  {results_dir/'protocol.json'}")
    log("Fairness: same dataset, same beach overlap, same train/test rows, "
        f"same Optuna trials ({args.trials}) for every (model, horizon).")

    summary = []
    for label in args.horizons:
        horizon = TARGET_HORIZONS[label]
        log("-" * 70)
        log(f"--- HORIZON {label} (H={horizon}) on {dataset_tag} ---")
        log("-" * 70)

        for m in args.models:
            t0 = time.time()
            try:
                if m == "xgb":
                    r = run_xgb(label, horizon, train_df, test_df, capacity,
                                 xgb_run, args.trials)
                else:
                    r = run_nf_model(m, label, horizon, train_df, test_df,
                                      static_df, capacity, nf_run, args.trials)
                if r:
                    r["elapsed_s"] = round(time.time() - t0, 1)
                    r["dataset"] = dataset_tag
                    summary.append(r)
            except Exception as e:
                log(f"[{m}/{label}] FAILED on {dataset_tag}: {e}", "ERROR")
                traceback.print_exc()

    if not summary:
        log(f"No successful runs on {dataset_tag}", "ERROR")
        return pd.DataFrame(), nf_run

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(results_dir / "summary.csv", index=False)
    ranking = summary_df.sort_values(["horizon", "relMAE_summer"]).reset_index(drop=True)
    ranking.to_csv(results_dir / "ranking.csv", index=False)

    md = ["# Cross-year ranking — " + dataset_tag, "",
          "## Fairness protocol",
          "- Same dataset for all three model families.",
          "- Same beach overlap (train 2022 ∩ test 2025 summer).",
          "- Same train rows, same test rows.",
          f"- Same Optuna budget: {args.trials} trials per (model, horizon).",
          "- Same validation window: " + args.test_start + " → " + args.test_end + ".",
          "- Same metric: P90-normalised relMAE on daytime hours.",
          "",
          "## Cells",
          f"- Train: cache_2022 (full year, daytime 8-20) — {len(train_df):,} rows",
          f"- Test:  django_2025 {args.test_start} → {args.test_end} — {len(test_df):,} rows",
          f"- Beaches: {train_df['unique_id'].nunique()}",
          "",
          ranking.to_markdown(index=False, floatfmt=".4f")]
    (results_dir / "ranking.md").write_text("\n".join(md))
    log(f"Outputs:\n  NF set:  {nf_run}\n  XGB set: {xgb_run}\n  "
        f"Protocol: {results_dir/'protocol.json'}")
    return summary_df, nf_run


def main():
    args = parse_args()
    datasets = _resolve_dataset_list(args)

    # Default out_root = apps/prediction/tft_models (so tft_service picks up TFT+LSTM)
    here = Path(__file__).resolve().parent.parent
    nf_root = Path(args.out_root) if args.out_root else here / "tft_models"
    xgb_root = here / "xgb_models"
    nf_root.mkdir(parents=True, exist_ok=True)
    xgb_root.mkdir(parents=True, exist_ok=True)

    log("Cross-Year Training — 2022 train, 2025 test, three model families")
    log(f"Datasets ({len(datasets)}): {[d.name for d in datasets]}")
    log(f"Models:   {args.models}")
    log(f"Horizons: {args.horizons}")
    log(f"Test:     {args.test_start} → {args.test_end}")
    log(f"Trials:   {args.trials} per (model, horizon)")
    log("Fairness invariants across the entire run:")
    log("  * Identical model list, horizons, optuna budget on every dataset.")
    log("  * Per dataset: identical train/test split for TFT, LSTM and XGB.")
    log("  * Identical normalisation (P90 per series) and metric (relMAE).")

    all_summaries: list[pd.DataFrame] = []
    run_dirs: list[Path] = []
    for data_dir in datasets:
        df, run_dir = run_one_dataset(args, data_dir, here, nf_root, xgb_root)
        if not df.empty:
            all_summaries.append(df)
            run_dirs.append(run_dir)

    if not all_summaries:
        log("No successful runs on any dataset", "ERROR")
        return

    # Combined cross-dataset comparison
    combined = pd.concat(all_summaries, ignore_index=True)
    combined_sorted = combined.sort_values(["horizon", "dataset", "relMAE_summer"]) \
                                .reset_index(drop=True)
    cross_dir = here / "cross_year_results" / datetime.now().strftime("%Y%m%d_%H%M%S")
    cross_dir.mkdir(parents=True, exist_ok=True)
    combined_sorted.to_csv(cross_dir / "combined_ranking.csv", index=False)
    md = ["# Cross-dataset ranking", "",
          "All cells trained and validated under the same protocol "
          "(see each run's protocol.json for the invariants).",
          "",
          combined_sorted.to_markdown(index=False, floatfmt=".4f")]
    (cross_dir / "combined_ranking.md").write_text("\n".join(md))

    log("\n" + "=" * 72)
    log("CROSS-DATASET RANKING (horizon, then dataset, then summer relMAE)")
    log("=" * 72)
    print(combined_sorted.to_string(index=False))
    log(f"Combined output: {cross_dir/'combined_ranking.csv'}")


if __name__ == "__main__":
    main()
