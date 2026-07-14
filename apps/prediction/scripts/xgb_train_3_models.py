#!/usr/bin/env python3
"""
XGBoost Hourly Training — Castelle 2025 schema, per-horizon models.

Mirrors the phase structure of tft_train_3_models.py but with an XGBRegressor
trained on shifted-feature Castelle vectors. The deliverable for each horizon
is a self-contained directory loadable by a future xgb_service (same shape as
the cross_year_unified.py XGB save: model.joblib + config.json + per_beach.csv).

Phases:
    Per horizon (3d, 10d, 15d):
        Phase 1: hist-feature backward elimination (drop weakest until 3 remain
                 or relMAE > 2x baseline)
        Phase 3: Optuna HP search (n_estimators, max_depth, lr, subsample,
                 colsample_bytree, min_child_weight, reg_alpha, reg_lambda)
        Phase 4: refit on the full panel, save model + per-beach metrics

Output structure:
    <prefix>_<timestamp>/
        results/             — CSVs, checkpoints
        xgb_model_3d/        — model.joblib + config.json + per_beach.csv + best_params.json
        xgb_model_10d/
        xgb_model_15d/

Default save root is `xgb_models/` (sibling of TFT_MODELS_DIR) when run via the
beachcamweb tree, or the cwd otherwise.

Usage:
    python xgb_train_3_models.py --data-dir clean_datasets
    python xgb_train_3_models.py --deploy-only           # skip HP search, use fallback HP
    python xgb_train_3_models.py --trials 50             # Optuna trials per horizon
    python xgb_train_3_models.py --eval-dirs clean_dataset_backup
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from crowd_outliers import cap_outliers  # canonical per-series P90 outlier cap

warnings.filterwarnings("ignore")

SEED = 42
SEASON_EVAL = [4, 5, 6, 7, 8, 9]
HOURS_PER_DAY = 13

TARGET_HORIZONS = {"3d": 39, "10d": 130, "15d": 195}

# Castelle 2025 schema — keep aligned with run_cross_year_unified.py
CASTELLE_WEATHER = [
    "om_temperature_2m", "om_precipitation", "om_wind_speed_10m",
    "om_wind_direction_10m", "om_shortwave_radiation",
]
CALENDAR = ["hour", "day_of_week", "month", "is_weekend"]

# Fallback HP for --deploy-only and as the Optuna starting point.
FALLBACK_HP = {
    "n_estimators":     500,
    "max_depth":        6,
    "learning_rate":    0.05,
    "subsample":        0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha":        0.01,
    "reg_lambda":       0.1,
}


def log(msg, level="INFO"):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _prediction_dir() -> Path:
    """Path of apps/prediction (script lives in apps/prediction/scripts)."""
    return Path(__file__).resolve().parent.parent


def _default_xgb_root() -> str:
    return str(_prediction_dir() / "xgb_models")


def _resolve_data_dir(p: str) -> str:
    """Resolve --data-dir relative to apps/prediction so the script works from any cwd."""
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return str(pp)
    candidates = [pp, _prediction_dir() / pp, _prediction_dir().parent / pp]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    return str(_prediction_dir() / pp)


def make_run_dir(prefix, root=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = Path(root) if root else Path(_default_xgb_root())
    parent.mkdir(parents=True, exist_ok=True)
    run_dir = parent / f"{prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f"Run directory: {run_dir}")
    return str(run_dir)


def save_checkpoint(results_dir, name, data):
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/_checkpoint_{name}.json", "w") as f:
        json.dump(data, f, indent=2, default=str)


# ── Data loading (mirrors tft_train_3_models.py) ────────────────────────────

def load_data(data_dir):
    log("Loading data...")
    cache_df = pd.read_csv(f"{data_dir}/cache_2022_clean.csv", parse_dates=["ds"])
    django_df = pd.read_csv(f"{data_dir}/django_2025_clean.csv", parse_dates=["ds"])

    all_clean_path = f"{data_dir}/all_clean.csv"
    if os.path.exists(all_clean_path):
        all_clean = pd.read_csv(all_clean_path)
        valid = all_clean["unique_id"].unique().tolist()
        cache_df = cache_df[cache_df["unique_id"].isin(valid)]
        django_df = django_df[django_df["unique_id"].isin(valid)]
        log(f"Filtered to {len(valid)} quality beaches")

    log(f"Cache 2022:  {len(cache_df):,} rows, {cache_df['unique_id'].nunique()} beaches")
    log(f"Django 2025: {len(django_df):,} rows, {django_df['unique_id'].nunique()} beaches")
    return cache_df, django_df


def build_panel(cache, django, season_months=None):
    """Concatenate cache_2022 + django_2025 with calendar features, daytime-only."""
    parts = []
    if cache is not None:
        c = cache.copy()
        c["period"], c["original_id"] = "cache_2022", c["unique_id"]
        c["unique_id"] = c["unique_id"] + "_2022"
        parts.append(c)
    if django is not None:
        d = django.copy()
        d["period"], d["original_id"] = "django_2025", d["unique_id"]
        d["unique_id"] = d["unique_id"] + "_2025"
        parts.append(d)

    df = pd.concat(parts, ignore_index=True).dropna(subset=["y"])
    df, _ = cap_outliers(df)          # per-series P90 outlier cap on the real y
    df["hour"] = df["ds"].dt.hour
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

    if season_months is not None:
        df = df[df["month"].isin(season_months)].copy()

    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    counts = df.groupby("unique_id").size()
    df = df[df["unique_id"].isin(counts[counts >= 96].index)]
    return df


def build_xgb_features(df: pd.DataFrame, horizon_hours: int,
                       hist_cols: list[str]) -> pd.DataFrame:
    """Castelle-style shifted-feature matrix: target is y[t+h].

    Always includes y_lag1, y_roll12, y_roll36 plus the *_t_plus_h shift of any
    column in (CASTELLE_WEATHER + CALENDAR) that's present, restricted to the
    hist_cols allowlist (this lets Phase 1 do backward elimination).
    """
    df = df.copy().sort_values(["unique_id", "ds"]).reset_index(drop=True)
    df["y_lag1"] = df.groupby("unique_id")["y"].shift(1)
    df["y_roll12"] = df.groupby("unique_id")["y"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    df["y_roll36"] = df.groupby("unique_id")["y"].transform(
        lambda s: s.shift(1).rolling(36, min_periods=1).mean())

    ahead_cols = [c for c in (CASTELLE_WEATHER + CALENDAR)
                   if c in df.columns and c in hist_cols]
    df_future = df.groupby("unique_id")[ahead_cols].shift(-horizon_hours)
    df_future.columns = [f"{c}_t_plus_h" for c in ahead_cols]

    df["y_target"] = df.groupby("unique_id")["y"].shift(-horizon_hours)
    df["ds_target"] = df.groupby("unique_id")["ds"].shift(-horizon_hours)

    out = pd.concat([
        df[["unique_id", "ds", "ds_target", "period", "y_lag1", "y_roll12", "y_roll36"]],
        df_future,
        df[["y", "y_target"]],
    ], axis=1)
    return out.dropna(subset=["y_target", "y_lag1"]).reset_index(drop=True)


# ── Metric ──────────────────────────────────────────────────────────────────

def compute_rel_mae(pred_df: pd.DataFrame, capacity: dict[str, float]) -> tuple[float, pd.DataFrame]:
    rows = []
    for uid, sub in pred_df.groupby("unique_id"):
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


# ── Train + cross-validate one fold ─────────────────────────────────────────

def _xgb_fit_predict(train_x, train_y, val_x, val_y, hp):
    from xgboost import XGBRegressor
    params = dict(hp)
    params.update({"tree_method": "hist", "n_jobs": -1,
                    "random_state": SEED, "verbosity": 0})
    model = XGBRegressor(**params)
    model.fit(train_x, train_y, verbose=False)
    pred_val = model.predict(val_x).clip(0, None)
    return model, pred_val


def cv_evaluate(features: pd.DataFrame, feat_cols: list[str], hp: dict,
                n_splits: int = 3) -> tuple[float, pd.DataFrame]:
    """Forward-chaining time-series CV. Returns (mean_relMAE, predictions_df).

    The returned df carries unique_id + ds_target so callers can do season
    filtering without re-merging.
    """
    missing = [c for c in feat_cols if c not in features.columns]
    if missing:
        raise KeyError(f"feat_cols missing from features: {missing}")
    features = features.sort_values(["unique_id", "ds_target"]).reset_index(drop=True)
    splits = np.array_split(features.index, n_splits + 1)
    cap = features.groupby("unique_id")["y_target"].quantile(0.9).to_dict()

    fold_pred = []
    for i in range(n_splits):
        train_idx = np.concatenate(splits[: i + 1])
        val_idx = splits[i + 1]
        train_x = features.loc[train_idx, feat_cols].values
        train_y = features.loc[train_idx, "y_target"].values
        val_x = features.loc[val_idx, feat_cols].values
        val_y = features.loc[val_idx, "y_target"].values
        if len(train_x) < 100 or len(val_x) < 20:
            continue
        _, pred = _xgb_fit_predict(train_x, train_y, val_x, val_y, hp)
        fold_pred.append(pd.DataFrame({
            "unique_id": features.loc[val_idx, "unique_id"].values,
            "ds_target": features.loc[val_idx, "ds_target"].values,
            "y_true":    val_y,
            "y_pred":    pred,
        }))
    if not fold_pred:
        return float("nan"), pd.DataFrame()
    cv_df = pd.concat(fold_pred, ignore_index=True)
    return compute_rel_mae(cv_df, cap)[0], cv_df


# ── Phase 1: hist feature selection ─────────────────────────────────────────

def phase1(features: pd.DataFrame, hist_pool: list[str],
            horizon: int, results_dir: str) -> list[str]:
    """Backward elimination on the hist features.

    `hist_pool` contains the raw column names ('om_temperature_2m', etc.). The
    features dataframe carries them as '<name>_t_plus_h' — we map between the
    two as needed so the rest of the pipeline keeps using the raw names.
    """
    log(f"PHASE 1: Backward elimination (H={horizon})")
    static_cols = [c for c in ("y_lag1", "y_roll12", "y_roll36") if c in features.columns]

    def _feat_cols(hist: list[str]) -> list[str]:
        return static_cols + [f"{h}_t_plus_h" for h in hist if f"{h}_t_plus_h" in features.columns]

    cur_hist = [h for h in hist_pool if f"{h}_t_plus_h" in features.columns]
    base_rel, _ = cv_evaluate(features, _feat_cols(cur_hist), FALLBACK_HP)
    log(f"  Baseline relMAE={base_rel:.4f}  (n_hist={len(cur_hist)})")
    if np.isnan(base_rel):
        log("  Baseline failed — keeping full feature set", "WARN")
        return cur_hist
    stop_threshold = base_rel * 2.0

    best_rel, best_hist = base_rel, cur_hist.copy()
    flog = [{"step": 0, "n_hist": len(cur_hist), "relMAE": base_rel, "dropped": "-"}]
    step = 0
    while len(cur_hist) > 3:
        step += 1
        best_drop, best_drop_rel = None, float("inf")
        for f in cur_hist:
            trial_hist = [h for h in cur_hist if h != f]
            rel, _ = cv_evaluate(features, _feat_cols(trial_hist), FALLBACK_HP)
            if rel < best_drop_rel:
                best_drop_rel, best_drop = rel, f
        cur_hist = [h for h in cur_hist if h != best_drop]
        flog.append({"step": step, "n_hist": len(cur_hist),
                      "relMAE": best_drop_rel, "dropped": best_drop})
        log(f"  Step {step}: drop {best_drop!r} -> relMAE={best_drop_rel:.4f}")
        if best_drop_rel < best_rel:
            best_rel, best_hist = best_drop_rel, cur_hist.copy()
        if best_drop_rel > stop_threshold:
            log("  relMAE > 2x baseline -> stopping")
            break

    pd.DataFrame(flog).to_csv(f"{results_dir}/feature_selection_H{horizon}.csv", index=False)
    log(f"  SELECTED HIST ({len(best_hist)}): {best_hist}")
    return best_hist


# ── Phase 3: Optuna HP search ───────────────────────────────────────────────

def phase3(features: pd.DataFrame, feat_cols: list[str], horizon: int,
            trials: int, results_dir: str) -> dict:
    log(f"PHASE 3: Optuna HP sweep (H={horizon}, trials={trials})")
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        hp = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
        }
        rel, _ = cv_evaluate(features, feat_cols, hp, n_splits=2)
        return rel if not np.isnan(rel) else 1e6

    study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_hp = study.best_params
    log(f"  OPTIMAL H={horizon}: relMAE={study.best_value:.4f}  hp={best_hp}")

    pd.DataFrame([{**t.params, "value": t.value} for t in study.trials]) \
        .to_csv(f"{results_dir}/hp_search_H{horizon}.csv", index=False)
    return best_hp


# ── Phase 4: deploy ─────────────────────────────────────────────────────────

def phase4(label, horizon, cache_df, django_df, hist_pool, feat_cols, best_hp,
            run_dir, results_dir) -> dict | None:
    model_dir = f"{run_dir}/xgb_model_{label}"
    os.makedirs(model_dir, exist_ok=True)
    days = horizon / HOURS_PER_DAY
    log(f"  === PHASE 4: Deploy {label} (H={horizon}, {days:.0f} days) ===")

    try:
        panel = build_panel(cache_df, django_df)
        features = build_xgb_features(panel, horizon, hist_pool)
        features = features.dropna(subset=feat_cols + ["y_target"])
        if len(features) < 100:
            log("  Not enough features after dropna — skipping", "ERROR")
            return None
        cap = panel.groupby("unique_id")["y"].quantile(0.9).to_dict()

        # CV pass for honest in-sample performance
        cv_rel, cv_df = cv_evaluate(features, feat_cols, best_hp, n_splits=3)
        cv_df.to_csv(f"{model_dir}/cv_predictions.csv", index=False)
        pb_rel, pb_df = compute_rel_mae(cv_df, cap)
        pb_df.to_csv(f"{model_dir}/per_beach.csv", index=False)

        # Season-restricted slice (uses ds_target carried through cv_evaluate)
        cv_df["month"] = pd.to_datetime(cv_df["ds_target"]).dt.month
        cv_season = cv_df[cv_df["month"].isin(SEASON_EVAL)]
        season_rel = compute_rel_mae(cv_season, cap)[0] if len(cv_season) > 10 else float("nan")
        mae_g = float(mean_absolute_error(cv_df["y_true"], cv_df["y_pred"]))
        log(f"  overall={cv_rel:.4f}  season={season_rel:.4f}  MAE={mae_g:.1f}")

        # Refit on the full panel for deploy artifact
        x = features[feat_cols].values
        y = features["y_target"].values
        model, _ = _xgb_fit_predict(x, y, x[:1], y[:1], best_hp)
        joblib.dump(model, f"{model_dir}/model.joblib")
        with open(f"{model_dir}/best_params.json", "w") as f:
            json.dump(best_hp, f, indent=2)

        config = {
            "model_type":         "XGBRegressor",
            "horizon":            int(horizon),
            "horizon_days":       int(days),
            "hours_per_day":      HOURS_PER_DAY,
            "features":           feat_cols,
            "hist_pool":          hist_pool,
            "schema":             "castelle_2025",
            "hp":                 best_hp,
            "season_relMAE":      None if np.isnan(season_rel) else float(season_rel),
            "overall_relMAE":     float(cv_rel),
            "season_months":      SEASON_EVAL,
            "trained_at":         datetime.now().isoformat(),
        }
        with open(f"{model_dir}/config.json", "w") as f:
            json.dump(config, f, indent=2)
        log(f"  Saved to {model_dir}/")

        return {"season_relMAE": season_rel, "relMAE": cv_rel, "model_dir": model_dir,
                "feat_cols": feat_cols, "hp": best_hp}
    except Exception as e:
        log(f"  Phase 4 FAILED: {e}", "ERROR")
        traceback.print_exc()
        return None


# ── Eval on additional datasets ─────────────────────────────────────────────

def evaluate_on_dataset(data_dir, ds_label, label, horizon, hist_pool, feat_cols,
                         best_hp, results_dir) -> dict | None:
    if not os.path.isdir(data_dir):
        log(f"    [{ds_label}] skipping (dir missing)")
        return None
    for f in ["cache_2022_clean.csv", "django_2025_clean.csv"]:
        if not os.path.exists(f"{data_dir}/{f}"):
            log(f"    [{ds_label}] missing {f}")
            return None
    log(f"    [{ds_label}] Evaluating {label}...")
    try:
        cache_df, django_df = load_data(data_dir)
        panel = build_panel(cache_df, django_df)
        features = build_xgb_features(panel, horizon, hist_pool)
        features = features.dropna(subset=feat_cols + ["y_target"])
        rel, cv_df = cv_evaluate(features, feat_cols, best_hp, n_splits=3)
        cap = panel.groupby("unique_id")["y"].quantile(0.9).to_dict()
        pb_rel, pb_df = compute_rel_mae(cv_df, cap)
        cv_df["month"] = pd.to_datetime(cv_df["ds_target"]).dt.month
        season_rel = compute_rel_mae(cv_df[cv_df["month"].isin(SEASON_EVAL)], cap)[0]
        mae = float(mean_absolute_error(cv_df["y_true"], cv_df["y_pred"]))
        log(f"    [{ds_label}] overall={rel:.4f}  season={season_rel:.4f}  MAE={mae:.1f}")
        pb_df.to_csv(f"{results_dir}/eval_{label}_{ds_label}.csv", index=False)
        return {"dataset": ds_label, "data_dir": data_dir,
                "relMAE": rel, "season_relMAE": season_rel, "MAE": mae,
                "n_series": panel["unique_id"].nunique()}
    except Exception as e:
        log(f"    [{ds_label}] FAILED: {e}", "ERROR")
        return None


# ── Main ────────────────────────────────────────────────────────────────────

KNOWN_DATASETS = {
    "pipeline_workspace/clean_datasets_210526": "latest",
    "pipeline_workspace/clean_datasets":        "default",
    "pipeline_workspace/clean_dataset_backup":  "old",
}


def parse_args():
    p = argparse.ArgumentParser(description="XGBoost Hourly Training")
    p.add_argument("--data-dir", default="pipeline_workspace/clean_datasets_210526",
                    help="Directory containing cache_2022_clean.csv + django_2025_clean.csv "
                          "(default: pipeline_workspace/clean_datasets_210526, latest synced)")
    p.add_argument("--prefix", default="xgb_run")
    p.add_argument("--out-root", default=None,
                    help="Parent directory for the run dir (default: apps/prediction/xgb_models)")
    p.add_argument("--trials", type=int, default=30,
                    help="Optuna trials per horizon (default 30; use 50+ for final runs)")
    p.add_argument("--deploy-only", action="store_true",
                    help="Skip Phase 1 + Phase 3, use FALLBACK_HP and the full hist pool")
    p.add_argument("--cache-only", action="store_true",
                    help="Train on cache_2022 only (drop django_2025 data)")
    p.add_argument("--eval-dirs", nargs="*", default=None)
    p.add_argument("--skip-eval", action="store_true")
    return p.parse_args()


def resolve_eval_dirs(args):
    if args.skip_eval:
        return {}
    if args.eval_dirs is not None:
        out = {}
        for d in args.eval_dirs:
            rd = _resolve_data_dir(d)
            if os.path.isdir(rd):
                out[rd] = KNOWN_DATASETS.get(d, os.path.basename(rd))
            else:
                log(f"  Eval dir not found, skipping: {d}", "WARN")
        return out
    out = {}
    for d, name in KNOWN_DATASETS.items():
        rd = _resolve_data_dir(d)
        if os.path.isdir(rd):
            out[rd] = name
    return out


def define_features_xgb(panel_df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return (static_cols, hist_pool). Static = always-present lag features."""
    static = ["y_lag1", "y_roll12", "y_roll36"]
    hist_pool = [c for c in (CASTELLE_WEATHER + CALENDAR) if c in panel_df.columns]
    return static, hist_pool


def main(args):
    run_dir = make_run_dir(args.prefix, root=args.out_root)
    results_dir = f"{run_dir}/results"
    os.makedirs(results_dir, exist_ok=True)
    t0 = time.time()

    args.data_dir = _resolve_data_dir(args.data_dir)
    log("XGBoost Training — Per-Horizon Optimization")
    log(f"Target horizons: {TARGET_HORIZONS}")
    log(f"Data dir: {args.data_dir}")
    log(f"Trials per horizon: {args.trials}")

    eval_dirs = resolve_eval_dirs(args)
    log(f"Eval dirs: {eval_dirs}")

    cache_df, django_df = load_data(args.data_dir)
    if getattr(args, 'cache_only', False):
        django_df = None
        log("cache-only mode: django_2025 data excluded from training")
    panel_probe = build_panel(cache_df, django_df)
    static_cols, hist_pool = define_features_xgb(panel_probe)
    log(f"Static features (always kept): {static_cols}")
    log(f"Hist pool ({len(hist_pool)}): {hist_pool}")

    model_results, horizon_configs, eval_results = {}, {}, {}

    for label, horizon in TARGET_HORIZONS.items():
        log("=" * 70)
        log(f"=== HORIZON {label} (H={horizon}, {horizon/HOURS_PER_DAY:.0f} days) ===")
        log("=" * 70)

        # Build features once per horizon for Phase 1/3
        features = build_xgb_features(panel_probe, horizon, hist_pool)
        log(f"Feature rows: {len(features):,}  candidate cols: {features.shape[1]}")

        if args.deploy_only:
            sel_hist = hist_pool
            best_hp = dict(FALLBACK_HP)
            log("  [deploy-only] Skipping Phase 1 + Phase 3")
        else:
            try:
                sel_hist = phase1(features, hist_pool, horizon, results_dir)
            except Exception as e:
                log(f"Phase 1 failed: {e}", "ERROR"); sel_hist = hist_pool

            # Rebuild with selected hist columns
            features = build_xgb_features(panel_probe, horizon, sel_hist)
            feat_cols = [c for c in features.columns
                          if c not in {"unique_id", "ds", "ds_target", "period",
                                       "y", "y_target"}]
            try:
                best_hp = phase3(features, feat_cols, horizon, args.trials, results_dir)
            except Exception as e:
                log(f"Phase 3 failed: {e}", "ERROR"); best_hp = dict(FALLBACK_HP)

        features = build_xgb_features(panel_probe, horizon, sel_hist)
        feat_cols = [c for c in features.columns
                      if c not in {"unique_id", "ds", "ds_target", "period",
                                   "y", "y_target"}]
        horizon_configs[label] = {"hist": sel_hist, "feat_cols": feat_cols, "hp": best_hp}

        r = phase4(label, horizon, cache_df, django_df if not getattr(args, 'cache_only', False) else None,
                    sel_hist, feat_cols, best_hp, run_dir, results_dir)
        if r:
            model_results[label] = r

        if eval_dirs:
            for eval_dir, ds_label in eval_dirs.items():
                er = evaluate_on_dataset(eval_dir, ds_label, label, horizon,
                                          sel_hist, feat_cols, best_hp, results_dir)
                if er:
                    eval_results.setdefault(label, {})[ds_label] = er

    total = time.time() - t0
    log("=" * 70)
    log(f"COMPLETE in {total/60:.1f} min")
    for label, r in model_results.items():
        s = f"{r['season_relMAE']:.4f}" if not np.isnan(r['season_relMAE']) else "n/a"
        log(f"  {label}: season={s}  overall={r['relMAE']:.4f}")
    log(f"  Output: {run_dir}/")

    save_checkpoint(results_dir, "final", {
        "run_dir":         run_dir,
        "data_dir":        args.data_dir,
        "horizon_configs": horizon_configs,
        "models":          {k: {kk: vv for kk, vv in v.items() if kk != "feat_cols"}
                             for k, v in model_results.items()},
        "evaluation":      eval_results,
        "total_time_min":  total / 60,
        "status":          "complete",
    })


if __name__ == "__main__":
    main(parse_args())
