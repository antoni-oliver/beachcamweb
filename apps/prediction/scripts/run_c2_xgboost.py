#!/usr/bin/env python3
"""
C2 — Castelle-style XGBoost baseline, three configurations × A2 scenarios.

Replicates the feature schema of Castelle et al. (2025, JMSE 13:1181) on the
Balearic panel: air temperature, precipitation, wind speed, wind direction,
insolation — humidity explicitly omitted. Runs three configurations side by
side, optionally across the same scenario windows used by run_a2_scenarios.py.

Configurations
--------------
  latest      — train on the latest cleaned panel (clean_datasets_180326),
                everything before the test window
  backup      — train on the 22-beach old-camera backup
  cross_year  — train on 2022 rows only, test on the chosen window (S1 mirror
                of the TFT's tft_train_2022_validate_2025_summer)

Scenarios (optional via --scenarios)
------------------------------------
  S1, S3, S4   — same windows as A2. When this flag is used, a fresh retrain
                 happens per scenario × horizon, and the output table includes
                 a `scenario` column so it can be merged with A2's ranking.

Metrics
-------
  relMAE_*_p90 — P90-normalised (canonical operational metric)
  relMAE_*_mean — mean-normalised (A2-comparable legacy)
  R² median / mean / min / max across the per-beach panel

Usage
-----
    cd apps/prediction
    python scripts/run_c2_xgboost.py \\
        --configs latest,backup,cross_year \\
        --scenarios S1,S3,S4 \\
        --trials 50 \\
        --output_dir c2_results/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from crowd_outliers import cap_outliers   # canonical 1.5xP90 real-count cap (I2)


# ─── Castelle 2025 feature schema (humidity omitted) ─────────────────────

CASTELLE_WEATHER = [
    "om_temperature_2m",       # T
    "om_precipitation",        # P
    "om_wind_speed_10m",       # W
    "om_wind_direction_10m",   # α
    "om_shortwave_radiation",  # I
]
CALENDAR = ["hour", "day_of_week", "month", "is_weekend", "is_summer"]
HORIZONS = {"3d": 39, "10d": 130, "15d": 195}
SEASON_MONTHS = {4, 5, 6, 7, 8, 9}
SUMMER_MONTHS = {6, 7, 8}

# Mirror A2's scenario windows so the XGBoost ranking is directly comparable.
SCENARIOS = {
    "S1": ("2025-06-01", "2025-08-31"),
    "S3": ("2025-04-01", "2025-09-30"),
    "S4": ("2025-09-01", "2025-09-30"),
}

HERE = Path(__file__).resolve().parent.parent  # apps/prediction
PROJ = HERE.parent.parent  # beachcamweb root

DEFAULT_CSVS = {
    "latest": HERE / "pipeline_workspace" / "clean_datasets_180326" / "all_clean.csv",
    "backup": PROJ.parent / "weather_data" / "clean_dataset_backup" / "all_clean.csv",
}


# ─── Configurations ──────────────────────────────────────────────────────

def build_configs(args) -> dict[str, dict]:
    return {
        "latest": {
            "label": "Latest dataset (production training panel)",
            "csv": Path(args.latest_csv) if args.latest_csv else DEFAULT_CSVS["latest"],
            "train_filter": None,
            "test_start": args.test_start,
            "test_end":   args.test_end,
        },
        "backup": {
            "label": "Backup dataset (22-beach old-camera panel)",
            "csv": Path(args.backup_csv) if args.backup_csv else DEFAULT_CSVS["backup"],
            "train_filter": None,
            "test_start": args.test_start,
            "test_end":   args.test_end,
        },
        "cross_year": {
            "label": "Cross-year stress test (train on 2022, test on summer 2025)",
            "csv": Path(args.backup_csv) if args.backup_csv else DEFAULT_CSVS["backup"],
            "train_filter": "year_eq_2022",
            "test_start": args.test_start,
            "test_end":   args.test_end,
        },
    }


# ─── Feature engineering ─────────────────────────────────────────────────

def build_features(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """For each row at time t, produce a feature row whose target is y[t+h].

    Features:
      - y_lag1, y_roll12, y_roll36       autoregressive context
      - <weather>_t_plus_h               known-future weather (Castelle schema)
      - <calendar>_t_plus_h              calendar at t+h
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    df["y_lag1"] = df.groupby("unique_id")["y"].shift(1)
    df["y_roll12"] = df.groupby("unique_id")["y"].transform(
        lambda s: s.shift(1).rolling(12, min_periods=1).mean())
    df["y_roll36"] = df.groupby("unique_id")["y"].transform(
        lambda s: s.shift(1).rolling(36, min_periods=1).mean())

    cols_ahead = [c for c in (CASTELLE_WEATHER + CALENDAR) if c in df.columns]
    df_future = df.groupby("unique_id")[cols_ahead].shift(-horizon_hours)
    df_future.columns = [f"{c}_t_plus_h" for c in cols_ahead]

    df["y_target"] = df.groupby("unique_id")["y"].shift(-horizon_hours)

    out = pd.concat([
        df[["unique_id", "ds", "y_lag1", "y_roll12", "y_roll36"]],
        df_future,
        df[["y_target"]],
    ], axis=1)
    return out.dropna(subset=["y_target", "y_lag1"]).reset_index(drop=True)


def split_panel(features: pd.DataFrame, test_start: str, test_end: str,
                train_filter: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_mask = (features["ds"] >= test_start) & (features["ds"] <= test_end)
    train = features.loc[features["ds"] < pd.Timestamp(test_start)].copy()   # strictly before test — no post-test leakage (I6)
    test = features.loc[test_mask].copy()
    if train_filter == "year_eq_2022":
        train = train[train["ds"].dt.year == 2022]
    return train, test


# ─── Optuna sweep ────────────────────────────────────────────────────────

def objective_factory(X_train, y_train, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-4, 1.0, log=True),
            "tree_method":      "hist",
            "n_jobs":           -1,
            "random_state":     42,
            "verbosity":        0,
        }
        model = XGBRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        pred = model.predict(X_val)
        return float(np.mean(np.abs(pred - y_val)))
    return objective


def train_xgboost_horizon(panel: pd.DataFrame, horizon: int,
                          test_start: str, test_end: str,
                          train_filter: str | None, trials: int) -> dict:
    features = build_features(panel, horizon)
    train_df, test_df = split_panel(features, test_start, test_end, train_filter)
    if len(train_df) < 100 or len(test_df) < 50:
        return {"error": f"insufficient data: train={len(train_df)}, test={len(test_df)}"}

    train_df = train_df.sort_values("ds")
    cut = int(len(train_df) * 0.9)
    train, val = train_df.iloc[:cut], train_df.iloc[cut:]

    feat_cols = [c for c in train.columns
                 if c not in {"unique_id", "ds", "y_target"}]
    X_tr, y_tr = train[feat_cols].values, train["y_target"].values
    X_va, y_va = val[feat_cols].values,   val["y_target"].values
    X_te, y_te = test_df[feat_cols].values, test_df["y_target"].values

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_factory(X_tr, y_tr, X_va, y_va),
                   n_trials=trials, show_progress_bar=True)

    best = study.best_params
    final = XGBRegressor(**best, tree_method="hist", n_jobs=-1,
                          random_state=42, verbosity=0)
    final.fit(np.concatenate([X_tr, X_va]), np.concatenate([y_tr, y_va]),
              verbose=False)
    pred = final.predict(X_te)

    result = test_df[["unique_id", "ds", "y_target"]].copy()
    result["y_pred"] = pred
    return {
        "model": final,
        "best_params": best,
        "best_val_mae": float(study.best_value),
        "n_train": len(train),
        "n_val":   len(val),
        "n_test":  len(test_df),
        "test_predictions": result,
        "feature_columns": feat_cols,
    }


# ─── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(predictions: pd.DataFrame, horizon_label: str,
                    capacity_lookup: dict[str, float],
                    normalisation: str = "p90") -> dict:
    """relMAE under (all / season / summer) for one prediction frame.

    Normalisation:
      - p90 : per-beach P90 (production). Fallback to on-the-fly P90 if
              capacity_lookup misses; never falls back to mean.
      - mean: per-beach mean of y_target (legacy/debug).
    """
    df = predictions.copy()
    df["month"] = df["ds"].dt.month
    df["abs_err"] = (df["y_pred"] - df["y_target"]).abs()
    df["beach"] = df["unique_id"]

    if normalisation == "mean":
        df["capacity"] = df.groupby("beach")["y_target"].transform("mean")
    else:
        df["capacity"] = df["beach"].map(capacity_lookup)
        missing = df["capacity"].isna() | (df["capacity"] <= 0)
        if missing.any():
            df.loc[missing, "capacity"] = df.loc[missing].groupby("beach")[
                "y_target"].transform(lambda s: s.quantile(0.9))

    def _rel(g):
        d = g["capacity"].mean()
        return float(g["abs_err"].mean() / d) if d > 0 else float("nan")

    out = {
        "horizon":       horizon_label,
        "n_rows":        len(df),
        "relMAE_all":    _rel(df),
        "relMAE_season": _rel(df[df["month"].isin(SEASON_MONTHS)]),
        "relMAE_summer": _rel(df[df["month"].isin(SUMMER_MONTHS)]),
    }
    r2_per_beach = {}
    for beach, g in df.groupby("beach"):
        if len(g) < 10:
            continue
        try:
            r2_per_beach[beach] = float(r2_score(g["y_target"], g["y_pred"]))
        except Exception:
            pass
    if r2_per_beach:
        vals = np.array(list(r2_per_beach.values()))
        out["r2_median"] = float(np.median(vals))
        out["r2_mean"]   = float(np.mean(vals))
        out["r2_min"]    = float(np.min(vals))
        out["r2_max"]    = float(np.max(vals))
        out["r2_per_beach"] = r2_per_beach
    return out


# ─── Per-config driver ───────────────────────────────────────────────────

def run_config(name: str, spec: dict, trials: int, out_dir: Path,
               scenarios: dict | None = None) -> pd.DataFrame:
    """Train + evaluate one configuration. If `scenarios` is given, retrain
    once per scenario × horizon and emit one row per (scenario, horizon).
    """
    print(f"\n{'#'*72}\n# Configuration: {name}\n# {spec['label']}\n# CSV: {spec['csv']}\n{'#'*72}")
    if not spec["csv"].exists():
        print(f"[fatal] CSV not found: {spec['csv']}")
        return pd.DataFrame()

    panel = pd.read_csv(spec["csv"])
    panel["ds"] = pd.to_datetime(panel["ds"])
    print(f"[info] panel: {len(panel)} rows, {panel['unique_id'].nunique()} series, "
          f"ds range {panel['ds'].min().date()} → {panel['ds'].max().date()}")
    panel["y_raw"] = pd.to_numeric(panel["y"], errors="coerce")    # raw for the relMAE denominator (I3)
    panel, _ncap = cap_outliers(panel, y_col="y", verbose=False)   # cap targets at 1.5xP90 real daytime, like every other family (I2)

    cfg_out = out_dir / name
    cfg_out.mkdir(parents=True, exist_ok=True)

    windows = scenarios if scenarios else {"_single": (spec["test_start"], spec["test_end"])}
    summary_rows = []
    for scen_code, (ts, te) in windows.items():
        print(f"\n=== scenario {scen_code}: {ts} → {te} ===")
        capacity = {u: max(float(v), 1.0)                          # train-only (ds<test_start), RAW y, floor 1.0 (I3)
                    for u, v in panel.loc[panel["ds"] < pd.Timestamp(ts)]
                                     .groupby("unique_id")["y_raw"].quantile(0.9).items()}
        for hz_label, hz_h in HORIZONS.items():
            print(f"--- horizon {hz_label} (h={hz_h}) ---")
            res = train_xgboost_horizon(panel, hz_h, ts, te,
                                        spec["train_filter"], trials)
            if "error" in res:
                print(f"[warn] {scen_code}/{hz_label}: {res['error']}")
                continue

            tag = f"{scen_code}_{hz_label}" if scenarios else hz_label
            res["test_predictions"].to_csv(cfg_out / f"predictions_{tag}.csv", index=False)
            with open(cfg_out / f"best_params_{tag}.json", "w") as f:
                json.dump(res["best_params"], f, indent=2)

            metrics_p90  = compute_metrics(res["test_predictions"], hz_label, capacity, "p90")
            metrics_mean = compute_metrics(res["test_predictions"], hz_label, capacity, "mean")
            r2_per_beach = metrics_p90.pop("r2_per_beach", {})
            metrics_mean.pop("r2_per_beach", None)
            with open(cfg_out / f"r2_per_beach_{tag}.json", "w") as f:
                json.dump(r2_per_beach, f, indent=2)

            row = {
                "scenario":            scen_code,
                "config":              name,
                "horizon":             hz_label,
                "n_train":             res["n_train"],
                "n_test":              res["n_test"],
                "relMAE_all_p90":      metrics_p90["relMAE_all"],
                "relMAE_season_p90":   metrics_p90["relMAE_season"],
                "relMAE_summer_p90":   metrics_p90["relMAE_summer"],
                "relMAE_all_mean":     metrics_mean["relMAE_all"],
                "relMAE_season_mean":  metrics_mean["relMAE_season"],
                "relMAE_summer_mean":  metrics_mean["relMAE_summer"],
                "r2_median":           metrics_p90.get("r2_median", float("nan")),
                "r2_mean":             metrics_p90.get("r2_mean",   float("nan")),
                "r2_min":              metrics_p90.get("r2_min",    float("nan")),
                "r2_max":              metrics_p90.get("r2_max",    float("nan")),
            }
            summary_rows.append(row)
            print(f"[info] {scen_code}/{hz_label}: relMAE_summer_mean="
                  f"{row['relMAE_summer_mean']:.4f}  r2_median={row['r2_median']:.3f}")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(cfg_out / "summary.csv", index=False)
    print(f"\n[done] config {name} → {cfg_out}/summary.csv")
    return summary


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="latest,backup,cross_year")
    ap.add_argument("--latest_csv", default="",
                    help=f"Override latest CSV (default {DEFAULT_CSVS['latest']})")
    ap.add_argument("--backup_csv", default="",
                    help=f"Override backup CSV (default {DEFAULT_CSVS['backup']})")
    ap.add_argument("--scenarios", default="",
                    help="Comma-separated A2 scenario codes (S1,S3,S4).")
    ap.add_argument("--test_start", default="2025-06-01")
    ap.add_argument("--test_end",   default="2025-08-31")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--output_dir", default="c2_results")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    configs = build_configs(args)
    selected = [c.strip() for c in args.configs.split(",")]

    scenarios = None
    if args.scenarios:
        codes = [c.strip() for c in args.scenarios.split(",")]
        scenarios = {c: SCENARIOS[c] for c in codes if c in SCENARIOS}
        print(f"[info] running scenarios: {list(scenarios)}")

    combined = []
    for name in selected:
        if name not in configs:
            print(f"[warn] unknown config {name!r}; skip")
            continue
        df = run_config(name, configs[name], args.trials, out, scenarios)
        if not df.empty:
            combined.append(df)

    if not combined:
        print("[fatal] no configurations produced output")
        return

    all_summary = pd.concat(combined, ignore_index=True)
    sort_cols = (["scenario", "horizon", "relMAE_summer_mean"]
                 if "scenario" in all_summary else ["horizon", "relMAE_summer_mean"])
    all_summary = all_summary.sort_values(sort_cols)
    all_summary.to_csv(out / "c2_summary_all_configs.csv", index=False)

    print("\n" + "=" * 72)
    print("Castelle-style XGBoost — combined summary")
    print("=" * 72)
    print(all_summary.to_string(index=False))
    print(f"\n[done] combined summary saved to {out}/c2_summary_all_configs.csv")


if __name__ == "__main__":
    main()
