#!/usr/bin/env python3
"""
A2 — Multi-scenario fixed validation.

Evaluates every candidate model_set against frozen test partitions and produces
a ranking table per scenario. P90 normalisation is the canonical operational
metric (aligned with the production classification thresholds).

Scenarios
---------
- S1 (cross-year):   model trained on 2022 only         → test Jun-Aug 2025
- S2 (full season):  model trained on 2022 + 2025 spr   → test Apr-Sep 2025
- S3 (recent month): model trained up to 2025-08-31     → test Sep 2025

The eligibility filter requires ≥100 labelled snapshots in the test window per
camera (and ≥100 before the window, except for S2 where pre-window data may not
exist for the labelling era — see SCENARIOS dict).

Usage
-----
    cd apps/prediction
    python scripts/run_a2_scenarios.py \\
        --scenarios S1,S2,S3 \\
        --normalisation p90 \\
        --output_dir a2_results/
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# ─── Django setup ────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent.parent  # apps/prediction
PROJ = HERE.parent.parent  # beachcamweb root
sys.path.insert(0, str(PROJ))

import django  # noqa: E402
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from apps.prediction.tft_service import tft_service  # noqa: E402
from apps.webcam.models import WebCam  # noqa: E402

# Reuse the eligibility filter and ground-truth join from A3
from run_a3_dm_wilcoxon import (  # noqa: E402
    eligible_webcams,
    attach_ground_truth,
    issuance_dates,
)
from crowd_outliers import cap_outliers  # noqa: E402  canonical per-series P90 cap


HOURS_PER_DAY = 13
HORIZON_HOURS = {"3d": 39, "10d": 130, "15d": 195}
SEASON_MONTHS = {4, 5, 6, 7, 8, 9}
SUMMER_MONTHS = {6, 7, 8}

# ─── Scenarios — frozen test windows ─────────────────────────────────────

SCENARIOS = {
    "S1": {
        "label": "Cross-year (train 2022 → test summer 2025)",
        "test_start": "2025-06-01",
        "test_end":   "2025-08-31",
        "min_before": 100,
        "min_during": 100,
        "expected_anchor_model": "tft_train_2022_validate_2025_summer",
    },
    "S2": {
        "label": "Full season (train 2022+2025 spring → test Apr-Sep 2025)",
        "test_start": "2025-04-01",
        "test_end":   "2025-09-30",
        # Pre-window check disabled: labelling era began ~April 2025, so
        # earlier predicted_crowd_count rows do not exist for most cams.
        "min_before": 0,
        "min_during": 100,
        "expected_anchor_model": "tft_old_20260312_215417",
    },
    "S3": {
        "label": "Recent month (train up to 2025-08-31 → test Sep 2025)",
        "test_start": "2025-09-01",
        "test_end":   "2025-09-30",
        "min_before": 100,
        "min_during": 100,
        "expected_anchor_model": "tft_old_20260312_215417",
    },
}


def list_model_sets() -> list[str]:
    """Every model_set the prediction service knows how to serve.

    Combines auto-discovered TFT runs (tft_models/) with explicitly configured
    sets in settings.TFT_MODEL_SETS (where the LSTM baseline lives).
    """
    from django.conf import settings
    discovered = set(tft_service._discovered)
    configured = set(getattr(settings, "TFT_MODEL_SETS", {}).keys())
    return sorted(discovered | configured)


# ─── Prediction collection ───────────────────────────────────────────────

def collect_for_scenario(scen_code: str, model_sets: list[str]) -> pd.DataFrame:
    spec = SCENARIOS[scen_code]
    start, end = spec["test_start"], spec["test_end"]
    print(f"\n{'='*72}\n[scenario {scen_code}] {spec['label']}\n{'='*72}")

    cams = eligible_webcams(start, end,
                            min_before=spec.get("min_before"),
                            min_during=spec.get("min_during"))
    issuances = issuance_dates(start, end)
    print(f"[info] {len(cams)} cams × {len(issuances)} issuances × "
          f"{len(model_sets)} models × {len(HORIZON_HOURS)} horizons")
    if not cams:
        print(f"[warn] scenario {scen_code}: no eligible cameras, skipping")
        return pd.DataFrame(columns=[
            "scenario", "model_set", "webcam_slug", "issue_date",
            "horizon_label", "step_hour", "y_pred", "ds", "available",
        ])

    rows = []
    for ms in model_sets:
        print(f"[info] model_set={ms}")
        for wc in cams:
            for issue in issuances:
                for hz_label, hz_hours in HORIZON_HOURS.items():
                    hz_days = hz_hours // HOURS_PER_DAY
                    try:
                        result = tft_service.predict(
                            wc, days=hz_days, since=issue,
                            model_set=ms, model_key=hz_label,
                        )
                    except Exception as e:
                        print(f"  [warn] {ms}/{wc.camera_slug}/{issue.date()}/{hz_label}: {e}")
                        continue
                    for step_hour, p in enumerate(result.get("predictions") or [], 1):
                        rows.append({
                            "scenario":      scen_code,
                            "model_set":     ms,
                            "webcam_slug":   wc.camera_slug,
                            "issue_date":    issue.date().isoformat(),
                            "horizon_label": hz_label,
                            "step_hour":     step_hour,
                            "y_pred":        p.get("crowd_count"),
                            "ds":            p.get("timestamp"),
                            "available":     p.get("available", True),
                        })
    return pd.DataFrame(rows)


# ─── Aggregation: P90-normalised relMAE ──────────────────────────────────

def compute_relmae_table(df: pd.DataFrame, capacity_lookup: dict,
                          normalisation: str = "p90") -> pd.DataFrame:
    """Compute relMAE per (scenario, model_set, horizon) under three windows.

    - all:    every row
    - season: rows whose ds-month ∈ {4..9}
    - summer: rows whose ds-month ∈ {6, 7, 8}

    Normalisation modes:
      - "p90"  : per-camera WebCam.max_crowd_count --- the OPERATIONAL capacity
                 (P90 of the PREDICTED count over full history), matching the
                 four-level classification thresholds and the production CV.
                 This is the operational lens, NOT the statistical per-series P90
                 of ACTUAL daytime counts used for the cross-family headline; the
                 two must never be mixed in one table. Cameras lacking
                 max_crowd_count are DROPPED (not back-filled with an actual-count
                 P90), so one table uses exactly one denominator.
      - "mean" : per-camera mean of y_true (legacy/debug). All cameras included.
    """
    df = df.dropna(subset=["y_true", "y_pred"]).copy()
    if "available" in df.columns:
        df = df[df["available"].astype(bool)].copy()   # drop night placeholder rows (I4)
    df["ds_parsed"] = pd.to_datetime(df["ds"], errors="coerce")
    df = df.dropna(subset=["ds_parsed"])
    df["abs_err"] = (df["y_pred"] - df["y_true"]).abs()
    df["month"] = df["ds_parsed"].dt.month

    if normalisation == "p90":
        df["capacity"] = df["webcam_slug"].map(capacity_lookup)
        missing = df["capacity"].isna() | (df["capacity"] <= 0)
        if missing.any():
            # Drop cameras without an operational max_crowd_count instead of
            # back-filling with an actual-count P90: mixing the operational and
            # statistical denominators in one table is the error the thesis flags.
            n_miss = df.loc[missing, "webcam_slug"].nunique()
            print(f"[info] dropping {n_miss} webcam(s) without Django max_crowd_count "
                  f"({int(missing.sum())} rows) to keep one operational denominator")
            df = df[~missing].copy()
    else:
        df["capacity"] = df.groupby("webcam_slug")["y_true"].transform("mean")

    def _rel(g: pd.DataFrame) -> float:
        denom = g["capacity"].mean()
        return float(g["abs_err"].mean() / denom) if denom > 0 else float("nan")

    rows = []
    for (scen, ms, hz), g in df.groupby(["scenario", "model_set", "horizon_label"]):
        season_g = g[g["month"].isin(SEASON_MONTHS)]
        summer_g = g[g["month"].isin(SUMMER_MONTHS)]
        rows.append({
            "scenario":      scen,
            "model_set":     ms,
            "horizon":       hz,
            "n_rows":        len(g),
            "relMAE_all":    _rel(g),
            "relMAE_season": _rel(season_g) if len(season_g) > 0 else float("nan"),
            "relMAE_summer": _rel(summer_g) if len(summer_g) > 0 else float("nan"),
        })
    return pd.DataFrame(rows)


def build_capacity_lookup() -> dict[str, float]:
    """Per-camera P90 from WebCam.max_crowd_count (production normalisation key)."""
    return {c.camera_slug: float(c.max_crowd_count or 0)
            for c in WebCam.objects.all() if (c.max_crowd_count or 0) > 0}


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="S1,S2,S3",
                    help="Comma-separated scenario codes to run.")
    ap.add_argument("--model_sets", default="",
                    help="Comma-separated model_set names; default = all registered sets.")
    ap.add_argument("--output_dir", default="a2_results")
    ap.add_argument("--normalisation", choices=["p90", "mean"], default="p90",
                    help="relMAE denominator: P90 (canonical, default) or mean (legacy).")
    ap.add_argument("--cap-k", dest="cap_k", type=float, default=1.5,
                    help="Cap real y_true above k*P90 per camera before scoring "
                         "(outlier removal; 0 disables). Predictions never capped.")
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    model_sets = args.model_sets.split(",") if args.model_sets else list_model_sets()
    print(f"[info] evaluating {len(model_sets)} model_sets: {model_sets}")
    capacity = build_capacity_lookup()
    print(f"[info] P90 capacity available for {len(capacity)} webcams "
          f"(normalisation={args.normalisation})")

    all_predictions = []
    for code in args.scenarios.split(","):
        code = code.strip()
        if code not in SCENARIOS:
            print(f"[warn] unknown scenario {code!r}; skip")
            continue
        df_pred = collect_for_scenario(code, model_sets)
        df_pred = attach_ground_truth(df_pred)
        df_pred.to_csv(out / f"a2_predictions_{code}.csv", index=False)
        all_predictions.append(df_pred)
        print(f"[info] saved {len(df_pred)} rows for {code}")

    if not all_predictions:
        print("[fatal] no scenarios produced output; aborting")
        return

    combined = pd.concat(all_predictions, ignore_index=True)

    if args.cap_k and args.cap_k > 0:   # cap the REAL y_true only (per camera); never predictions
        combined["unique_id"] = combined["webcam_slug"]
        cap_outliers(combined, y_col="y_true", k=args.cap_k)
        combined.drop(columns=["unique_id"], inplace=True)

    combined.to_csv(out / "a2_predictions_all.csv", index=False)

    print("\n[info] computing relMAE table …")
    table = compute_relmae_table(combined, capacity, normalisation=args.normalisation)
    table = table.sort_values(["scenario", "horizon", "relMAE_summer"])
    table.to_csv(out / "a2_ranking_table.csv", index=False)

    print("\n" + "=" * 72)
    print("Ranking by scenario × horizon (sorted by summer relMAE)")
    print("=" * 72)
    print(table.to_string(index=False))
    print(f"\n[done] tables saved under {out}/")


if __name__ == "__main__":
    main()
