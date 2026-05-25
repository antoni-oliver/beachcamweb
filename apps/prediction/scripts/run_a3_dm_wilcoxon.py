#!/usr/bin/env python3
"""
A3 — Diebold-Mariano + Wilcoxon paired test for the "are three models necessary?" question.

Tests whether the dedicated 3-day and 10-day TFT models are statistically more
accurate than slicing the first 36 (resp. 120) hours of the 15-day model.

Statistical protocol
--------------------
- DM is the PRIMARY test. It weights the loss differential by magnitude, which
  is the operationally correct behaviour because the highest-magnitude errors
  occur on peak-demand days (Saturdays in summer) whose forecast accuracy
  carries disproportionate weight for public-sector staffing.
- Wilcoxon paired signed-rank is reported alongside as a robustness check that
  the DM ranking is driven by a systematic gap, not by a small number of
  outlier observations.
- Both tests are run twice: per-series (one test per webcam → vote count with
  BH-FDR adjustment) and pooled (all rows stacked, sanity check).

Procedure
---------
1. Load the deployed 3d, 10d, 15d models from a single model_set directory
2. Define a fixed test partition (issuance dates × beaches)
3. For each issuance, generate the 3d, 10d and 15d forecasts
4. Pair predictions on the same (beach, issue_date, step_hour)
5. Compute per-pair absolute errors
6. Run DM (with HAC variance + HLN small-sample correction) and Wilcoxon
7. Aggregate via per-beach vote count + pooled sanity

Usage
-----
    cd apps/prediction
    python scripts/run_a3_dm_wilcoxon.py \\
        --model_set tft_old_20260312_215417 \\
        --start 2025-06-01 --end 2025-08-31 \\
        --output a3_results.csv

Outputs
-------
- a3_paired_predictions.csv: long-form (model_set, beach, issue_date, step_hour,
  y_true, y_pred) plus a `_raw.csv` snapshot before the ground-truth join
- a3_results_3d_vs_15d_truncated.csv: per-series + pooled DM/Wilcoxon stats
- a3_results_10d_vs_15d_truncated.csv: same for 10d vs 15d[:120]
- a3_summary.csv: headline vote counts for both tests (BH-FDR adjusted)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import acovf

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


HOURS_PER_DAY = 12
HORIZON_HOURS = {"3d": 36, "10d": 120, "15d": 180}


# ════════════════════════════════════════════════════════════════════════
# Section 1 — Generate paired predictions
# ════════════════════════════════════════════════════════════════════════

def issuance_dates(start: str, end: str, weekday: int = 0) -> list[datetime]:
    """Return Monday issuances (weekday=0) between start and end inclusive."""
    cur = datetime.fromisoformat(start)
    last = datetime.fromisoformat(end)
    out = []
    while cur <= last:
        if cur.weekday() == weekday:
            out.append(cur)
        cur += timedelta(days=1)
    return out


def eligible_webcams(start: str, end: str, min_labeled: int = 100,
                     min_before: int | None = None,
                     min_during: int | None = None):
    """Return webcams with enough labeled snapshots for a valid hindcast.

    A valid hindcast on [start, end] needs both:
      - >=min_before labeled snapshots BEFORE `start` (so the model has
        recent context to anchor the forecast at the issuance date)
      - >=min_during labeled snapshots WITHIN [start, end] (so the ground
        truth join in attach_ground_truth has something to match against)

    Both default to `min_labeled` for backward compatibility.
    """
    from apps.prediction.models import Snapshot
    mb = min_before if min_before is not None else min_labeled
    md = min_during if min_during is not None else min_labeled
    cams = WebCam.objects.select_related("beach").filter(max_crowd_count__gt=0)
    end_inclusive = f"{end} 23:59:59"
    eligible, skipped = [], []
    for c in cams:
        n_before = Snapshot.objects.filter(
            webcam=c, ts__lt=start, predicted_crowd_count__isnull=False
        ).count()
        n_during = Snapshot.objects.filter(
            webcam=c, ts__gte=start, ts__lte=end_inclusive,
            predicted_crowd_count__isnull=False,
        ).count()
        if n_before >= mb and n_during >= md:
            eligible.append(c)
        else:
            skipped.append((c.camera_slug, n_before, n_during))
    print(f"[info] eligible webcams: {len(eligible)} / {cams.count()} "
          f"(thresholds before>={mb} during>={md})")
    if skipped:
        print(f"[info] skipped {len(skipped)} cams (insufficient labels):")
        for slug, nb, nd in skipped:
            print(f"  - {slug:<40} before={nb:>5}  during={nd:>5}")
    return eligible


def collect_predictions(model_set: str, start: str, end: str) -> pd.DataFrame:
    """For each (webcam, issuance, horizon-model) generate a forecast."""
    issuances = issuance_dates(start, end)
    rows = []
    webcams = eligible_webcams(start, end)
    print(f"[info] webcams: {len(webcams)}, issuances: {len(issuances)}")

    for wc in webcams:
        for issue in issuances:
            for hz_label, hz_hours in HORIZON_HOURS.items():
                hz_days = hz_hours // HOURS_PER_DAY
                try:
                    result = tft_service.predict(
                        wc, days=hz_days, since=issue, model_set=model_set,
                        model_key=hz_label,
                    )
                except Exception as e:
                    print(f"[warn] predict failed: {wc.camera_slug} {issue.date()} {hz_label}: {e}")
                    continue
                preds = result.get("predictions") or []
                for step_hour, p in enumerate(preds, start=1):
                    rows.append({
                        "model_set":     model_set,
                        "webcam_slug":   wc.camera_slug,
                        "issue_date":    issue.date().isoformat(),
                        "horizon_label": hz_label,
                        "step_hour":     step_hour,
                        "y_pred":        p.get("crowd_count"),
                        "ds":            p.get("timestamp"),
                        "available":     p.get("available", True),
                    })
    if not rows:
        return pd.DataFrame(columns=[
            "model_set", "webcam_slug", "issue_date", "horizon_label",
            "step_hour", "y_pred", "ds", "available",
        ])
    df = pd.DataFrame(rows)
    print(f"[info] collected {len(df)} prediction rows")
    return df


def attach_ground_truth(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Join the observed predicted_crowd_count for each (webcam, ds).

    Batches one DB query per webcam and builds an in-memory bucket keyed by
    (slug, date, hour) → first snapshot in that hour.
    """
    from apps.prediction.models import Snapshot

    if pred_df.empty or "ds" not in pred_df.columns:
        print("[info] attach_ground_truth: empty prediction frame, nothing to match")
        out = pred_df.copy()
        out["y_true"] = pd.Series(dtype="float64")
        return out

    df = pred_df.copy()
    df["ds_parsed"] = pd.to_datetime(df["ds"], errors="coerce")
    valid = df.dropna(subset=["ds_parsed"])
    print(f"[info] attach_ground_truth: {len(df) - len(valid)} rows have null ds")

    obs_rows = []
    for slug, sub in valid.groupby("webcam_slug"):
        ts_min = sub["ds_parsed"].min()
        ts_max = sub["ds_parsed"].max() + pd.Timedelta(hours=1)
        snaps = Snapshot.objects.filter(
            webcam__camera_slug=slug,
            ts__gte=ts_min, ts__lt=ts_max,
        ).order_by("ts").values("ts", "predicted_crowd_count")
        bucket = {}
        for s in snaps:
            key = (s["ts"].date(), s["ts"].hour)
            if key not in bucket:
                bucket[key] = s["predicted_crowd_count"]
        for ds_val, ds_p in sub[["ds", "ds_parsed"]].drop_duplicates().itertuples(index=False):
            v = bucket.get((ds_p.date(), ds_p.hour))
            if v is not None:
                obs_rows.append({"webcam_slug": slug, "ds": ds_val, "y_true": float(v)})

    obs_df = pd.DataFrame(obs_rows, columns=["webcam_slug", "ds", "y_true"])
    print(f"[info] attach_ground_truth: matched {len(obs_df)} / {len(valid)} prediction rows")
    return pred_df.merge(obs_df, on=["webcam_slug", "ds"], how="left")


# ════════════════════════════════════════════════════════════════════════
# Section 2 — Statistical tests
# ════════════════════════════════════════════════════════════════════════

def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int) -> tuple[float, float]:
    """DM test with HAC variance (Newey-West, bandwidth h-1) and HLN correction.

    Returns (DM statistic, two-sided p-value).
    Convention: d = loss_a - loss_b; positive mean → A worse.
    """
    d = loss_a - loss_b
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    gamma = acovf(d, nlag=h - 1, fft=False)
    var_hac = float(gamma[0] + 2 * gamma[1:].sum())
    if var_hac <= 0 or n <= 0:
        return float("nan"), float("nan")
    dm = float(d.mean()) / np.sqrt(var_hac / n)
    correction = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    dm_hln = dm * correction
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_hln)))
    return dm_hln, float(p_value)


def wilcoxon_paired(loss_a: np.ndarray, loss_b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon signed-rank on per-pair absolute-error differences."""
    d = loss_a - loss_b
    nz = d[d != 0]
    if len(nz) < 10:
        return float("nan"), float("nan")
    res = stats.wilcoxon(nz, alternative="two-sided")
    return float(res.statistic), float(res.pvalue)


def run_tests(df: pd.DataFrame, dedicated_label: str, full_label: str, max_step: int):
    """Compare dedicated_label vs full_label[:max_step] per series + pooled."""
    df = df.dropna(subset=["y_true"]).copy()
    keep = (df["step_hour"] <= max_step) & df["horizon_label"].isin([dedicated_label, full_label])
    sub = df.loc[keep].copy()
    if sub.empty:
        print(f"[warn] no rows for {dedicated_label} vs {full_label} (max_step={max_step})")
        return pd.DataFrame()
    sub["abs_err"] = (sub["y_pred"] - sub["y_true"]).abs()

    paired = (
        sub.pivot_table(
            index=["webcam_slug", "issue_date", "step_hour"],
            columns="horizon_label",
            values="abs_err",
            aggfunc="first",
        )
        .reset_index()
    )
    missing = [c for c in (dedicated_label, full_label) if c not in paired.columns]
    if missing:
        print(f"[warn] pivot missing columns {missing}")
        return pd.DataFrame()
    paired = paired.dropna(subset=[dedicated_label, full_label])

    rows = []
    for slug, g in paired.groupby("webcam_slug"):
        loss_a = g[full_label].to_numpy()       # A: 15d truncated
        loss_b = g[dedicated_label].to_numpy()  # B: dedicated
        dm_stat, dm_p = diebold_mariano(loss_a, loss_b, h=max_step)
        wx_stat, wx_p = wilcoxon_paired(loss_a, loss_b)
        rows.append({
            "webcam_slug": slug,
            "n_pairs": len(g),
            "mean_abs_err_dedicated": float(g[dedicated_label].mean()),
            "mean_abs_err_truncated": float(g[full_label].mean()),
            "delta_mean": float(g[full_label].mean() - g[dedicated_label].mean()),
            "DM_stat": dm_stat, "DM_p": dm_p,
            "Wilcoxon_stat": wx_stat, "Wilcoxon_p": wx_p,
        })

    # Pooled — sanity check, sensitive to cross-correlation
    pooled_a = paired[full_label].to_numpy()
    pooled_b = paired[dedicated_label].to_numpy()
    dm_stat, dm_p = diebold_mariano(pooled_a, pooled_b, h=max_step)
    wx_stat, wx_p = wilcoxon_paired(pooled_a, pooled_b)
    rows.append({
        "webcam_slug": "__POOLED__",
        "n_pairs": len(paired),
        "mean_abs_err_dedicated": float(paired[dedicated_label].mean()),
        "mean_abs_err_truncated": float(paired[full_label].mean()),
        "delta_mean": float(paired[full_label].mean() - paired[dedicated_label].mean()),
        "DM_stat": dm_stat, "DM_p": dm_p,
        "Wilcoxon_stat": wx_stat, "Wilcoxon_p": wx_p,
    })
    return pd.DataFrame(rows)


def summarise(per_series: pd.DataFrame, comparison_name: str, alpha: float = 0.05) -> dict:
    """Vote-count significants. DM is the primary test; Wilcoxon is sanity.

    Rationale for DM as primary: forecast errors on peak-demand days (Saturdays
    in summer) are operationally the most consequential — DM weights the loss
    differential by magnitude, so a model that errs most on those peak days is
    penalised proportionally. Wilcoxon ignores magnitude (rank-based) and would
    dilute the peak-day signal; it is reported as a robustness check that the
    DM ranking is not purely an outlier artefact.

    Direction from sign of delta_mean = mean(loss_a) - mean(loss_b):
      - delta_mean > 0 → A worse → B wins
      - delta_mean < 0 → A better → A wins
    BH-FDR adjusts across the per-series tests at alpha.
    """
    from statsmodels.stats.multitest import multipletests
    body = per_series[per_series["webcam_slug"] != "__POOLED__"].copy()
    n = len(body)

    body["DM_p_bh"] = multipletests(body["DM_p"].fillna(1.0),
                                     alpha=alpha, method="fdr_bh")[1]
    body["Wilcoxon_p_bh"] = multipletests(body["Wilcoxon_p"].fillna(1.0),
                                           alpha=alpha, method="fdr_bh")[1]

    sig_dm = body[body["DM_p_bh"] < alpha]
    a_wins_dm = (sig_dm["delta_mean"] > 0).sum()
    b_wins_dm = (sig_dm["delta_mean"] < 0).sum()
    ties_dm   = n - len(sig_dm)

    sig_wx = body[body["Wilcoxon_p_bh"] < alpha]
    a_wins_wx = (sig_wx["delta_mean"] > 0).sum()
    b_wins_wx = (sig_wx["delta_mean"] < 0).sum()
    ties_wx   = n - len(sig_wx)

    pooled = per_series[per_series["webcam_slug"] == "__POOLED__"].iloc[0]
    return {
        "comparison":              comparison_name,
        "n_series":                n,
        # Primary: DM
        "DM_dedicated_wins_bh":    int(a_wins_dm),
        "DM_truncated_wins_bh":    int(b_wins_dm),
        "DM_ties_bh":              int(ties_dm),
        "pooled_DM_stat":          pooled["DM_stat"],
        "pooled_DM_p":             pooled["DM_p"],
        # Sanity: Wilcoxon
        "WX_dedicated_wins_bh":    int(a_wins_wx),
        "WX_truncated_wins_bh":    int(b_wins_wx),
        "WX_ties_bh":              int(ties_wx),
        "pooled_Wilcoxon_p":       pooled["Wilcoxon_p"],
        "pooled_delta_mean":       pooled["delta_mean"],
    }


# ════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_set", default="tft_old_20260312_215417")
    ap.add_argument("--start", default="2025-06-01")
    ap.add_argument("--end", default="2025-08-31")
    ap.add_argument("--output", default="a3_paired_predictions.csv")
    ap.add_argument("--predictions_csv",
                    help="Skip prediction generation; load from this CSV.")
    args = ap.parse_args()

    if args.predictions_csv:
        df = pd.read_csv(args.predictions_csv)
        print(f"[info] loaded {len(df)} rows from {args.predictions_csv}")
        if "y_true" not in df.columns:
            df = attach_ground_truth(df)
            df.to_csv(args.output, index=False)
    else:
        print(f"[info] generating predictions for model_set={args.model_set}, "
              f"period {args.start} → {args.end}")
        df = collect_predictions(args.model_set, args.start, args.end)
        raw_path = args.output.replace(".csv", "_raw.csv")
        df.to_csv(raw_path, index=False)
        print(f"[info] saved raw predictions to {raw_path}")
        df = attach_ground_truth(df)
        df.to_csv(args.output, index=False)

    print("\n" + "=" * 72)
    print("Comparison 1: dedicated 3d  vs  15d truncated to first 36 hours")
    print("=" * 72)
    res1 = run_tests(df, dedicated_label="3d", full_label="15d", max_step=36)
    res1.to_csv("a3_results_3d_vs_15d_truncated.csv", index=False)
    print(res1.to_string(index=False))

    print("\n" + "=" * 72)
    print("Comparison 2: dedicated 10d vs  15d truncated to first 120 hours")
    print("=" * 72)
    res2 = run_tests(df, dedicated_label="10d", full_label="15d", max_step=120)
    res2.to_csv("a3_results_10d_vs_15d_truncated.csv", index=False)
    print(res2.to_string(index=False))

    print("\n" + "=" * 72)
    print("Headline summary (DM primary; Wilcoxon sanity)")
    print("=" * 72)
    summaries = [
        summarise(res1, "3d_dedicated vs 15d[:36]"),
        summarise(res2, "10d_dedicated vs 15d[:120]"),
    ]
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv("a3_summary.csv", index=False)
    print(summary_df.to_string(index=False))

    print("\n[done] tables saved:")
    print("  - a3_paired_predictions.csv         (long-form per-step errors)")
    print("  - a3_results_3d_vs_15d_truncated.csv (per-series + pooled)")
    print("  - a3_results_10d_vs_15d_truncated.csv (same)")
    print("  - a3_summary.csv                     (headline counts)")


if __name__ == "__main__":
    main()
