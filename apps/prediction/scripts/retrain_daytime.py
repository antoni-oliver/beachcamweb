#!/usr/bin/env python3
"""Daytime-frame validated retraining — the 15-DAY product evaluation.

CANONICAL cross-family comparator for apps/prediction. SUPERSEDES the wrong-frame
run_cross_year_unified.py (continuous freq="h" + Castelle h-step XGB). Uses the
daytime integer frame (8-20, per-series cumcount ds, freq=1), a recursive 1-step
XGB that shares forecast origins, identical-row matching on (unique_id, ds,
issue_date), per-series-P90 relMAE, and the Diebold--Mariano test. Run from this
scripts/ dir, e.g.: python retrain_daytime.py --panel
pipeline_workspace/clean_dataset_backup/all_clean.csv --protocol cache2022
--horizons 3 10 15 --seeds 1 --final-steps 150 --prior-db ../optuna/cross_year.db
--prior-campaign <cy_capped...> --out validated_daytime

The companion `retrain_3models_validated.py` reindexes each series to a
continuous 24h grid (NeuralForecast freq="h"), which silently halves the
horizon: H=180 continuous hours = 7.5 calendar days, NOT 15. The DEPLOYED models
(`cross_year_train_3_models.py` + `tft_service.py`) instead use a daytime-only
integer index (freq=1, night skipped), so H=180 daytime steps ≈ 15 calendar
days. This script evaluates THAT frame, so the thesis numbers match the product.

Same rigour as retrain_3models_validated: rolling weekly-Monday origins (see only
the past), identical-rows matching across families, per-series-P90 relMAE, and
the Diebold-Mariano test — on the daytime integer frame, all three models
producing the FULL 15-day hourly trajectory:
  - TFT / LSTM : NeuralForecast seq2seq, freq=1 over daytime steps.
  - XGB        : RECURSIVE 1-step model iterated H steps (feeds its own
                 prediction back as the next lag) — a fair trajectory baseline.

All rolling origins are predicted in a SINGLE batched nf.predict via pseudo-series
(unique_id = beach__origin), so the per-origin Lightning-trainer overhead does
not dominate runtime. Matching keys on (unique_id, ds_real, issue_date).

Run (server GPU):
  CUDA_VISIBLE_DEVICES=0 python new_training_pipeline/retrain_daytime.py \
    --panel beachcamweb/apps/prediction/pipeline_workspace/clean_dataset_backup/all_clean.csv \
    --protocol cache2022 --horizons 15 --trials 15 --seeds 1 \
    --out new_training_pipeline/validated_daytime
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import retrain_3models_validated as R   # reuse metric / DM / model-build helpers

CALENDAR_FUTR = ["hour", "day_of_week", "month", "is_weekend", "is_summer", "is_holiday"]
SUMMER_MONTHS = R.SUMMER_MONTHS
SEASON_MONTHS = R.SEASON_MONTHS
SEED = R.SEED


# ── Daytime integer-index frame ───────────────────────────────────────────────

def _es_holidays(years):
    try:
        import holidays as hol
    except ImportError:
        print("[daytime] 'holidays' not installed -> is_holiday=0 (harmless, same for all models)")
        return set()
    out = set()
    for y in years:
        out |= set(hol.country_holidays("ES", subdiv="IB", years=y).keys())
    return out


def load_panel_daytime(panel_csv: Path, cap_k: float = R.DEFAULT_CAP_K) -> pd.DataFrame:
    """Daytime-only (8-20) panel with a per-series integer step `t` (the NF time
    index, freq=1) and `ds` = the real timestamp (time logic + matching)."""
    df = pd.read_csv(panel_csv, low_memory=False)
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.dropna(subset=["unique_id", "y"]).copy()
    df["unique_id"] = df["unique_id"].map(R._slugify_uid)
    df = df.drop_duplicates(subset=["unique_id", "ds"], keep="first")
    df["hour"] = df["ds"].dt.hour
    df = df[df["hour"].between(8, 20)].copy()                       # DAYTIME only — no night padding
    df["day_of_week"] = df["ds"].dt.dayofweek
    df["month"] = df["ds"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    hol = _es_holidays(range(int(df["ds"].dt.year.min()), int(df["ds"].dt.year.max()) + 1))
    df["is_holiday"] = df["ds"].dt.normalize().isin({pd.Timestamp(h) for h in hol}).astype(int)
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    df["t"] = df.groupby("unique_id").cumcount()                   # NF integer time index (freq=1)
    df["y_raw"] = df["y"]
    df = R.cap_real_outliers(df, cap_k)
    print(f"[daytime] {len(df):,} daytime rows, {df['unique_id'].nunique()} series, "
          f"ds {df['ds'].min().date()} -> {df['ds'].max().date()}")
    return df


def split_train_test(df: pd.DataFrame, protocol: str, test_start: str, test_end: str):
    ts, te = pd.Timestamp(test_start), pd.Timestamp(test_end)
    train = df[df["ds"].dt.year == 2022].copy()
    test = df[(df["ds"] >= ts) & (df["ds"] <= te)].copy()
    if protocol == "cross_year":
        overlap = sorted(set(train["unique_id"]) & set(test["unique_id"]))
        train = train[train["unique_id"].isin(overlap)]
        test = test[test["unique_id"].isin(overlap)]
    return train.reset_index(drop=True), test.reset_index(drop=True)


# Multi-scenario fixed validation (mirrors run_a2_scenarios.py), all on the
# daytime frame so every family issues the same 180-hour daytime trajectory.
SCENARIOS = {
    "S1": ("2022", "2025-06-01", "2025-08-31"),   # cross-year: train 2022 -> test Jun-Aug
    "S3": ("pre",  "2025-04-01", "2025-09-30"),   # full season: train pre-Apr-2025 -> test Apr-Sep
    "S4": ("pre",  "2025-09-01", "2025-09-30"),   # recent month: train pre-Sep-2025 -> test Sep
}


def split_scenario(df: pd.DataFrame, scenario: str):
    era, ts, te = SCENARIOS[scenario]
    if era == "2022":
        train = df[df["ds"].dt.year == 2022]
    else:                                          # everything strictly before the test window
        train = df[df["ds"] < pd.Timestamp(ts)]
    test = df[(df["ds"] >= pd.Timestamp(ts)) & (df["ds"] <= pd.Timestamp(te))]
    return train.reset_index(drop=True), test.reset_index(drop=True), ts, te


def load_op_capacity(db_path: str) -> dict:
    """Per-camera operational capacity = WebCam.max_crowd_count (the production
    classification key), keyed by the slugified unique_id."""
    import sqlite3
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT camera_slug, max_crowd_count FROM webcam_webcam "
                           "WHERE max_crowd_count > 0").fetchall()
    finally:
        con.close()
    return {R._slugify_uid(s): float(m) for s, m in rows}


def issuance_dates(test_df: pd.DataFrame, weekday: int = 0) -> list[pd.Timestamp]:
    if test_df.empty:
        return []
    lo, hi = test_df["ds"].min().normalize(), test_df["ds"].max().normalize()
    out, cur = [], lo
    while cur <= hi:
        if cur.weekday() == weekday:
            out.append(cur)
        cur += pd.Timedelta(days=1)
    return out


def load_prior_hp(db: str, campaign: str, family: str, h_days: int) -> dict:
    """Best Optuna params for one (model, horizon) from a prior daytime campaign
    (e.g. the 150-trial cy_capped run), remapped to the keys build_nf_model wants.
    Skips the slow per-fit HP search and reuses a well-tuned, small config."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.load_study(study_name=f"{campaign}__{family}__{h_days}d",
                              storage=f"sqlite:///{db}")
    hp = dict(study.best_trial.params)
    if family == "lstm":                                    # servable -> retrain key names
        hp.setdefault("encoder_hidden_size", hp.get("hidden_size", 128))
        hp.setdefault("encoder_dropout", hp.get("dropout", 0.1))
        hp.setdefault("encoder_n_layers", hp.get("encoder_n_layers", 2))
    return hp


# ── NeuralForecast on the daytime integer frame (batched inference) ──────────

def _futr_hist(df):
    futr = [c for c in (CALENDAR_FUTR + R.CASTELLE_WEATHER) if c in df.columns]
    hist = [c for c in R.TFT_HIST if c in df.columns and c not in futr]   # no futr/hist overlap
    return futr, hist


def collect_origins(panel, test, issuances, input_size, h_steps, max_n=None):
    """List of (beach, issue, ctx_rows, tgt_rows) for each rolling origin with
    enough context and a full target window."""
    out = []
    for uid, tb in test.groupby("unique_id"):
        pb = panel[panel["unique_id"] == uid]
        for issue in issuances:
            ctx = pb[pb["ds"] < issue].tail(input_size)
            tgt = tb[tb["ds"] >= issue].head(h_steps)
            if len(ctx) < input_size or len(tgt) < h_steps:
                continue
            out.append((uid, issue, ctx, tgt))
            if max_n and len(out) >= max_n:
                return out
    return out


def batch_predict(nf, col, origins, static_df, futr, hist):
    """ONE nf.predict for ALL (beach, origin) pairs via pseudo-series
    (unique_id = beach__i) — kills the per-origin Lightning-trainer overhead."""
    if not origins:
        return []
    cparts, fparts, meta, st_rows = [], [], [], []
    stat_idx = static_df.set_index("unique_id") if static_df is not None else None
    for i, (beach, issue, ctx, tgt) in enumerate(origins):
        if stat_idx is not None and beach not in stat_idx.index:
            continue                                   # no static for this beach -> the static model can't predict it; drop it (keeps df and static aligned)
        pid = f"{beach}__{i}"
        n = len(ctx)
        c = ctx[["y"] + futr + hist].copy()
        c.insert(0, "unique_id", pid); c["ds"] = range(n)
        f = tgt[futr].copy()
        f.insert(0, "unique_id", pid); f["ds"] = range(n, n + len(tgt))
        cparts.append(c); fparts.append(f)
        meta.append((pid, beach, issue, tgt))
        if stat_idx is not None:
            r = stat_idx.loc[beach].to_dict(); r["unique_id"] = pid; st_rows.append(r)
    if not cparts:
        return []
    st = pd.DataFrame(st_rows) if st_rows else None
    out = nf.predict(df=pd.concat(cparts, ignore_index=True),
                     futr_df=pd.concat(fparts, ignore_index=True), static_df=st)
    if col not in out.columns:
        cands = [c for c in out.columns if c not in {"unique_id", "ds"}]
        out = out.rename(columns={cands[0]: col})
    pmap = {pid: g[col].clip(lower=0).values for pid, g in out.groupby("unique_id")}
    res = []
    for pid, beach, issue, tgt in meta:
        yp = pmap.get(pid, np.array([]))[:len(tgt)]
        if len(yp):
            res.append((beach, issue, tgt, yp))
    return res


def nf_search(family, h_steps, train, static_df, futr, hist, capacity, trials, startup,
              trial_steps=R.NF_TRIAL_MAX_STEPS):
    import optuna
    from neuralforecast import NeuralForecast
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cut = train["t"].quantile(0.82)
    cols = ["unique_id", "ds", "y"] + list(dict.fromkeys(futr + hist))
    tr = train[train["t"] < cut]
    va = train[train["t"] >= cut]

    def objective(trial):
        hp = R.nf_search_space(trial, family)
        model, col = R.build_nf_model(family, h_steps, hp, futr, hist, trial_steps)
        nf = NeuralForecast(models=[model], freq=1)
        try:
            nf.fit(df=tr.assign(ds=tr["t"])[cols], static_df=static_df, val_size=h_steps)
        except Exception as e:
            print(f"  [trial fail] {e}"); return 1e6
        origins = collect_origins(train, va, issuance_dates(va), model.input_size, h_steps, max_n=12)
        rows = [pd.DataFrame({"unique_id": b, "y_true": t["y"].values[:len(yp)], "y_pred": yp})
                for b, _, t, yp in batch_predict(nf, col, origins, static_df, futr, hist)]
        if not rows:
            return 1e6
        return R.per_series_rel_mae(pd.concat(rows, ignore_index=True), capacity)[0]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=startup))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    return dict(study.best_params), study.best_value


def nf_fit_predict(family, h_steps, hp, train, test, panel, static_df, futr, hist, seed,
                   final_steps=R.NF_FINAL_MAX_STEPS):
    from neuralforecast import NeuralForecast
    model, col = R.build_nf_model(family, h_steps, {**hp, "seed": seed}, futr, hist, final_steps)
    nf = NeuralForecast(models=[model], freq=1)
    cols = ["unique_id", "ds", "y"] + list(dict.fromkeys(futr + hist))
    nf.fit(df=train.assign(ds=train["t"])[cols], static_df=static_df, val_size=h_steps)
    origins = collect_origins(panel, test, issuance_dates(test), model.input_size, h_steps)
    recs = []
    for beach, issue, tgt, yp in batch_predict(nf, col, origins, static_df, futr, hist):
        recs.append(pd.DataFrame({"unique_id": beach, "ds": tgt["ds"].values[:len(yp)],
                                  "issue_date": issue, "y_true": tgt["y"].values[:len(yp)],
                                  "y_pred": yp}))
    return pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()


# ── Recursive XGB on the daytime frame ───────────────────────────────────────

def xgb_recursive(train, test, panel, capacity, futr, trials, seed, h_steps, static_src, prior_hp=None):
    import optuna
    from xgboost import XGBRegressor
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    feats_cal = [c for c in CALENDAR_FUTR + R.CASTELLE_WEATHER if c in panel.columns]
    # Static level from the SAME pre-test source as the TFT static covariates, so
    # zero-shot (2025-only) test beaches get their level too — not just 2022 train.
    bp90 = static_src.groupby("unique_id")["y"].quantile(0.9).to_dict()
    bmean = static_src.groupby("unique_id")["y"].mean().to_dict()

    def feat_frame(df):
        d = df.sort_values(["unique_id", "t"]).copy()
        g = d.groupby("unique_id")
        d["y_lag1"] = d["y"]
        d["y_roll12"] = g["y"].transform(lambda s: s.rolling(12, min_periods=1).mean())
        d["y_roll36"] = g["y"].transform(lambda s: s.rolling(36, min_periods=1).mean())
        d["beach_p90"] = d["unique_id"].map(bp90)
        d["beach_mean"] = d["unique_id"].map(bmean)
        nxt = g[feats_cal + ["y"]].shift(-1)
        for c in feats_cal:
            d[c + "_n"] = nxt[c]
        d["y_target"] = nxt["y"]
        return d

    cols = ["y_lag1", "y_roll12", "y_roll36", "beach_p90", "beach_mean"] + [c + "_n" for c in feats_cal]
    tr = feat_frame(train).dropna(subset=cols + ["y_target"])
    cut = tr["t"].quantile(0.82)
    a, b = tr[tr["t"] < cut], tr[tr["t"] >= cut]

    if prior_hp is not None:
        best = {k: v for k, v in prior_hp.items() if k not in {"seed"}}
    else:
        def objective(trial):
            hp = R.xgb_search_space(trial)
            m = XGBRegressor(**hp, tree_method="hist", n_jobs=-1, random_state=SEED, verbosity=0)
            m.fit(a[cols].values, a["y_target"].values, verbose=False)
            return float(np.mean(np.abs(m.predict(b[cols].values) - b["y_target"].values)))
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED, n_startup_trials=min(12, trials)))
        study.optimize(objective, n_trials=trials, show_progress_bar=False)
        best = dict(study.best_params)
    model = XGBRegressor(**best, tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0)
    model.fit(tr[cols].values, tr["y_target"].values, verbose=False)

    iss = issuance_dates(test)
    recs = []
    for uid, tb in test.groupby("unique_id"):
        pb = panel[panel["unique_id"] == uid].sort_values("t")
        ds = pb["ds"].to_numpy(); y = pb["y"].to_numpy(float)
        Wn = pb[feats_cal].to_numpy(float)
        bp, bm = bp90.get(uid, np.nan), bmean.get(uid, np.nan)
        if bp != bp:
            continue
        for issue in iss:
            tgt = tb[tb["ds"] >= issue].head(h_steps)
            if len(tgt) < h_steps:
                continue
            p = int(np.searchsorted(ds, np.datetime64(issue), side="left")) - 1
            if p < 36:
                continue
            buf = list(y[max(0, p - 36):p + 1]); cur = y[p]
            tgset = set(pd.to_datetime(tgt["ds"]).to_numpy())
            for k in range(1, h_steps + 1):
                ti = p + k
                if ti >= len(y):
                    break
                row = [cur, float(np.mean(buf[-12:])), float(np.mean(buf[-36:])), bp, bm] + list(Wn[ti])
                yp = max(float(model.predict(np.asarray(row, dtype=float)[None, :])[0]), 0.0)
                if ds[ti] in tgset:
                    recs.append((uid, pd.Timestamp(ds[ti]), issue, y[ti], yp))
                buf.append(yp); cur = yp
    out = pd.DataFrame(recs, columns=["unique_id", "ds", "issue_date", "y_true", "y_pred"])
    return out, best


# ── Driver ───────────────────────────────────────────────────────────────────

def summarise(preds, capacity, model, h_days, protocol, seed, op_capacity=None):
    d = preds.copy()
    d["month"] = pd.to_datetime(d["ds"]).dt.month
    d["abs_err"] = (d["y_pred"] - d["y_true"]).abs()
    base = {"model": model, "horizon_days": h_days, "protocol": protocol, "seed": seed,
            "n_rows": len(d), "n_beaches": d["unique_id"].nunique()}
    if d.empty:
        return base
    out = {**base,
           "relMAE_all": R.per_series_rel_mae(d, capacity)[0],
           "relMAE_season": R.per_series_rel_mae(d[d["month"].isin(SEASON_MONTHS)], capacity)[0],
           "relMAE_summer": R.per_series_rel_mae(d[d["month"].isin(SUMMER_MONTHS)], capacity)[0],
           "mae_users": float(d["abs_err"].mean())}
    if op_capacity:                                    # operational denominator = WebCam.max_crowd_count
        out["relMAE_op_season"] = R.per_series_rel_mae(d[d["month"].isin(SEASON_MONTHS)], op_capacity)[0]
        out["relMAE_op_summer"] = R.per_series_rel_mae(d[d["month"].isin(SUMMER_MONTHS)], op_capacity)[0]
    return out


def main():
    # MPS + the op-fallback thrashes the TFT (CPU<->GPU copies stall the run); force
    # CPU on Apple Silicon. No-op on CUDA servers (this only hides the MPS backend).
    import os
    import torch
    torch.backends.mps.is_available = lambda: False
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS") or min(10, os.cpu_count() or 8)))  # cap per-process threads (parallel scenarios set TORCH_THREADS to avoid 3x10-thread thrashing)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--panel", required=True, type=Path)
    ap.add_argument("--protocol", default="cache2022", choices=["cache2022", "cross_year"])
    ap.add_argument("--scenario", default=None, choices=["S1", "S3", "S4"],
                    help="multi-scenario fixed validation (overrides --protocol/--test-*)")
    ap.add_argument("--op-db", dest="op_db", default=None,
                    help="Django sqlite for WebCam.max_crowd_count (operational denominator)")
    ap.add_argument("--horizons", type=int, nargs="*", default=[3, 10, 15])
    ap.add_argument("--families", nargs="*", default=["tft", "lstm", "xgb"])
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--startup", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trial-steps", dest="trial_steps", type=int, default=R.NF_TRIAL_MAX_STEPS,
                    help="NF training steps per HP trial (lower = faster, e.g. 60 for MPS)")
    ap.add_argument("--final-steps", dest="final_steps", type=int, default=R.NF_FINAL_MAX_STEPS,
                    help="NF training steps for the final refit (e.g. 150 for MPS)")
    ap.add_argument("--prior-db", dest="prior_db", default=None,
                    help="Optuna SQLite db with a prior HP campaign — skip the slow per-fit search")
    ap.add_argument("--prior-campaign", dest="prior_campaign", default=None,
                    help="campaign prefix for the best HP (e.g. cy_capped_backup_20260623_115630)")
    ap.add_argument("--test-start", dest="test_start", default="2025-06-01")
    ap.add_argument("--test-end", dest="test_end", default="2026-02-28")
    ap.add_argument("--cap-k", dest="cap_k", type=float, default=R.DEFAULT_CAP_K)
    ap.add_argument("--out", type=Path, default=Path("new_training_pipeline/validated_daytime"))
    args = ap.parse_args()

    panel = load_panel_daytime(args.panel, args.cap_k)
    op_capacity = load_op_capacity(args.op_db) if args.op_db else None
    if args.scenario:
        train, test, ts, te = split_scenario(panel, args.scenario)
        static_src = panel[panel["ds"] < pd.Timestamp(ts)]
        args.protocol = args.scenario                  # label every output by the scenario
    else:
        train, test = split_train_test(panel, args.protocol, args.test_start, args.test_end)
        static_src = panel[panel["ds"] < pd.Timestamp(args.test_start)] if args.protocol == "cache2022" else train
    capacity = R.compute_capacity(static_src)
    static_df = R.build_static_df(static_src)
    futr, hist = _futr_hist(panel)
    out_dir = args.out / args.protocol
    out_dir.mkdir(parents=True, exist_ok=True)
    if op_capacity:
        n_map = len(set(op_capacity) & set(test["unique_id"]))
        print(f"[daytime] operational capacity loaded: {len(op_capacity)} cameras, "
              f"{n_map}/{test['unique_id'].nunique()} test beaches mapped")
    print(f"[daytime] protocol={args.protocol} train={len(train):,} test={len(test):,} "
          f"beaches={test['unique_id'].nunique()} futr={futr} hist={hist}")

    agg, dm_rows, matched_all = [], [], []
    for h in args.horizons:
        h_steps = R.HORIZON_HOURS[h]
        preds_by = {}
        for fam in args.families:
            if fam in ("tft", "lstm"):
                if args.prior_campaign:
                    hp = load_prior_hp(args.prior_db, args.prior_campaign, fam, h)
                    print(f"[{fam} {h}d] prior-best HP (no search): {hp}")
                else:
                    hp, val = nf_search(fam, h_steps, train, static_df, futr, hist, capacity,
                                        args.trials, args.startup, trial_steps=args.trial_steps)
                    print(f"[{fam} {h}d] best inner relMAE={val:.4f}")
                first = None
                for sd in range(SEED, SEED + args.seeds):
                    p = nf_fit_predict(fam, h_steps, hp, train, test, panel, static_df, futr, hist, sd,
                                       final_steps=args.final_steps)
                    agg.append(summarise(p, capacity, fam.upper(), h, args.protocol, sd, op_capacity))
                    if first is None:
                        first = p; p.to_csv(out_dir / f"per_row_{fam}_{h}d.csv", index=False)
                (out_dir / f"best_params_{fam}_{h}d.json").write_text(json.dumps(hp, indent=2))
                preds_by[fam.upper()] = first
                print(f"[{fam} {h}d] done ({first['unique_id'].nunique() if first is not None and len(first) else 0} beaches)")
            else:
                prior = load_prior_hp(args.prior_db, args.prior_campaign, "xgb", h) if args.prior_campaign else None
                first = None
                for sd in range(SEED, SEED + args.seeds):
                    p, hp = xgb_recursive(train, test, panel, capacity, futr, args.trials, sd, h_steps, static_src,
                                          prior_hp=prior)
                    agg.append(summarise(p, capacity, "XGB", h, args.protocol, sd, op_capacity))
                    if first is None:
                        first = p; p.to_csv(out_dir / f"per_row_xgb_{h}d.csv", index=False)
                        (out_dir / f"best_params_xgb_{h}d.json").write_text(json.dumps(hp, indent=2))
                preds_by["XGB"] = first
                print(f"[xgb {h}d] done")

        keys = ["unique_id", "ds", "issue_date"]
        merged = None
        for name, p in preds_by.items():
            if p is None or p.empty:
                continue
            d = p.copy(); d["ds"] = pd.to_datetime(d["ds"]); d["issue_date"] = pd.to_datetime(d["issue_date"])
            d = d[keys + ["y_true", "y_pred"]].rename(columns={"y_pred": name})
            merged = d if merged is None else merged.merge(d.drop(columns=["y_true"]), on=keys, how="inner")
        if merged is None or merged.empty:
            continue
        merged["lead_d"] = (merged["ds"] - merged["issue_date"]).dt.days
        merged.to_csv(out_dir / f"identical_rows_{h}d.csv", index=False)
        for fam in [f.upper() for f in args.families]:
            if fam in merged.columns:
                matched_all.append({**summarise(merged.rename(columns={fam: "y_pred"}),
                                                capacity, fam, h, args.protocol, -1, op_capacity), "matched": True})
        for a, b in [("TFT", "LSTM"), ("TFT", "XGB")]:
            if a in merged.columns and b in merged.columns:
                r = R.dm_pair(merged, a, b, h_steps)
                if r:
                    dm_rows.append({"protocol": args.protocol, "horizon_days": h, **r})

    pd.DataFrame(agg).to_csv(out_dir / "metrics_summary.csv", index=False)
    if matched_all:
        pd.DataFrame(matched_all).to_csv(out_dir / "matched_metrics.csv", index=False)
    if dm_rows:
        pd.DataFrame(dm_rows).to_csv(out_dir / "dm_results.csv", index=False)
    print(f"[daytime] done -> {out_dir}")


if __name__ == "__main__":
    main()
