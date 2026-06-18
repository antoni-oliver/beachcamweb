#!/usr/bin/env python3
"""
LSTM Hourly Training — Beach Occupancy Prediction

Same protocol as tft_train_3_models.py with a NeuralForecast LSTM model.
Saves to lstm_model_<H>/ subdirs whose layout is compatible with tft_service
auto-discovery (NeuralForecast.load handles both TFT and LSTM transparently).

Data is filtered 8:00-20:00 → 12 hours/day.
Trains per-horizon models: lstm_model_3d/, lstm_model_10d/, lstm_model_15d/.

Phases mirror the TFT script. Fixed: encoder_hidden_size=64, encoder_n_layers=2,
lr=0.001, input_size=48. The dropout sweep tunes encoder_dropout.

Output structure:
    <prefix>_<timestamp>/
        results/           — CSVs, checkpoints
        lstm_model_3d/     — model artifacts (config.json + nf_model/ + static_features.csv)
        lstm_model_10d/
        lstm_model_15d/

Usage matches the TFT script — replace `tft_train.py` with `lstm_train_3_models.py`.
"""

import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from crowd_outliers import cap_outliers  # canonical per-series P90 outlier cap

warnings.filterwarnings('ignore')

SEED = 42
SEASON_EVAL = [4, 5, 6, 7, 8, 9]
STAT_EXOG = ['stat_mean_y', 'stat_cv']
HOURS_PER_DAY = 12


TARGET_HORIZONS = {
    '3d': 36,
    '10d': 120,
    '15d': 180,
}


def log(msg, level='INFO'):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] [{level}] {msg}", flush=True)


def _prediction_dir() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_data_dir(p: str) -> str:
    pp = Path(p)
    if pp.is_absolute() and pp.exists():
        return str(pp)
    for c in (pp, _prediction_dir() / pp, _prediction_dir().parent / pp):
        if c.exists():
            return str(c.resolve())
    return str(_prediction_dir() / pp)


def make_run_dir(prefix):
    """Create a unique run directory: tft_models/<prefix>_<YYYYMMDD_HHMMSS>/"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    root = _prediction_dir() / 'tft_models'
    root.mkdir(parents=True, exist_ok=True)
    run_dir = str(root / f'{prefix}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    log(f"Run directory: {run_dir}")
    return run_dir


def detect_cuda_device():
    """Find the best available CUDA device. Returns device id string or empty for CPU."""
    try:
        import torch
        if not torch.cuda.is_available():
            log("No CUDA available, using CPU")
            return ''
        count = torch.cuda.device_count()
        if count == 0:
            log("No CUDA devices found, using CPU")
            return ''
        best_id = 0
        best_free = 0
        for i in range(count):
            free, total = torch.cuda.mem_get_info(i)
            name = torch.cuda.get_device_name(i)
            log(f"GPU {i}: {name} — {free/1e9:.1f}/{total/1e9:.1f} GB free")
            if free > best_free:
                best_free = free
                best_id = i
        log(f"Selected GPU {best_id} ({best_free/1e9:.1f} GB free)")
        return str(best_id)
    except Exception as e:
        log(f"CUDA detection failed: {e}, using CPU", 'WARN')
        return ''


def save_checkpoint(results_dir, name, data):
    os.makedirs(results_dir, exist_ok=True)
    path = f'{results_dir}/_checkpoint_{name}.json'
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


# ── Data Helpers ─────────────────────────────────────────────────────────────

def build_panel(cache, django, futr_exog, hist_exog, season_months=None):
    parts = []
    if cache is not None:
        c = cache.copy()
        c['period'], c['original_id'] = 'cache_2022', c['unique_id']
        c['unique_id'] = c['unique_id'] + '_2022'
        parts.append(c)
    if django is not None:
        d = django.copy()
        d['period'], d['original_id'] = 'django_2025', d['unique_id']
        d['unique_id'] = d['unique_id'] + '_2025'
        parts.append(d)

    df = pd.concat(parts, ignore_index=True)
    before_nan = len(df)
    df = df.dropna(subset=['y'])
    if len(df) < before_nan:
        log(f"Dropped {before_nan - len(df):,} rows with NaN y")
    df, _ = cap_outliers(df)          # per-series P90 outlier cap on the real y
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)

    if season_months is not None:
        before = len(df)
        df = df[df['month'].isin(season_months)].copy()
        log(f"Season filter {season_months}: {before:,} -> {len(df):,} rows")

    for col in [x for x in hist_exog + futr_exog if x in df.columns]:
        df[col] = df.groupby('unique_id')[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median()).fillna(0)

    df = df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    df['ds_real'] = df['ds']
    df['ds'] = df.groupby('unique_id').cumcount()

    counts = df.groupby('unique_id').size()
    df = df[df['unique_id'].isin(counts[counts >= 96].index)]

    stats = df.groupby('unique_id').agg(
        stat_mean_y=('y', 'mean'),
        stat_cv=('y', lambda x: x.std() / x.mean() if x.mean() > 0 else 0),
    ).reset_index()
    static_df = stats[['unique_id'] + STAT_EXOG]

    available = [c for c in set(futr_exog + hist_exog) if c in df.columns]
    keep = ['unique_id', 'ds', 'ds_real', 'y', 'original_id', 'period'] + available
    return df[keep], static_df, [c for c in futr_exog if c in available], [c for c in hist_exog if c in available]


def compute_rel_mae(cv_df, capacity_dict):
    results = []
    for uid in cv_df['unique_id'].unique():
        sub = cv_df[cv_df['unique_id'] == uid]
        mae = float(np.mean(np.abs(sub['y'].values - sub['TFT'].values)))
        rmse = float(np.sqrt(np.mean((sub['y'].values - sub['TFT'].values) ** 2)))
        r2 = r2_score(sub['y'], sub['TFT']) if len(sub) > 1 else 0
        p90 = max(capacity_dict.get(uid, float(sub['y'].quantile(0.9))), 1)
        results.append({'unique_id': uid, 'P90': p90, 'MAE': mae, 'RMSE': rmse, 'R2': r2, 'relMAE': mae / p90})
    rdf = pd.DataFrame(results)
    return rdf['relMAE'].mean(), rdf


# ── Subprocess Worker ────────────────────────────────────────────────────────

def _worker_main(work_dir):
    cuda_device = os.environ.get('TFT_CUDA_DEVICE', '')
    os.environ['CUDA_VISIBLE_DEVICES'] = cuda_device

    import torch
    torch.set_float32_matmul_precision('medium')
    import logging as _l
    _l.getLogger('pytorch_lightning').setLevel(_l.WARNING)
    _l.getLogger('lightning').setLevel(_l.WARNING)
    from neuralforecast import NeuralForecast
    from neuralforecast.models import LSTM
    from neuralforecast.losses.pytorch import MAE
    np.random.seed(SEED)

    with open(f'{work_dir}/input.pkl', 'rb') as f:
        args = pickle.load(f)

    panel_df, static_df = args['panel_df'], args['static_df']
    futr, hist = args['futr'], args['hist']
    input_size, horizon = args['input_size'], args['horizon']
    kw = args.get('kwargs', {})
    mode = args.get('mode', 'cv')

    lstm = LSTM(
        h=horizon, input_size=input_size,
        encoder_n_layers=kw.get('encoder_n_layers', 2),
        encoder_hidden_size=kw.get('hidden_size', 64),
        decoder_hidden_size=kw.get('hidden_size', 64),
        decoder_layers=kw.get('decoder_layers', 2),
        encoder_dropout=kw.get('dropout', 0.1),
        learning_rate=kw.get('lr', 1e-3), batch_size=32,
        max_steps=kw.get('max_steps', 500) if not os.environ.get('TFT_LIGHT') else min(kw.get('max_steps', 500), 50),
        early_stop_patience_steps=20 if mode == 'cv' else 30,
        scaler_type='minmax', loss=MAE(),
        futr_exog_list=futr if futr else None,
        hist_exog_list=hist if hist else None,
        stat_exog_list=STAT_EXOG,
        val_check_steps=50, random_seed=SEED, start_padding_enabled=True,
    )
    nf = NeuralForecast(models=[lstm], freq=1)
    train_cols = ['unique_id', 'ds', 'y'] + list(dict.fromkeys(futr + hist))

    if mode == 'cv':
        n_windows = kw.get('n_windows', 3)
        cv = nf.cross_validation(
            df=panel_df[train_cols].copy(), static_df=static_df,
            n_windows=n_windows, step_size=horizon, val_size=horizon)
        cv = cv.reset_index().dropna(subset=['y', 'LSTM'])
        cv['LSTM'] = cv['LSTM'].clip(lower=0)
        # Rename so downstream code keeps using the 'TFT' column name uniformly.
        cv = cv.rename(columns={'LSTM': 'TFT'})
        cap = panel_df.groupby('unique_id')['y'].quantile(0.9).to_dict()

        fi_rows = []
        try:
            fi_raw = nf.models[0].feature_importances()
            skip = {'importance', 'feature', 'index', 'level_0'}
            for key, data in fi_raw.items():
                if not isinstance(data, pd.DataFrame) or len(data) == 0:
                    continue
                cols = [c for c in data.columns if str(c) not in skip]
                if cols:
                    for feat, imp in data[cols].mean(axis=0).items():
                        fi_rows.append({'feature': str(feat), 'importance': float(imp)})
        except Exception:
            pass

        fi_df = pd.DataFrame(fi_rows)
        if len(fi_df) > 0:
            fi_df = fi_df.groupby('feature')['importance'].max().reset_index()
            fi_df = fi_df.sort_values('importance', ascending=False).reset_index(drop=True)
        else:
            fi_df = pd.DataFrame(columns=['feature', 'importance'])

        result = {'cv': cv, 'cap': cap, 'fi': fi_df}

    elif mode == 'fit':
        nf.fit(df=panel_df[train_cols].copy(), static_df=static_df, val_size=horizon)
        save_path = args.get('save_path', 'lstm_model/nf_model')
        nf.save(save_path, overwrite=True)
        result = {'saved': save_path}

    with open(f'{work_dir}/output.pkl', 'wb') as f:
        pickle.dump(result, f)


def _run_worker(work_dir, cuda_device, timeout=None):
    env = os.environ.copy()
    env['TFT_CUDA_DEVICE'] = cuda_device
    if os.environ.get('TFT_LIGHT'):
        env['TFT_LIGHT'] = '1'
    r = subprocess.run(
        [sys.executable, os.path.abspath(__file__), '--worker', work_dir],
        capture_output=True, text=True, timeout=timeout, env=env)
    return r


def safe_quick_cv(panel_df, static_df, futr, hist, input_size, horizon,
                  cuda_device='', timeout=None, **kwargs):
    work_dir = tempfile.mkdtemp(prefix='tft_w_')
    try:
        with open(f'{work_dir}/input.pkl', 'wb') as f:
            pickle.dump({
                'panel_df': panel_df, 'static_df': static_df,
                'futr': futr, 'hist': hist,
                'input_size': input_size, 'horizon': horizon,
                'kwargs': kwargs, 'mode': 'cv',
            }, f)
        r = _run_worker(work_dir, cuda_device, timeout)
        if r.returncode != 0:
            raise RuntimeError(f"Worker exit {r.returncode}: {(r.stderr or '')[-800:]}")
        with open(f'{work_dir}/output.pkl', 'rb') as f:
            out = pickle.load(f)
        rel, pb = compute_rel_mae(out['cv'], out['cap'])
        return rel, pb, out['cv'], out['fi']
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def safe_fit(panel_df, static_df, futr, hist, input_size, horizon,
             save_path='lstm_model/nf_model', cuda_device='', timeout=None, **kwargs):
    work_dir = tempfile.mkdtemp(prefix='tft_w_')
    try:
        with open(f'{work_dir}/input.pkl', 'wb') as f:
            pickle.dump({
                'panel_df': panel_df, 'static_df': static_df,
                'futr': futr, 'hist': hist,
                'input_size': input_size, 'horizon': horizon,
                'kwargs': kwargs, 'mode': 'fit', 'save_path': save_path,
            }, f)
        r = _run_worker(work_dir, cuda_device, timeout)
        if r.returncode != 0:
            raise RuntimeError(f"Fit exit {r.returncode}: {(r.stderr or '')[-800:]}")
        with open(f'{work_dir}/output.pkl', 'rb') as f:
            return pickle.load(f)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ── Data Loading ─────────────────────────────────────────────────────────────


def _subsample_panel(panel_df, static_df, fraction=0.1, min_series=3):
    """Keep a random fraction of series for light/test runs."""
    uids = panel_df['unique_id'].unique()
    n = max(min_series, int(len(uids) * fraction))
    chosen = np.random.choice(uids, size=min(n, len(uids)), replace=False)
    panel_s = panel_df[panel_df['unique_id'].isin(chosen)].copy()
    static_s = static_df[static_df['unique_id'].isin(chosen)].copy()
    return panel_s, static_s

def load_data(data_dir):
    log("Loading data...")
    cache_df = pd.read_csv(f'{data_dir}/cache_2022_clean.csv', parse_dates=['ds'])
    django_df = pd.read_csv(f'{data_dir}/django_2025_clean.csv', parse_dates=['ds'])

    all_clean_path = f'{data_dir}/all_clean.csv'
    if os.path.exists(all_clean_path):
        all_clean = pd.read_csv(all_clean_path)
        valid = all_clean['unique_id'].unique().tolist()
        cache_df = cache_df[cache_df['unique_id'].isin(valid)]
        django_df = django_df[django_df['unique_id'].isin(valid)]
        log(f"Filtered to {len(valid)} quality beaches")

    log(f"Cache 2022:  {len(cache_df):,} rows, {cache_df['unique_id'].nunique()} beaches")
    log(f"Django 2025: {len(django_df):,} rows, {django_df['unique_id'].nunique()} beaches")
    return cache_df, django_df


def define_features(cache_df, django_df):
    temporal = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_summer']
    all_cols = set(cache_df.columns) | set(django_df.columns)
    om_only = sorted([c for c in all_cols if c.startswith('om_')])

    all_futr = temporal + om_only
    om_hist = om_only

    log(f"TEMPORAL: {temporal}")
    log(f"OM weather: {len(om_only)} features")
    return all_futr, om_hist


# ── Phase 0: Horizon Limit Analysis (optional) ──────────────────────────────

def phase0(panel_df, static_df, futr, hist, best_input, best_hp, beach_cap,
           results_dir, cuda_device=''):
    log("=" * 70)
    log("PHASE 0: Horizon Limit Analysis (12h→168h, stop on failure)")

    ds_real_map = panel_df[['unique_id', 'ds', 'ds_real']].drop_duplicates()
    hz_log = []

    for hz in range(12, 168 + 1, 12):
        try:
            t0 = time.time()
            rel, pb, cv_h, _ = safe_quick_cv(
                panel_df, static_df, futr, hist, best_input, hz,
                cuda_device=cuda_device, **best_hp)
            mae_g = mean_absolute_error(cv_h['y'], cv_h['TFT'])
            elapsed = time.time() - t0

            cv_h = cv_h.merge(ds_real_map, on=['unique_id', 'ds'], how='left')
            cv_h['month'] = cv_h['ds_real'].dt.month
            cv_season = cv_h[cv_h['month'].isin(SEASON_EVAL)]
            season_rel = compute_rel_mae(cv_season, beach_cap)[0] if len(cv_season) > 10 else float('nan')

            days = hz / HOURS_PER_DAY
            hz_log.append({
                'horizon': hz, 'days': days,
                'relMAE': rel, 'MAE': mae_g,
                'season_relMAE': season_rel, 'time_s': elapsed,
            })
            log(f"H={hz:3d} ({days:.0f}d) -> relMAE={rel:.4f}  season={season_rel:.4f} ({elapsed:.0f}s)")
            pd.DataFrame(hz_log).to_csv(f'{results_dir}/horizon_analysis.csv', index=False)

        except Exception as e:
            log(f"H={hz} FAILED: {e}", 'WARN')
            log(f"Stopping. Completed {len(hz_log)} horizons")
            break

    save_checkpoint(results_dir, 'phase0', {'completed': len(hz_log), 'status': 'complete'})
    return pd.DataFrame(hz_log) if hz_log else pd.DataFrame()


# ── Phase 1: Feature Selection (per horizon) ────────────────────────────────

def phase1(panel_df, static_df, futr_exog, hist_exog, horizon, results_dir, cuda_device=''):
    log(f"PHASE 1: Feature Selection (H={horizon}, backward elimination, stop at 2x baseline)")
    INP = 48

    try:
        base_rel, _, _, base_fi = safe_quick_cv(
            panel_df, static_df, futr_exog, hist_exog, INP, horizon, cuda_device=cuda_device)
    except Exception as e:
        log(f"Baseline failed: {e}", 'ERROR')
        return futr_exog, hist_exog

    log(f"Baseline: relMAE={base_rel:.4f} with {len(hist_exog)} hist features")
    stop_threshold = base_rel * 2.0

    cur_hist = hist_exog.copy()
    flog = [{'step': 0, 'n_hist': len(cur_hist), 'relMAE': base_rel, 'dropped': '-'}]
    best_rel, best_hist = base_rel, cur_hist.copy()
    fi_cur = base_fi.copy()

    step = 0
    while len(cur_hist) > 3:
        step += 1
        drop_feat = None
        if len(fi_cur) > 0:
            for _, r in fi_cur.sort_values('importance').iterrows():
                if r['feature'] in cur_hist:
                    drop_feat = r['feature']
                    break
        if drop_feat is None:
            drop_feat = cur_hist[-1]

        cur_hist = [f for f in cur_hist if f != drop_feat]

        try:
            rel, _, _, fi_cur = safe_quick_cv(
                panel_df, static_df, futr_exog, cur_hist, INP, horizon, cuda_device=cuda_device)
        except Exception as e:
            log(f"Step {step} failed: {e}", 'WARN')
            break

        if rel < best_rel:
            best_rel, best_hist = rel, cur_hist.copy()
        log(f"Step {step}: drop '{drop_feat}' -> relMAE={rel:.4f}")
        flog.append({'step': step, 'n_hist': len(cur_hist), 'relMAE': rel, 'dropped': drop_feat})

        if rel > stop_threshold:
            log(f"relMAE > 2x baseline -> stopping")
            break

    pd.DataFrame(flog).to_csv(f'{results_dir}/feature_selection_H{horizon}.csv', index=False)
    log(f"SELECTED HIST ({len(best_hist)}): {best_hist}")
    return futr_exog, best_hist


# ── Phase 2: Input Size Sweep (per horizon) ─────────────────────────────────

def phase2(panel_df, static_df, futr, hist, horizon, results_dir, cuda_device=''):
    log(f"PHASE 2: Input Size Sweep (H={horizon})")
    ilog = []
    for isz in [120, 96, 72, 60, 48, 36, 24, 18, 12]:
        try:
            t0 = time.time()
            rel, _, _, _ = safe_quick_cv(
                panel_df, static_df, futr, hist, isz, horizon, cuda_device=cuda_device)
            log(f"input={isz:3d} -> relMAE={rel:.4f} ({time.time()-t0:.0f}s)")
            ilog.append({'input_size': isz, 'relMAE': rel})
        except Exception as e:
            log(f"input={isz} FAILED: {e}", 'WARN')
    if not ilog:
        return 48
    idf = pd.DataFrame(ilog)
    idf.to_csv(f'{results_dir}/input_size_sweep_H{horizon}.csv', index=False)
    best = int(idf.loc[idf['relMAE'].idxmin(), 'input_size'])
    log(f"BEST INPUT: {best}")
    return best


# ── Phase 3: HP Optimization (per horizon) ─��────────────────────────────────

FIXED_HP = {'hidden_size': 64, 'n_head': 4, 'lr': 0.001}
TEMPORAL_FUTR = ['hour', 'day_of_week', 'month', 'is_weekend', 'is_summer']
FUTR_WEATHER = ['om_temperature_2m', 'om_apparent_temperature', 'om_cloud_cover']
SELECTED_FUTR = TEMPORAL_FUTR + FUTR_WEATHER
DROPOUT_VALUES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3]


def phase3(panel_df, static_df, futr_available, hist, best_input, horizon, results_dir, cuda_device=''):
    log(f"PHASE 3: Dropout Sweep (H={horizon})")
    log(f"  Fixed HP: {FIXED_HP}")
    log(f"  Fixed futr: {SELECTED_FUTR}")

    futr = [f for f in SELECTED_FUTR if f in futr_available]
    log(f"  Available futr: {futr}")

    dropout_log = []
    best_dropout_rel = float('inf')
    best_dropout = 0.05

    for d in DROPOUT_VALUES:
        hp_test = {**FIXED_HP, 'dropout': d}
        t0 = time.time()
        try:
            rel, _, _, _ = safe_quick_cv(
                panel_df, static_df, futr, hist, best_input, horizon,
                cuda_device=cuda_device, **hp_test)
            log(f"    dropout={d:.2f} -> relMAE={rel:.5f} ({time.time()-t0:.0f}s)")
            dropout_log.append({'dropout': d, 'relMAE': rel})
            if rel < best_dropout_rel:
                best_dropout_rel, best_dropout = rel, d
        except Exception as e:
            log(f"    dropout={d:.2f} -> FAILED: {e}")

    pd.DataFrame(dropout_log).to_csv(f'{results_dir}/dropout_sweep_H{horizon}.csv', index=False)

    best_hp = {**FIXED_HP, 'dropout': best_dropout}
    log(f"  OPTIMAL H={horizon}: dropout={best_dropout} relMAE={best_dropout_rel:.5f}")

    return futr, best_hp


# ── Phase 4: Deploy Model ───────────────────────────────────────────────────

def phase4(label, horizon, cache_df, django_df, all_futr, om_hist,
           selected_futr, selected_hist, best_input, best_hp,
           run_dir, results_dir, cuda_device=''):
    model_dir = f'{run_dir}/lstm_model_{label}'
    os.makedirs(model_dir, exist_ok=True)

    days = horizon / HOURS_PER_DAY
    log(f"\n  === PHASE 4: Deploying {label} model (H={horizon}, {days:.0f} days) ===")

    try:
        panel, static, futr, hist = build_panel(cache_df, django_df, all_futr, om_hist)
        futr = [f for f in selected_futr if f in futr]
        hist = [f for f in selected_hist if f in hist]
        cap = panel.groupby('unique_id')['y'].quantile(0.9).to_dict()

        min_required = best_input + horizon * 3
        series_len = panel.groupby('unique_id').size()
        short = series_len[series_len < min_required].index.tolist()
        if short:
            log(f"  Dropped {len(short)} short series (<{min_required} rows), using {len(series_len)-len(short)}")
            panel = panel[~panel['unique_id'].isin(short)]
            static = static[~static['unique_id'].isin(short)]

        median_len = int(panel.groupby('unique_id').size().median())
        n_win = min(max(2, (median_len - best_input) // horizon - 1), 5)
        log(f"  {panel['unique_id'].nunique()} series, {len(panel):,} rows, n_windows={n_win}")

        hp_cv = best_hp.copy()
        hp_cv['max_steps'] = 1500
        hp_cv['n_windows'] = n_win

        t0 = time.time()
        rel_m, pb_m, cv_m, fi_m = safe_quick_cv(
            panel, static, futr, hist, best_input, horizon,
            cuda_device=cuda_device, **hp_cv)
        elapsed = time.time() - t0

        ds_map = panel[['unique_id', 'ds', 'ds_real']].copy()
        cv_m = cv_m.merge(ds_map, on=['unique_id', 'ds'], how='left')
        cv_m['month'] = cv_m['ds_real'].dt.month

        mae_m = mean_absolute_error(cv_m['y'], cv_m['TFT'])
        r2_m = r2_score(cv_m['y'], cv_m['TFT'])

        cv_season = cv_m[cv_m['month'].isin(SEASON_EVAL)]
        if len(cv_season) > 10:
            season_rel, season_pb = compute_rel_mae(cv_season, cap)
        else:
            season_rel, season_pb = float('nan'), pb_m

        log(f"  overall={rel_m:.4f}  season={season_rel:.4f}  MAE={mae_m:.1f}  R²={r2_m:.4f}  ({elapsed:.0f}s)")

    except Exception as e:
        log(f"  CV evaluation FAILED: {e}", 'ERROR')
        traceback.print_exc()
        return None

    cv_m.to_csv(f'{model_dir}/cv_predictions.csv', index=False)
    pb_m.to_csv(f'{model_dir}/per_beach.csv', index=False)

    try:
        hp_fit = best_hp.copy()
        hp_fit['max_steps'] = 1500
        log(f"  Fitting deployment model...")
        t0 = time.time()
        safe_fit(panel, static, futr, hist,
                 best_input, horizon, save_path=f'{model_dir}/nf_model',
                 cuda_device=cuda_device, **hp_fit)
        log(f"  Fitted in {time.time()-t0:.0f}s")
    except Exception as e:
        log(f"  Fit FAILED: {e}", 'ERROR')
        return None

    static.to_csv(f'{model_dir}/static_features.csv', index=False)
    panel.to_csv(f'{model_dir}/panel_data.csv', index=False)

    config = {
        'horizon': int(horizon),
        'horizon_days': int(days),
        'hours_per_day': HOURS_PER_DAY,
        'input_size': int(best_input),
        'futr_exog': futr,
        'hist_exog': hist,
        'stat_exog': STAT_EXOG,
        'hp': {k: (float(v) if isinstance(v, float) else int(v)) for k, v in best_hp.items()},
        'model_type': 'hourly',
        'season_relMAE': float(season_rel),
        'overall_relMAE': float(rel_m),
        'season_months': SEASON_EVAL,
    }
    with open(f'{model_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=2)

    log(f"  Saved to {model_dir}/")
    return {
        'season_relMAE': season_rel,
        'relMAE': rel_m,
        'model_dir': model_dir,
        'futr': futr,
        'hist': hist,
        'input_size': best_input,
        'hp': best_hp,
    }



# ── Evaluate on additional datasets ──────────────────────────────────────────

def evaluate_on_dataset(data_dir, ds_label, label, horizon, all_futr, om_hist,
                        selected_futr, selected_hist, best_input, best_hp,
                        results_dir, cuda_device=''):
    if not os.path.isdir(data_dir):
        log(f"    [{ds_label}] Skipping — {data_dir} not found")
        return None
    for f in ['cache_2022_clean.csv', 'django_2025_clean.csv']:
        if not os.path.exists(f'{data_dir}/{f}'):
            log(f"    [{ds_label}] Skipping — missing {f} in {data_dir}")
            return None

    log(f"    [{ds_label}] Evaluating {label} on {data_dir}...")
    try:
        cache_df, django_df = load_data(data_dir)
        panel, static, futr, hist = build_panel(cache_df, django_df, all_futr, om_hist)
        futr = [f for f in selected_futr if f in futr]
        hist = [f for f in selected_hist if f in hist]
        cap = panel.groupby('unique_id')['y'].quantile(0.9).to_dict()

        min_required = best_input + horizon * 3
        slen = panel.groupby('unique_id').size()
        short = slen[slen < min_required].index.tolist()
        if short:
            panel = panel[~panel['unique_id'].isin(short)]
            static = static[~static['unique_id'].isin(short)]

        median_len = int(panel.groupby('unique_id').size().median())
        n_win = min(max(2, (median_len - best_input) // horizon - 1), 5)

        hp_cv = best_hp.copy()
        hp_cv['max_steps'] = 1500
        hp_cv['n_windows'] = n_win

        t0 = time.time()
        rel, pb, cv, _ = safe_quick_cv(
            panel, static, futr, hist, best_input, horizon,
            cuda_device=cuda_device, **hp_cv)
        elapsed = time.time() - t0

        ds_map = panel[['unique_id', 'ds', 'ds_real']].drop_duplicates()
        cv = cv.merge(ds_map, on=['unique_id', 'ds'], how='left')
        cv['month'] = cv['ds_real'].dt.month

        cv_season = cv[cv['month'].isin(SEASON_EVAL)]
        season_rel = compute_rel_mae(cv_season, cap)[0] if len(cv_season) > 10 else float('nan')

        mae = mean_absolute_error(cv['y'], cv['TFT'])

        s = f"{season_rel:.4f}" if not np.isnan(season_rel) else 'n/a'
        log(f"    [{ds_label}] overall={rel:.4f} season={s} "
            f"MAE={mae:.1f} ({panel['unique_id'].nunique()} series, {elapsed:.0f}s)")

        pb.to_csv(f'{results_dir}/eval_{label}_{ds_label}.csv', index=False)

        return {
            'dataset': ds_label, 'data_dir': data_dir,
            'relMAE': rel, 'season_relMAE': season_rel, 'MAE': mae,
            'n_series': panel['unique_id'].nunique(),
        }
    except Exception as e:
        log(f"    [{ds_label}] FAILED: {e}", 'ERROR')
        return None


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='LSTM Hourly Training')
    parser.add_argument('--data-dir', default='pipeline_workspace/clean_datasets_210526',
                        help='Data directory (default: pipeline_workspace/clean_datasets_210526, latest synced)')
    parser.add_argument('--prefix', default='lstm_run', help='Output directory prefix (default: lstm_run)')
    parser.add_argument('--cuda-device', default=None,
                        help='CUDA device id (e.g. "0", "1"). Default: auto-detect best available GPU')
    parser.add_argument('--phase0', action='store_true', help='Run optional horizon limit analysis')
    parser.add_argument('--deploy-only', action='store_true', help='Skip optimization, use known config')
    parser.add_argument('--eval-dirs', nargs='*', default=None,
                        help='Directories to evaluate on (default: train dir + all known dirs)')
    parser.add_argument('--skip-eval', action='store_true', help='Skip evaluation phase')
    parser.add_argument('--worker', type=str, default=None, help='Internal subprocess work dir')
    parser.add_argument('--light', action='store_true',
                        help='Light mode: use 10%% of series and reduced HP sweep to test pipeline flow')
    return parser.parse_args()



KNOWN_DATASETS = {
    'pipeline_workspace/clean_datasets_210526': 'latest',
    'pipeline_workspace/clean_datasets':        'default',
    'pipeline_workspace/clean_dataset_backup':  'old',
}


def resolve_eval_dirs(args):
    if args.skip_eval:
        return {}
    if args.eval_dirs is not None:
        dirs = {}
        for d in args.eval_dirs:
            rd = _resolve_data_dir(d)
            if os.path.isdir(rd):
                dirs[rd] = KNOWN_DATASETS.get(d, os.path.basename(rd))
            else:
                log(f"  Eval dir not found, skipping: {d}", 'WARN')
        return dirs
    dirs = {}
    for d, name in KNOWN_DATASETS.items():
        rd = _resolve_data_dir(d)
        if os.path.isdir(rd):
            dirs[rd] = name
    return dirs


def main(args):
    data_dir = _resolve_data_dir(args.data_dir)
    args.data_dir = data_dir
    cuda_device = args.cuda_device if args.cuda_device is not None else detect_cuda_device()

    run_dir = make_run_dir(args.prefix)
    results_dir = f'{run_dir}/results'
    os.makedirs(results_dir, exist_ok=True)
    total_t0 = time.time()

    log("LSTM Training — Per-Horizon Optimization")
    log(f"Target models: {TARGET_HORIZONS}")
    log(f"Hours per day: {HOURS_PER_DAY} (8:00-20:00)")
    log(f"Data dir: {data_dir}")
    log(f"Run dir:  {run_dir}")
    log(f"CUDA device: {cuda_device if cuda_device else 'CPU'}")

    eval_dirs = resolve_eval_dirs(args)
    log(f"Eval dirs: {eval_dirs}")

    cache_df, django_df = load_data(data_dir)
    all_futr, om_hist = define_features(cache_df, django_df)

    log("Building merged panel (OM-only)...")
    panel_df, static_df, futr_exog, hist_exog = build_panel(cache_df, django_df, all_futr, om_hist)
    beach_cap = panel_df.groupby('unique_id')['y'].quantile(0.9).to_dict()
    log(f"Panel: {len(panel_df):,} rows, {panel_df['unique_id'].nunique()} series")

    if args.light:
        log("LIGHT MODE — using 10% of series, reduced sweeps", 'WARN')
        os.environ['TFT_LIGHT'] = '1'
        panel_df, static_df = _subsample_panel(panel_df, static_df, fraction=0.1)
        log(f"Light panel: {len(panel_df):,} rows, {panel_df['unique_id'].nunique()} series")

    if args.phase0:
        try:
            default_hp = {'hidden_size': 64, 'n_head': 4, 'dropout': 0.1, 'lr': 1e-3}
            phase0(panel_df, static_df, futr_exog, hist_exog, 48, default_hp,
                   beach_cap, results_dir, cuda_device=cuda_device)
        except Exception as e:
            log(f"Phase 0 crashed: {e}", 'ERROR')

    model_results = {}
    horizon_configs = {}
    eval_results = {}

    for label, horizon in TARGET_HORIZONS.items():
        log("=" * 70)
        log(f"=== HORIZON {label} (H={horizon}, {horizon/HOURS_PER_DAY:.0f} days) ===")
        log("=" * 70)

        # Phase 1: Hist backward elimination (uses temporal-only futr)
        if args.light:
            sel_hist = ['om_cloud_cover_low', 'om_shortwave_radiation', 'om_vapour_pressure_deficit']
            log(f"  [light] Skipping feature selection, using fixed hist: {sel_hist}")
        else:
            try:
                _, sel_hist = phase1(
                    panel_df, static_df, TEMPORAL_FUTR, hist_exog, horizon, results_dir,
                    cuda_device=cuda_device)
            except Exception as e:
                log(f"Phase 1 failed for {label}: {e}", 'ERROR')
                sel_hist = hist_exog

        best_input = 48

        # Phase 3: Dropout sweep
        if args.light:
            sel_futr = SELECTED_FUTR
            best_hp = {**FIXED_HP, 'dropout': 0.05}
            log(f"  [light] Skipping dropout sweep, using fixed HP: {best_hp}")
        else:
            try:
                sel_futr, best_hp = phase3(
                    panel_df, static_df, futr_exog, sel_hist, best_input, horizon, results_dir,
                    cuda_device=cuda_device)
            except Exception as e:
                log(f"Phase 3 failed for {label}: {e}", 'ERROR')
                sel_futr = SELECTED_FUTR
                best_hp = {**FIXED_HP, 'dropout': 0.05}

        horizon_configs[label] = {
            'futr': sel_futr, 'hist': sel_hist,
            'input_size': best_input, 'hp': best_hp,
        }

        try:
            r = phase4(
                label, horizon, cache_df, django_df, all_futr, om_hist,
                sel_futr, sel_hist, best_input, best_hp,
                run_dir, results_dir, cuda_device=cuda_device)
            if r:
                model_results[label] = r
        except Exception as e:
            log(f"Phase 4 failed for {label}: {e}", 'ERROR')
            traceback.print_exc()

        # Evaluate on additional datasets
        if eval_dirs:
            log(f"  Evaluating {label} on {len(eval_dirs)} dataset(s)...")
            for eval_dir, ds_label in eval_dirs.items():
                er = evaluate_on_dataset(
                    eval_dir, ds_label, label, horizon, all_futr, om_hist,
                    sel_futr, sel_hist, best_input, best_hp,
                    results_dir, cuda_device=cuda_device)
                if er:
                    eval_results.setdefault(label, {})[ds_label] = er

    total = time.time() - total_t0
    log("=" * 70)
    log(f"COMPLETE in {total/60:.1f} min")
    for label, r in model_results.items():
        s = f"{r['season_relMAE']:.4f}" if not np.isnan(r['season_relMAE']) else 'n/a'
        log(f"  {label}: season={s} | overall={r['relMAE']:.4f}")

    if eval_results:
        log(f"\n  EVALUATION:")
        log(f"  {'Model':<6} {'Dataset':<15} {'Overall':>10} {'Season':>10} {'MAE':>8} {'Series':>7}")
        log(f"  {'-'*60}")
        for label in eval_results:
            for ds_label, er in eval_results[label].items():
                s = f"{er['season_relMAE']:.4f}" if not np.isnan(er['season_relMAE']) else 'n/a'
                log(f"  {label:<6} {ds_label:<15} {er['relMAE']:10.4f} {s:>10} {er['MAE']:8.1f} {er['n_series']:>7}")

    log(f"  Output: {run_dir}/")

    save_checkpoint(results_dir, 'final', {
        'run_dir': run_dir,
        'train_dir': data_dir,
        'eval_dirs': eval_dirs,
        'cuda_device': cuda_device,
        'horizon_configs': {k: {kk: vv for kk, vv in v.items()} for k, v in horizon_configs.items()},
        'models': {k: {kk: vv for kk, vv in v.items()} for k, v in model_results.items()},
        'evaluation': eval_results,
        'total_time_min': total / 60, 'status': 'complete',
    })


def main_deploy_only(args):
    data_dir = _resolve_data_dir(args.data_dir)
    args.data_dir = data_dir
    cuda_device = args.cuda_device if args.cuda_device is not None else detect_cuda_device()

    run_dir = make_run_dir(args.prefix)
    results_dir = f'{run_dir}/results'
    os.makedirs(results_dir, exist_ok=True)
    total_t0 = time.time()

    log("LSTM Training — Deploy Only (skip optimization)")
    log(f"Target models: {TARGET_HORIZONS}")
    log(f"Run dir:  {run_dir}")
    log(f"CUDA device: {cuda_device if cuda_device else 'CPU'}")

    eval_dirs = resolve_eval_dirs(args)
    log(f"Eval dirs: {eval_dirs}")

    cache_df, django_df = load_data(data_dir)
    all_futr, om_hist = define_features(cache_df, django_df)

    selected_futr = SELECTED_FUTR
    selected_hist = ['om_cloud_cover_low', 'om_shortwave_radiation', 'om_vapour_pressure_deficit']
    best_input = 48
    best_hp = {'hidden_size': 64, 'n_head': 4, 'dropout': 0.05, 'lr': 0.001}

    log(f"FUTR: {selected_futr}")
    log(f"HIST: {selected_hist}")
    log(f"Input: {best_input}, HP: {best_hp}")

    model_results = {}
    for label, hz in TARGET_HORIZONS.items():
        try:
            r = phase4(
                label, hz, cache_df, django_df, all_futr, om_hist,
                selected_futr, selected_hist, best_input, best_hp,
                run_dir, results_dir, cuda_device=cuda_device)
            if r:
                model_results[label] = r
        except Exception as e:
            log(f"Model {label} crashed: {e}", 'ERROR')
            traceback.print_exc()

    total = time.time() - total_t0
    log("=" * 70)
    log(f"COMPLETE in {total/60:.1f} min")
    for label, r in model_results.items():
        log(f"  {label}: season={r['season_relMAE']:.4f} | overall={r['relMAE']:.4f}")
    log(f"  Output: {run_dir}/")

    save_checkpoint(results_dir, 'final', {
        'run_dir': run_dir,
        'cuda_device': cuda_device,
        'selected_futr': selected_futr, 'selected_hist': selected_hist,
        'best_input': best_input, 'best_hp': best_hp,
        'models': {k: {kk: vv for kk, vv in v.items()} for k, v in model_results.items()},
        'total_time_min': total / 60, 'status': 'complete',
    })


if __name__ == '__main__':
    args = parse_args()
    if args.worker:
        _worker_main(args.worker)
    elif args.deploy_only:
        main_deploy_only(args)
    else:
        main(args)