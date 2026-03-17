"""
Full training pipeline: crowd count → export → build dataset → clean → train → evaluate.

Usage:
    python manage.py run_pipeline              # run all steps
    python manage.py run_pipeline --step crowd_count
    python manage.py run_pipeline --step export
    python manage.py run_pipeline --step build_dataset
    python manage.py run_pipeline --step train_data_notebook
    python manage.py run_pipeline --step train
    python manage.py run_pipeline --step evaluate
    python manage.py run_pipeline --from-step build_dataset   # skip earlier steps
"""
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

from django.core.management import call_command
from django.core.management.base import BaseCommand

from apps.prediction import pipeline_config as cfg

STEPS = ['crowd_count', 'export', 'build_dataset', 'train_data_notebook', 'train', 'evaluate']


def _log(msg, step=None):
    prefix = f'[{step}] ' if step else ''
    print(f'{datetime.now().strftime("%H:%M:%S")} {prefix}{msg}', flush=True)


def _run_subprocess(cmd, env=None, cwd=None):
    merged_env = {
        **os.environ,
        **(env or {}),
        'PYTHONPATH': str(cfg.SCRIPTS_DIR),
    }
    result = subprocess.run(cmd, env=merged_env, cwd=cwd or cfg.PREDICTION_DIR)
    if result.returncode != 0:
        raise RuntimeError(f'Subprocess failed: {" ".join(str(c) for c in cmd)}')


def step_crowd_count(options):
    _log('Running crowd counting on media_2022', 'crowd_count')
    args = []
    if options.get('force_reprocess'):
        args.append('--force-reprocess')
    call_command('run_crowd_counting', *args)


def step_export(options):
    _log('Exporting snapshots and beach profiles from Django', 'export')
    cfg.EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    since = options.get('since')
    call_command(
        'export_snapshots',
        output=str(cfg.DJANGO_EXPORT_FILE),
        **({'since': since} if since else {})
    )
    call_command('export_beach_profiles', output=str(cfg.BEACH_PROFILES_FILE))
    _log(f'Exports saved to {cfg.EXPORTS_DIR}', 'export')


def step_build_dataset(options):
    _log('Building dataset (cache + django + weather)', 'build_dataset')

    if not cfg.BUILD_DATASET_SCRIPT.exists():
        raise FileNotFoundError(f'build_dataset script not found: {cfg.BUILD_DATASET_SCRIPT}')

    cmd = [
        sys.executable, str(cfg.BUILD_DATASET_SCRIPT),
        '--django', str(cfg.DJANGO_EXPORT_FILE),
        '--cache', str(cfg.CROWD_CACHE_DIR),
        '--beach-profiles', str(cfg.BEACH_PROFILES_FILE),
        '--output', str(cfg.DATASET_CSV),
    ]
    _run_subprocess(cmd)
    _log(f'Dataset saved to {cfg.DATASET_CSV}', 'build_dataset')


def step_train_data_notebook(options):
    _log('Running generate_train_data notebook via papermill', 'train_data_notebook')

    try:
        import papermill as pm
        import nbconvert
    except ImportError:
        raise ImportError('papermill and nbconvert are required: pip install papermill nbconvert')

    if not cfg.NOTEBOOK_INPUT.exists():
        raise FileNotFoundError(f'Notebook not found: {cfg.NOTEBOOK_INPUT}')

    timestamp = datetime.now().strftime('%d%m%y')
    out_dir_name = f'clean_datasets_{timestamp}'
    clean_dir = cfg.PIPELINE_DIR / out_dir_name

    pm.execute_notebook(
        str(cfg.NOTEBOOK_INPUT),
        str(cfg.NOTEBOOK_OUTPUT),
        parameters={
            'CSV_PATH': cfg.DATASET_CSV.name,
            'OUT_DIR': out_dir_name,
        },
        cwd=str(cfg.PIPELINE_DIR),
    )

    # Export to HTML
    from nbconvert import HTMLExporter
    import nbformat

    with open(cfg.NOTEBOOK_OUTPUT) as f:
        nb = nbformat.read(f, as_version=4)

    exporter = HTMLExporter()
    body, _ = exporter.from_notebook_node(nb)
    with open(cfg.NOTEBOOK_HTML, 'w') as f:
        f.write(body)

    # Symlink clean_datasets → latest output
    symlink = cfg.CLEAN_DATASETS_DIR
    if symlink.is_symlink():
        symlink.unlink()
    elif symlink.exists():
        shutil.rmtree(symlink)
    symlink.symlink_to(clean_dir)

    _log(f'Clean datasets at {clean_dir}', 'train_data_notebook')
    _log(f'HTML report at {cfg.NOTEBOOK_HTML}', 'train_data_notebook')


def step_train(options):
    _log('Training TFT models', 'train')

    if not cfg.TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f'Train script not found: {cfg.TRAIN_SCRIPT}')

    prefix = 'tft'

    cmd = [
        sys.executable, str(cfg.TRAIN_SCRIPT),
        '--data-dir', str(cfg.CLEAN_DATASETS_DIR),
        '--prefix', prefix,
    ]

    if options.get('deploy_only'):
        cmd.append('--deploy-only')

    if options.get('light'):
        cmd.append('--light')

    _run_subprocess(cmd, cwd=cfg.PIPELINE_DIR)

    # Training script creates {prefix}_{timestamp}/tft_model_{horizon}/ inside pipeline_workspace
    # Find the actual run dir (training appends its own timestamp to the prefix)
    run_dirs = sorted(cfg.PIPELINE_DIR.glob(f'{prefix}*'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise RuntimeError(f'No run directory found matching {prefix}* in {cfg.PIPELINE_DIR}')
    run_dir = run_dirs[0]
    _log(f'Run dir: {run_dir.name}', 'train')

    # Deploy as named set under TFT_MODELS_DIR so tft_service auto-discovers it
    set_dir = cfg.TFT_MODELS_DIR / run_dir.name
    deployed = []
    for horizon in ['3d', '10d', '15d']:
        src = run_dir / f'tft_model_{horizon}'
        if not src.exists():
            _log(f'Model dir not found, skipping: tft_model_{horizon}', 'train')
            continue
        dst = set_dir / f'tft_model_{horizon}'
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        deployed.append(horizon)
        _log(f'Deployed {horizon} → {dst}', 'train')

    if deployed:
        _log(f'Model set "{run_dir.name}" deployed with horizons: {deployed}', 'train')
    else:
        raise RuntimeError('No model horizons were deployed — check training logs')


def step_evaluate(options):
    _log('Evaluating model sets', 'evaluate')
    call_command('evaluate_model_sets')


STEP_FN_MAP = {
    'crowd_count':          step_crowd_count,
    'export':               step_export,
    'build_dataset':        step_build_dataset,
    'train_data_notebook':  step_train_data_notebook,
    'train':                step_train,
    'evaluate':             step_evaluate,
}


class Command(BaseCommand):
    help = 'Run the full TFT training pipeline end-to-end'

    def add_arguments(self, parser):
        parser.add_argument('--step', choices=STEPS, default=None,
                            help='Run a single step only')
        parser.add_argument('--from-step', choices=STEPS, default=None,
                            help='Start from this step (skip earlier ones)')
        parser.add_argument('--since', type=str, default=None,
                            help='ISO date filter for export_snapshots (e.g. 2025-01-01)')
        parser.add_argument('--force-reprocess', action='store_true',
                            help='Force re-run crowd counting even if cached')
        parser.add_argument('--deploy-only', action='store_true',
                            help='Pass --deploy-only to training script')
        parser.add_argument('--light', action='store_true',
                            help='Light mode: 10%% data, skip HP sweeps — for testing pipeline flow')
        parser.add_argument('--apply-crowd-count', action='store_true',
                            help='Apply crowd counting step (use existing cache)')

    def handle(self, *args, **options):
        cfg.ensure_dirs()

        if options['step']:
            steps_to_run = [options['step']]
        elif options['from_step']:
            start = STEPS.index(options['from_step'])
            steps_to_run = STEPS[start:]
        else:
            steps_to_run = STEPS

        if not options['apply_crowd_count'] and 'crowd_count' in steps_to_run:
            steps_to_run = [s for s in steps_to_run if s != 'crowd_count']

        self.stdout.write(f'Pipeline steps: {" → ".join(steps_to_run)}')

        for step in steps_to_run:
            _log(f'Starting step: {step}')
            try:
                STEP_FN_MAP[step](options)
                _log(f'Completed: {step}')
            except Exception as e:
                self.stderr.write(self.style.ERROR(f'[{step}] FAILED: {e}'))
                raise

        self.stdout.write(self.style.SUCCESS('Pipeline completed successfully'))