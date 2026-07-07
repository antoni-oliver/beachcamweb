#!/usr/bin/env bash
#
# start.sh — run the BeachCamWeb Django project locally (macOS / Linux).
#
# First run: creates a .venv, installs requirements.txt, applies migrations,
# creates a dev superuser, then starts the dev server.
# Later runs: just migrates and starts the server.
#
# Usage:
#   ./start.sh                 # start on 127.0.0.1:8000
#   ./start.sh -i              # force reinstall dependencies
#   HOST=0.0.0.0 PORT=9000 ./start.sh
#   SKIP_SUPERUSER=1 ./start.sh
#   PYTHON=python3.12 ./start.sh
#
# Env overrides: HOST, PORT, PYTHON, SKIP_SUPERUSER,
#                DJANGO_SUPERUSER_USERNAME / _PASSWORD / _EMAIL
#
set -euo pipefail

# --- locate the project (this script lives next to manage.py) ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$SCRIPT_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"
VENV_PY="$VENV/bin/python"

log()  { printf '\033[1;34m[start]\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m[start] error:\033[0m %s\n' "$*" >&2; exit 1; }

# --- flags -------------------------------------------------------------------
INSTALL=0
for arg in "$@"; do
  case "$arg" in
    -i|--install) INSTALL=1 ;;
    -h|--help)    sed -n '2,18p' "$0"; exit 0 ;;
    *)            fail "unknown option: $arg (try --help)" ;;
  esac
done

[ -f manage.py ] || fail "manage.py not found in $SCRIPT_DIR"
command -v "$PYTHON" >/dev/null 2>&1 || fail "$PYTHON not found — install Python 3.10+"

# --- virtualenv --------------------------------------------------------------
if [ ! -x "$VENV_PY" ]; then
  log "Creating virtualenv ($VENV) with $PYTHON ..."
  "$PYTHON" -m venv "$VENV"
  INSTALL=1
fi

if [ "$INSTALL" = "1" ]; then
  log "Installing dependencies (requirements.txt) — this can take a while on first run ..."
  "$VENV_PY" -m pip install --upgrade pip >/dev/null
  "$VENV_PY" -m pip install -r requirements.txt
fi

# --- database ----------------------------------------------------------------
log "Applying database migrations ..."
"$VENV_PY" manage.py migrate --noinput

# --- dev superuser (idempotent; dev defaults, override via env) --------------
if [ "${SKIP_SUPERUSER:-0}" != "1" ]; then
  export DJANGO_SUPERUSER_USERNAME="${DJANGO_SUPERUSER_USERNAME:-admin}"
  export DJANGO_SUPERUSER_EMAIL="${DJANGO_SUPERUSER_EMAIL:-admin@example.com}"
  export DJANGO_SUPERUSER_PASSWORD="${DJANGO_SUPERUSER_PASSWORD:-admin12345}"
  "$VENV_PY" manage.py shell <<'PYEOF'
import os
from django.contrib.auth import get_user_model
U = get_user_model()
name = os.environ["DJANGO_SUPERUSER_USERNAME"]
if U.objects.filter(username=name).exists():
    print(f"[start] superuser '{name}' already exists — skipping")
else:
    U.objects.create_superuser(name, os.environ["DJANGO_SUPERUSER_EMAIL"],
                               os.environ["DJANGO_SUPERUSER_PASSWORD"])
    print(f"[start] created dev superuser '{name}' (password from DJANGO_SUPERUSER_PASSWORD)")
PYEOF
fi

# --- run ---------------------------------------------------------------------
log "Starting dev server at http://$HOST:$PORT  (Ctrl-C to stop)"
exec "$VENV_PY" manage.py runserver "$HOST:$PORT"
