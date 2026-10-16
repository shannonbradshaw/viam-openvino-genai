#!/bin/sh
# Module entrypoint (meta.json "entrypoint"). viam-server passes the module socket path as the argument.
cd "$(dirname "$0")"
if [ ! -x .venv/bin/python ] || ! cmp -s requirements.txt .venv/.installed; then
    # first_run.sh normally does this; fall back here for local modules and hot reloads.
    ./setup.sh >&2
fi
exec .venv/bin/python src/main.py "$@"
