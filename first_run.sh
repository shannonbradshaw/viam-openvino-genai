#!/bin/sh
# Runs once per module version before viam-server starts the module (meta.json "first_run"). Building the venv
# here keeps the download out of module startup, which viam-server times out.
# uv reports normal progress on stderr, which viam-server logs as warnings; send it to stdout. A failure still
# shows up through the exit status.
cd "$(dirname "$0")" && exec ./setup.sh 2>&1
