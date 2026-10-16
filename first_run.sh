#!/bin/sh
# Runs once per module version before viam-server starts the module (meta.json "first_run"). Building the venv
# here keeps the download out of module startup, which viam-server times out.
cd "$(dirname "$0")" && exec ./setup.sh
