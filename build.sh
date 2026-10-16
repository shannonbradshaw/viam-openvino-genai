#!/bin/sh
# Packages the module source for the registry. Dependencies are installed on the machine by first_run.sh, so the
# tarball is small and contains no native libraries (in particular no bundled libstdc++ that could shadow the
# newer one Intel's GPU/NPU drivers need).
set -e
cd "$(dirname "$0")"
tar -czf module.tar.gz --exclude=__pycache__ meta.json README.md requirements.txt setup.sh first_run.sh run.sh src
echo "built module.tar.gz"
