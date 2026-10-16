#!/bin/sh
# Creates ./.venv with the pinned requirements. Run by first_run.sh on install; safe to re-run.
#
# Uses a pinned, checksum-verified uv binary and a uv-managed Python 3.12, so the host needs neither python3-venv
# (not installed on stock Ubuntu) nor a particular system Python.
set -eu
cd "$(dirname "$0")"

UV_VERSION="0.12.21"
UV_SHA256="23f02075b652bb1df64178cfae41b5caf160822e720e2663568f3f5d63bc52c0"
PYTHON_VERSION="3.12"

if [ -f .venv/.installed ] && cmp -s requirements.txt .venv/.installed; then
    echo "venv up to date"
    exit 0
fi

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) ;;
    *) echo "unsupported platform $(uname -s)-$(uname -m); this module is linux/amd64 only" >&2; exit 1 ;;
esac

if [ ! -x bin/uv ]; then
    echo "downloading uv $UV_VERSION"
    mkdir -p bin
    tarball="bin/uv.tar.gz"
    curl -fsSL -o "$tarball" "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
    echo "$UV_SHA256  $tarball" | sha256sum -c -
    tar -xzf "$tarball" -C bin --strip-components=1
    rm -f "$tarball"
fi

export UV_PYTHON_INSTALL_DIR="$PWD/bin/python"
export UV_CACHE_DIR="${VIAM_MODULE_DATA:-$PWD}/uv-cache"
bin/uv venv --python "$PYTHON_VERSION" --python-preference only-managed --allow-existing .venv
bin/uv pip install --python .venv/bin/python -r requirements.txt
cp requirements.txt .venv/.installed
echo "venv ready"
