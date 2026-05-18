#!/usr/bin/env bash
# Install LightFM 1.17 around an upstream packaging bug that breaks PEP-517
# isolated builds. See https://github.com/lyst/lightfm/issues/709.
#
# The sdist's setup.py does `__builtins__.__LIGHTFM_SETUP__ = True` which
# fails because PEP-517 hands setup.py a `__builtins__` *dict*, not the
# module. Replacing that line with `import builtins; builtins.__LIGHTFM_SETUP__ = True`
# lets the build succeed unchanged otherwise.
set -euo pipefail

PYTHON="${PYTHON:-python}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

cd "$WORK"
curl -sLf -o lightfm-1.17.tar.gz \
    https://files.pythonhosted.org/packages/source/l/lightfm/lightfm-1.17.tar.gz
tar xzf lightfm-1.17.tar.gz
sed -i.bak \
    's|__builtins__.__LIGHTFM_SETUP__ = True|import builtins; builtins.__LIGHTFM_SETUP__ = True|' \
    lightfm-1.17/setup.py
"$PYTHON" -m pip install "setuptools<70" wheel cython numpy scipy scikit-learn requests
"$PYTHON" -m pip install --no-build-isolation ./lightfm-1.17
