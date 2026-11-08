#!/bin/bash
# Build cpu_kern for one python env into cpu_ext/build (cpu_kernels.py adds that dir to sys.path).
#   cpu_ext/build.sh <path-to-python>
# Needs nanobind==3.0.1 (nanobind internals v22, what mlx 0.32.4 is built with) and cmake in that env.
set -euo pipefail
: "${1:?usage: cpu_ext/build.sh <path-to-python>}"
PY=$("$1" -c 'import sys; print(sys.executable)')
HERE=$(cd "$(dirname "$0")" && pwd)
NB=$("$PY" -c 'import nanobind; print(nanobind.__version__)' 2>/dev/null || true)
[ "$NB" = 3.0.1 ] || "$PY" -m pip install nanobind==3.0.1
find_cmake() { "$PY" -c 'import cmake, os; print(os.path.join(cmake.CMAKE_BIN_DIR, "cmake"))' 2>/dev/null || true; }
CMAKE=$(find_cmake)
[ -x "$CMAKE" ] || { "$PY" -m pip install cmake==4.4.4; CMAKE=$(find_cmake); }
"$CMAKE" -S "$HERE" -B "$HERE/build" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY" > /dev/null
"$CMAKE" --build "$HERE/build" -j 2
"$PY" -I -c "import sys; sys.path.insert(0, '$HERE/build'); import mlx.core as mx, cpu_kern; print('cpu_kern built:', cpu_kern.__file__)"
