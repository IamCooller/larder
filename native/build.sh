#!/bin/bash
# Build the native agent (moe_native) for one python env and install it into
# that env's site-packages, where stream_fast finds it first (native/README.md).
#   native/build.sh <path-to-python>
# Installs nanobind==2.15.0 and cmake==4.4.4 into the env when missing; refuses any
# mlx other than 0.32.2 (the extension uses MLX internals of that version).
set -euo pipefail
: "${1:?usage: native/build.sh <path-to-python>}"
# Absolute, venv path kept (cmake takes a relative one as not found).
PY=$("$1" -c 'import sys; print(sys.executable)')
HERE=$(cd "$(dirname "$0")" && pwd)

MLX=$("$PY" -c 'import mlx.core as mx; print(mx.__version__)' 2>/dev/null || echo none)
if [ "$MLX" != 0.32.2 ]; then
  echo "build.sh: $PY has mlx $MLX; moe_native is pinned to mlx 0.32.2" >&2
  exit 1
fi
# Another nanobind builds too, but does not share the mx.array type with mlx.
NB=$("$PY" -c 'import nanobind; print(nanobind.__version__)' 2>/dev/null || true)
[ "$NB" = 2.15.0 ] || "$PY" -m pip install nanobind==2.15.0
find_cmake() {
  "$PY" -c 'import cmake, os; print(os.path.join(cmake.CMAKE_BIN_DIR, "cmake"))' 2>/dev/null || true
}
CMAKE=$(find_cmake)
if [ ! -x "$CMAKE" ]; then
  "$PY" -m pip install cmake==4.4.4
  CMAKE=$(find_cmake)
fi

BUILD=$(mktemp -d)
trap 'rm -rf "$BUILD"' EXIT
"$CMAKE" -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DPython_EXECUTABLE="$PY"
"$CMAKE" --build "$BUILD" -j "$(sysctl -n hw.ncpu)"
DEST=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["platlib"])')
rm -f "$DEST"/moe_native*.so
cp "$BUILD"/moe_native*.so "$DEST"/
# Import it from the env (not this tree) and build one graph op: a nanobind
# mismatch shows here as a TypeError.
"$PY" -I -c '
import mlx.core as mx, moe_native as nv
nv.wait_for(mx.zeros(4, mx.int32), mx.zeros(1), nv.SharedEvent(), 1)
print("moe_native installed:", nv.__file__)'

# The Neural Engine extension behind MOE_ANE_OVERLAP (ane_mlx.mm), into this env's site-packages; without it the overlap stays off.
build_ane() {
  local mlx nb inc ext
  mlx=$("$PY" -c 'import mlx.core, os; print(os.path.dirname(mlx.core.__file__))') || return 1
  nb=$("$PY" -c 'import nanobind, os; print(os.path.dirname(nanobind.__file__))') || return 1
  inc=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["include"])') || return 1
  ext=$("$PY" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))') || return 1
  xcrun clang++ -std=c++20 -O2 -fobjc-arc -fvisibility=hidden -shared -undefined dynamic_lookup \
    -DNB_DOMAIN=mlx -I"$inc" -I"$nb/include" -I"$nb/src" -I"$nb/ext/robin_map/include" \
    -I"$mlx/include" -I"$mlx/include/metal_cpp" "$HERE/../ane_mlx.mm" "$nb/src/nb_combined.cpp" \
    -L"$mlx/lib" -lmlx -Wl,-rpath,"$mlx/lib" -framework Foundation -framework IOSurface -framework Metal \
    -o "$DEST/_ane_mlx$ext" && echo "_ane_mlx installed: $DEST/_ane_mlx$ext"
}
build_ane || echo "build.sh: _ane_mlx not built, the ANE prefill overlap stays off" >&2
