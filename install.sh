#!/usr/bin/env bash
# Install the MoE SSD engine on this computer (curl -fsSL <RAW_URL>/install.sh | bash), changing
# nothing outside INSTALL_DIR but its parent directory and using no root.
set -euo pipefail

RAW_URL="${MOE_RAW_URL:-https://raw.githubusercontent.com/IamCooller/larder/main}"
REPO_URL="${MOE_REPO_URL:-https://github.com/IamCooller/larder.git}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/larder}"

PROG=install.sh
DRY=0
HAVE_XCODE=0

say() { printf '%s\n' "$*"; }
die() { printf '%s: %s\n' "$PROG" "$*" >&2; exit 1; }
step() { printf '\n[%s] %s\n' "$1" "$2"; }

usage() {
  cat <<EOF
Usage: install.sh [--dry-run]

  --dry-run   print every step and change nothing
  --help      this text

Environment:
  INSTALL_DIR     where the repository goes (default: $HOME/larder)
  MOE_REPO_URL    git URL to clone (default: $REPO_URL)
  MOE_RAW_URL     raw URL of install.sh (default: $RAW_URL)
EOF
}

run() {
  if [ "$DRY" = 1 ]; then
    printf '  would run: %s\n' "$*"
  else
    "$@"
  fi
}

# The whole install is one function called on the last line, so a download cut short runs nothing.
main() {
  for arg in "$@"; do
    case "$arg" in
      --dry-run) DRY=1 ;;
      -h|--help) usage; exit 0 ;;
      *) die "unknown argument: $arg (try --help)" ;;
    esac
  done

  step "1/7" "Platform"
  UNAME_S=$(uname -s)
  UNAME_M=$(uname -m)
  SUPPORTED="macOS on Apple Silicon (MLX Metal) and Linux x86_64 with an NVIDIA GPU (MLX CUDA)"
  case "$UNAME_S/$UNAME_M" in
    Darwin/arm64)
      PLATFORM=metal
      PLATFORM_NAME="macOS on Apple Silicon (MLX Metal)"
      ;;
    Linux/x86_64|Linux/amd64)
      if [ -e /dev/nvidia0 ] || command -v nvidia-smi >/dev/null 2>&1; then
        PLATFORM=cuda
        PLATFORM_NAME="Linux x86_64 with an NVIDIA GPU (MLX CUDA)"
      else
        die "this Linux machine has no NVIDIA GPU; supported: $SUPPORTED"
      fi
      ;;
    Darwin/*)
      die "this Mac is not Apple Silicon; supported: $SUPPORTED"
      ;;
    *)
      die "$UNAME_S on $UNAME_M is not supported; supported: $SUPPORTED"
      ;;
  esac
  say "  $PLATFORM_NAME"

  step "2/7" "Required tools"
  command -v git >/dev/null 2>&1 || die "git is missing: on macOS run 'xcode-select --install'; on Linux install git without root (https://git-scm.com/download/linux) and run this again"
  PY=""
  for cand in python3.13 python3.12 python3.11 python3.10 python3 \
              /opt/homebrew/bin/python3.13 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
              /usr/local/bin/python3.13 /usr/local/bin/python3.12 /usr/local/bin/python3.11; do
    if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
      PY=$(command -v "$cand")
      break
    fi
  done
  if [ -z "$PY" ]; then
    say "  Python 3.10 or newer was not found."
    if [ "$UNAME_S" = Darwin ]; then
      say "  Run 'xcode-select --install' for Apple's tools, or install Python from https://www.python.org/downloads/macos/ (no sudo)."
      say "  With Homebrew present: 'brew install python@3.12'."
    else
      say "  Without root: pyenv (https://github.com/pyenv/pyenv), or a Python 3.10+ from your distribution."
    fi
    die "Python 3.10 or newer is required"
  fi
  PY_VER=$("$PY" -V 2>&1)
  say "  python: $PY ($PY_VER)"
  say "  git: $(git --version)"

  if [ "$UNAME_S" = Darwin ]; then
    if xcode-select -p >/dev/null 2>&1; then
      HAVE_XCODE=1
      say "  Xcode command line tools: $(xcode-select -p)"
    else
      say "  Xcode command line tools: missing. Run 'xcode-select --install' (no sudo), then run this installer again."
      say "  Without them the native Metal agent is not built and the engine uses its Python read path."
    fi
  fi

  if ! "$PY" -c 'import sys; raise SystemExit(0 if sys.version_info < (3, 14) else 1)' >/dev/null 2>&1; then
    say "  Note: $PY_VER is newer than the tested range (3.10 to 3.13); continuing."
  fi
  "$PY" -c 'import venv' >/dev/null 2>&1 || die "this Python has no venv module: install the venv support (Debian/Ubuntu: the python3-venv package) or use a pyenv Python"

  step "3/7" "Repository"
  PARENT=$(dirname "$INSTALL_DIR")
  [ -d "$PARENT" ] || run mkdir -p "$PARENT"
  if [ -d "$INSTALL_DIR/.git" ]; then
    say "  Updating $INSTALL_DIR"
    run git -C "$INSTALL_DIR" pull --ff-only
  elif [ -e "$INSTALL_DIR" ] && [ "$DRY" = 0 ]; then
    die "$INSTALL_DIR exists and is not a git clone; remove it or set INSTALL_DIR"
  else
    say "  Cloning $REPO_URL"
    run git clone "$REPO_URL" "$INSTALL_DIR"
  fi
  if [ "$DRY" = 0 ] && [ ! -f "$INSTALL_DIR/requirements.txt" ]; then
    die "$INSTALL_DIR has no requirements.txt; remove it and run this installer again"
  fi

  step "4/7" "Virtual environment"
  VENV_PY="$INSTALL_DIR/.venv/bin/python"
  if [ -x "$VENV_PY" ]; then
    say "  Reusing $VENV_PY"
  else
    say "  Creating $INSTALL_DIR/.venv"
    run "$PY" -m venv "$INSTALL_DIR/.venv"
  fi
  if [ "$DRY" = 0 ] && ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    die "pip is missing in the virtual environment: install the venv support for this Python and run again"
  fi

  step "5/7" "Python packages"
  say "  Installing requirements.txt"
  run "$VENV_PY" -m pip install --disable-pip-version-check -r "$INSTALL_DIR/requirements.txt"

  step "6/7" "Native agent"
  if [ "$PLATFORM" = metal ]; then
    if [ "$HAVE_XCODE" = 1 ]; then
      say "  Building the Metal agent with the Xcode command line tools"
      if ! run "$INSTALL_DIR/native/build.sh" "$VENV_PY"; then
        say "  The Metal agent did not build; the engine will use its Python read path (slower)."
        say "  Rerun this installer after fixing the compiler error to get the faster path."
      fi
    else
      say "  Skipped: run 'xcode-select --install' to build the faster Metal agent."
    fi
  else
    CUDA_DIR="${CUDA_HOME:-}"
    if [ -z "$CUDA_DIR" ] && [ -d /usr/local/cuda ]; then CUDA_DIR=/usr/local/cuda; fi
    if [ -z "$CUDA_DIR" ] && [ -d /opt/cuda ]; then CUDA_DIR=/opt/cuda; fi
    if [ -n "$CUDA_DIR" ] && [ -d "$CUDA_DIR" ] && command -v g++ >/dev/null 2>&1; then
      say "  CUDA toolkit: $CUDA_DIR; building the CUDA agent with g++"
      if ! run env PYTHONPATH="$INSTALL_DIR/native" "$VENV_PY" -c 'import moe_pipe_cuda; print("  agent:", moe_pipe_cuda._lib._name)'; then
        say "  The CUDA agent did not build; the engine will use its Python read path (slower)."
      fi
    else
      say "  Skipped: no CUDA toolkit (CUDA_HOME, /usr/local/cuda or /opt/cuda) or no g++."
      say "  The engine will use its Python read path (slower)."
    fi
  fi

  step "7/7" "Done"
  if [ "$DRY" = 1 ]; then
    say "  Dry run: nothing was downloaded, installed or changed."
    say "  A real run would finish with the command below."
  fi
  RUN_DIR=$INSTALL_DIR
  case "$RUN_DIR" in
    "$HOME"/*) RUN_DIR="~/${RUN_DIR#"$HOME"/}" ;;
  esac
  printf '\n  Run it:\n\ncd %s && ./larder\n' "$RUN_DIR"
}

main "$@"
