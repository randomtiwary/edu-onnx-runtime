#!/usr/bin/env bash
# LEARNER: Before debugging "why won't CUDA build?", capture the machine state.
# Run from repo root:
#   ./scripts/env_report.sh | tee docs/env-report-local.txt
#
# This script never fails the build; missing tools just print "not found".

set -u

section() {
  printf '\n======== %s ========\n' "$1"
}

section "Host"
uname -a 2>/dev/null || true
date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || true

section "CPU"
if command -v lscpu >/dev/null 2>&1; then
  lscpu | sed -n '1,25p'
else
  echo "lscpu not found"
fi

section "Compilers"
for c in g++ clang++ cmake ninja nvcc protoc python3; do
  if command -v "$c" >/dev/null 2>&1; then
    printf '%s: ' "$c"
    case "$c" in
      cmake) cmake --version | head -1 ;;
      ninja) ninja --version ;;
      nvcc) nvcc --version | tail -1 ;;
      protoc) protoc --version ;;
      python3) python3 --version ;;
      *) "$c" --version 2>&1 | head -1 ;;
    esac
  else
    echo "$c: not found"
  fi
done

section "NVIDIA / CUDA devices"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi 2>&1 || echo "(nvidia-smi exited non-zero — treat as flaky GPU probe)"
else
  echo "nvidia-smi: not found"
fi
ls -la /dev/nvidia* 2>&1 || echo "no /dev/nvidia* nodes"

section "PCI display adapters"
if command -v lspci >/dev/null 2>&1; then
  lspci | grep -iE 'vga|3d|display' || true
else
  echo "lspci not found"
fi

section "eduort configure hint"
cat <<'HINT'
# CPU-only (default, always works for learning):
cmake -G Ninja -B build -DEDUORT_ENABLE_CUDA=OFF
cmake --build build
./build/eduort_version_smoke

# CUDA (after PR10; may fail on flaky drivers — that is OK):
# cmake -G Ninja -B build -DEDUORT_ENABLE_CUDA=ON
HINT

echo
echo "env_report complete."
