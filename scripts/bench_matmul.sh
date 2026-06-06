#!/usr/bin/env bash
# Measure device latency of the matmul baseline vs optimized via neuron-bench.
# For each variant: jit-run once under NEURON_RT_INSPECT to emit a NEFF, then neuron-bench it.
# Run ON a Neuron instance inside the Neuron venv.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
cd "$REPO/examples/matmul"

for V in tiled fully_optimized; do
  FN="nki_matmul_${V}_"
  echo "== $FN: jit-run to emit NEFF =="
  rm -rf "prof_$V"; mkdir -p "prof_$V"
  NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=1 \
  NEURON_RT_INSPECT_OUTPUT_DIR="$PWD/prof_$V" NEURON_CC_FLAGS="--target trn1" \
    python "$REPO/scripts/_run_one_kernel.py" "$FN" 2>&1 | grep -E "^ran" || true
  NEFF="$(find "prof_$V" -name '*.neff' | head -1)"
  echo "   NEFF=$NEFF"
  echo "== $FN: neuron-bench =="
  rm -rf "/tmp/bench_$V"
  # DataParallel=2 fails on a single Neuron device (LNC); the DataParallel=1 (i1) run gives device latency.
  neuron-bench exec "$NEFF" -w 20 -n 300 -o "/tmp/bench_$V" >/dev/null 2>&1 || true
done

echo "== summary =="
python "$REPO/scripts/_summarize_bench.py"
