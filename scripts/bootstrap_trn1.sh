#!/usr/bin/env bash
# Bootstrap a Neuron DLAMI instance (trn1/inf2) and run the matmul e2e.
# Assumes the repo has been copied to the instance (e.g. via scp) at ~/neuron-kernel-autotuner.
# Run ON the instance.
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/neuron-kernel-autotuner}"
TARGET="${TARGET:-trn1}"

echo "== activating Neuron venv =="
# DLAMI ships a pre-built venv; pick the first torch-neuronx one found.
VENV="$(ls -d "$HOME"/opt/aws_neuronx_venv_pytorch_* 2>/dev/null | head -1 || true)"
if [ -z "$VENV" ]; then
  VENV="$(ls -d /opt/aws_neuronx_venv_pytorch_* 2>/dev/null | head -1 || true)"
fi
[ -n "$VENV" ] || { echo "No Neuron venv found; is this a Neuron DLAMI?"; exit 1; }
# shellcheck disable=SC1091
source "$VENV/bin/activate"
echo "venv: $VENV"

echo "== versions =="
neuron-ls || true
python -c "import torch, torch_neuronx; print('torch', torch.__version__)" || true
neuronx-cc --version 2>/dev/null || true
export NEURON_SDK_VERSION="$(neuronx-cc --version 2>&1 | head -1 || echo unknown)"

echo "== deps =="
pip -q install numpy >/dev/null 2>&1 || true
# Optional: AWS's agentic-dev (we compose it; not required for this fixed-pair e2e)
pip -q install neuron-agentic-development \
  --extra-index-url https://pip.repos.neuron.amazonaws.com >/dev/null 2>&1 || \
  echo "(neuron-agentic-development install skipped/failed -- not required for fixed-pair e2e)"

cd "$REPO_DIR"
echo "== 1/3 correctness (verify both variants vs torch over 3 input sets) =="
python scripts/run_e2e.py --target "$TARGET" --iters 20 --warmup 5 --input-sets 3 || true

echo "== 2/3 device-latency speedup (neuron-bench) =="
bash scripts/bench_matmul.sh

echo "== 3/3 build proof artifacts (parse summary-json -> profile.json + diff) =="
# capture per-variant summary-json for the metrics used in the profiles
cd "$REPO_DIR/examples/matmul"
for Vv in tiled fully_optimized; do
  NEFF=$(find prof_$Vv -name "*.neff" 2>/dev/null | head -1)
  NTFF=$(find prof_$Vv -name "*.ntff" 2>/dev/null | head -1)
  [ -n "$NEFF" ] && neuron-explorer view --output-format summary-json -n "$NEFF" -s "$NTFF" > /tmp/sum_$Vv.json 2>/dev/null || true
done
cd "$REPO_DIR"
python scripts/_build_profiles.py || true

echo "== artifacts =="
ls -la results/
