# How-to / walkthrough: prove a matmul speedup on Trainium, end to end

This walks the first end-to-end run of `neuron-kernel-autotuner` on a real AWS Trainium
instance, using a known-good baseline/optimized matmul pair from
[nki-samples](https://github.com/aws-neuron/nki-samples). The point: validate the
**measure → verify → diff** chain on ground truth before trusting LLM-generated variants.

It also doubles as the runbook. Measured numbers are filled in at the end after the run.

## What we're comparing

Two NKI kernels from the matmul tutorial (vendored in `examples/matmul/`):

| | function | optimization | arithmetic intensity | expected |
|---|---|---|--:|---|
| baseline | `nki_matmul_tiled_` | basic tiling | ~102 (memory-bound) | under-uses tensor engine |
| "variant 2" | `nki_matmul_fully_optimized_` | block M/N/K + layout + hoisting | ~683 (compute-bound) | should be markedly faster |

Shapes: 4096×1024 · 1024×2048, bf16. The 222 FLOPs/byte bf16 threshold for NeuronCore-v2 is
why we expect the 683-intensity variant to win — but **we measure, we don't assume.**

## Prerequisites

- A Trainium/Inferentia instance with the **Neuron DLAMI** (trn1.2xlarge is plenty;
  NeuronCore-v2). The agent/tooling runs **on** the instance (AWS co-locates agent + hardware).
- This repo copied to the instance.

## Step 1 — Launch and verify the instance

```bash
# trn1.2xlarge, us-west-2, Neuron DLAMI. neuron-ls should show NeuronCores.
neuron-ls
source ~/opt/aws_neuronx_venv_pytorch_*/bin/activate
neuronx-cc --version
```

## Step 2 — (optional) install AWS's agentic-dev skills

We compose them in the full loop; for this fixed-pair e2e they're not required.

```bash
pip install neuron-agentic-development --extra-index-url https://pip.repos.neuron.amazonaws.com
deploy-neuron-agentic-development-to-claude
```

## Step 3 — Run the e2e

```bash
bash scripts/bootstrap_trn1.sh
# equivalently:
python scripts/run_e2e.py --target trn1 --iters 50 --warmup 10 --input-sets 3
```

What it does per variant: runs the kernel on device, **verifies numerical correctness vs a
torch reference over 3 randomized input sets** (the hardened gate — see why below), measures
median device latency over 50 timed iters after 10 warmups, and writes `results/profile_*.json`.
Then it diffs optimized vs baseline and writes `results/diff.json`.

## Step 4 — (optional) richer profile via neuron-explorer

For % of peak / engine utilization, capture a real profile (this is what AWS's
`neuron-nki-profiling` skill automates):

```bash
export NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=1 NEURON_RT_INSPECT_OUTPUT_DIR=./output
python -c "import examples.matmul.matrix_multiplication_torch"   # runs kernels, emits NEFFs
NEFF=$(python scripts/identify-neffs.py ./output nki_matmul_fully_optimized_ 2>/dev/null || ls ./output/*.neff | head -1)
neuron-explorer capture -n "$NEFF" -s profile.ntff --profile-nth-exec=2 --enable-dge-notifs
neuron-explorer view --output-format summary-json -n "$NEFF" -s profile.ntff > summary.json
```

`summary.json` → `nka.parse.parse_summary` gives `mfu_percent` (% of peak) on the
`profile.json`. The first run is also where we lock the exact `summary-json` field names.

## Why the correctness gate is hard to fool

[AccelOpt](https://arxiv.org/abs/2511.15915) found LLMs game a weak checker by computing
partial results → fake speedup. So `nka/verify.py` requires **multiple randomized input
sets**, **full-output** comparison (`torch.allclose(atol=1e-4, rtol=1e-2)`), and keeps the
reference hidden from any generator. A single cherry-picked passing input is explicitly
rejected. This matters even here: it's the same gate that will guard generated variants in v1.

## Results

<!-- RESULTS:START -->
_Filled in after the run on trn1.2xlarge (us-west-2)._

| variant | latency (µs) | % of peak | verified |
|---|--:|--:|:--:|
| `nki_matmul_tiled_` | _TBD_ | _TBD_ | _TBD_ |
| `nki_matmul_fully_optimized_` | _TBD_ | _TBD_ | _TBD_ |
| **measured speedup** | **_TBD_×** | | |

Raw artifacts: `results/profile_nki_matmul_tiled_.json`,
`results/profile_nki_matmul_fully_optimized_.json`, `results/diff.json`.
<!-- RESULTS:END -->

## What this proves for the project

The measure→verify→diff plumbing works on ground truth, with a real, honest speedup number
and a correctness proof. v1 swaps the fixed "variant 2" for **LLM-generated** variants from
`neuron-nki-writing`, guarded by this same gate, ranked by this same diff. The loop is the
product; this run proves its spine.
