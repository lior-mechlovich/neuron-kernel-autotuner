---
name: neuron-kernel-autotune
description: |
  Autonomously optimize an AWS Neuron (NKI) kernel: generate variants, compile them, verify
  numerical correctness, profile each on hardware, then rank and keep the fastest CORRECT one.
  Composes AWS's neuron-agentic-development skills and adds the closed loop they lack (ranking,
  hardened correctness gate, optimization memory, cost). Runs ON a trn/inf2 instance.
  Use when the user says "optimize this kernel", "make this NKI kernel faster",
  "auto-tune", "find a faster variant", or "tune and prove the speedup".
argument-hint: "[kernel file] [--max-variants N]"
---

# Neuron Kernel Auto-Tune

You are the orchestrator of a closed kernel-optimization loop on AWS Neuron hardware. AWS's
`neuron-agentic-development` ships the individual steps (write / profile / query / debug a
kernel). Your job is the loop they do not ship: a measured, ranked, verified, repeatable
auto-tuner. **Never** report a faster kernel that is not proven correct.

## Prerequisites

- Running ON a `trn1`/`trn2`/`inf2` instance (agent is co-located with hardware; no SSH/copy).
- Neuron venv active (`source <neuron_venv>/bin/activate`); `neuronx-cc` + `neuron-explorer`.
- AWS `neuron-agentic-development` installed (`pip install neuron-agentic-development
  --extra-index-url https://pip.repos.neuron.amazonaws.com`) so its skills are available.

## The loop

For each iteration, keep the gate order strict: **compile -> correctness -> profile -> rank.**

1. **Baseline.** Profile the input kernel with `/neuron-nki-profiling`, query the `Summary`
   row with `/neuron-nki-profile-querying` (`total_time`, `mfu_estimated_percent`,
   `tensor_engine_active_time_percent`, `dma_active_time_percent`). Normalize with
   `nka.parse.parse_summary` -> `profile.json`. Record as the baseline.
2. **Seed.** Use `/neuron-nki-profile-analysis-agent` to localize the bottleneck to source
   lines. That finding is the seed for variant generation.
3. **Generate.** Use `/neuron-nki-writing` (or the writer agent) to produce a variant from the
   seed. Inject relevant slow->fast summaries from the optimization memory into the prompt.
   Use beam search: branch from the top-K survivors, not from scratch each time.
4. **Compile gate.** `neuronx-cc` compile. On failure, hand the error to
   `/neuron-nki-debugging` and retry (bounded, e.g. 3). Still failing -> record as a negative
   in memory and drop the variant.
5. **Correctness gate (load-bearing).** Run `nka.verify.verify_outputs` against the reference
   over MULTIPLE randomized input sets, full-output, reference hidden from the generator.
   A weak checker gets gamed (see AccelOpt) -- do not relax this. Fail -> drop + record negative.
6. **Profile.** Profile the correct variant exactly as in step 1 -> `profile.json`.
7. **Rank.** `nka.diff.rank(baseline, candidates)` -> speedup + % of peak. `nka.cost.cost_delta`
   for the dollar view. Keep the top-K correct variants for the next iteration.
8. **Stop.** Stop on budget (`--max-variants`), no-improvement for K rounds, or near ~80% of
   peak (saturation). Curate the optimization memory (store positive rewrites above threshold).
9. **Report.** Present baseline vs best: speedup, % of peak, $/1M-units, the diff, and the
   correctness proof. Offer to write a how-to from the run.

## Fast path: validate plumbing first

Before trusting generated variants, prove the measure->verify->diff chain on a known-good pair:
`python scripts/run_e2e.py` runs the vendored matmul `nki_matmul_tiled_` (baseline) vs
`nki_matmul_fully_optimized_` ("variant 2") and reports the measured, verified speedup.

## Helper modules (pure, hardware-free, tested)

`nka.parse` (summary -> profile.json), `nka.verify` (hardened correctness),
`nka.diff` (rank + % of peak), `nka.cost` ($/1M-units). See `tests/` for behavior.

## Rules

- Correctness before speed, always. Never surface an unverified "speedup".
- Compose AWS's skills; do not reimplement capture/generation.
- Report % of peak, not just relative speedup. Be honest when throughput (cost) is unknown.
