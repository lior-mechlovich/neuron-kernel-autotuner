# Orchestration: composing neuron-agentic-development into the auto-tuning loop

How our tool (`neuron-sdk-profiler`) drives AWS's open-source **neuron-agentic-development**
agents/skills as the building blocks, and what we add on top. The net-new asset is the
**orchestrator loop**; AWS already ships the individual steps.

Source reviewed: `github.com/aws-neuron/neuron-agentic-development` @ main (commit 648923a,
2026-06-03). Install: `pip install neuron-agentic-development --extra-index-url
https://pip.repos.neuron.amazonaws.com` then `deploy-neuron-agentic-development-to-claude`.
**Runs ON the trn/inf2 instance** (agent co-located with hardware — no SSH/copy).

## What AWS ships

**Agents** (conversational, single-shot):
- `neuron-nki-agent` — unified lifecycle (write/debug/profile/optimize) — capable but not a measured multi-variant loop.
- `neuron-nki-writer-agent` — author/modify kernels from PyTorch/NumPy/NL; refactor tiling.
- `neuron-nki-debugger-agent` — autonomous compile-error fixing.
- `neuron-nki-profile-analysis-agent` — capture → ingest → compute bounds → localize bottleneck to source lines → report. **Has a single before/after "After Optimization" compare mode.**
- `neuron-framework-autoport-agent` — model porting (not relevant to kernels).

**Skills:** `neuron-nki-writing`, `neuron-nki-debugging`, `neuron-nki-docs`,
`neuron-nki-profiling` (capture: `NEURON_RT_INSPECT_*` env + `neuron-explorer capture`),
`neuron-nki-profile-querying` (SQL/DuckDB API over parquet; `Summary` table has
`total_time`, `mfu_estimated_percent`, `tensor_engine_active_time_percent`,
`dma_active_time_percent`), `neuron-framework-autoport`, `neuron-framework-equivalence`
(model-level, NOT kernel-level).

## The mapping — our loop step → their block → what we add

| # | Loop step | AWS provides | We add (net-new) |
|---|-----------|--------------|------------------|
| 1 | Measure baseline | `neuron-nki-profiling` (capture) + `neuron-nki-profile-querying` (`Summary` row) | normalize into `profile.json`; record as baseline |
| 2 | Decide what to optimize | `neuron-nki-profile-analysis-agent` (bounds, gaps, source-line localization) | use its finding as the **variant seed** |
| 3 | Generate variant(s) | `neuron-nki-writing` / `neuron-nki-writer-agent` | **beam search**: N variants/iter from top-K survivors |
| 4 | Compile + fix | `neuronx-cc` + `neuron-nki-debugger-agent` | bounded retry policy; structured pass/fail |
| 5 | Correctness gate | (kernel-level: tutorial's `torch.allclose`) | **hardened** multi-input, full-output, hidden-reference verify (anti-reward-hack) |
| 6 | Profile each correct variant | `neuron-nki-profiling` | — |
| 7 | Diff + rank | analysis-agent does *single* side-by-side | **rank across N variants**; roofline **% of peak** (`mfu_estimated_percent`); cost $ |
| 8 | Self-improve across iters | — | **optimization memory** (slow→fast pairs + summaries); early-stop near ~80% peak |
| 9 | Report / share | — | verified diff report + how-to/blog artifact |
| 10 | CI gate (later) | — | headless CLI + GitHub Action over the same engine |

**Honest overlap:** steps 1–7 single-pass are largely achievable by chaining AWS's
agents by hand today. Our value is making it an **automated, measured, ranked, memoized,
verified, repeatable loop** (steps 3+5+7+8+9) and packaging it as one command — plus cost
(7) and CI (10) which AWS has nowhere.

## Orchestrator contract (what our `loop` actually calls)

```
neuron-optimize(kernel, inputs, budget):
  base = profile(kernel)                      # Skill: neuron-nki-profiling -> profile-querying Summary
  seed = analyze(base)                         # Agent: neuron-nki-profile-analysis-agent (bottleneck + source lines)
  memory.recall(kernel_signature)              # OURS: inject relevant slow->fast summaries into the writer prompt
  pool = [kernel]
  while not budget.spent and not early_stop(base, pool):   # OURS: early-stop near ~80% peak (AccelOpt)
    cands = []
    for parent in beam_top_k(pool):            # OURS: beam search (AccelOpt)
      v = write_variant(parent, seed, memory)  # Skill: neuron-nki-writing / writer-agent
      ok = compile_with_retries(v)             # neuronx-cc + neuron-nki-debugger-agent, bounded
      if not ok: memory.record_negative(v); continue
      if not verify(kernel, v, input_sets):    # OURS: hardened correctness gate (AccelOpt warning)
          memory.record_negative(v); continue
      p = profile(v)                           # neuron-nki-profiling -> Summary row -> profile.json
      cands.append((v, p))
    pool = select_best(pool + cands)           # OURS: rank by total_time / mfu_estimated_percent (+ cost)
    memory.curate(cands)                        # OURS: store positive rewrites (speedup > t_pos)
  return report(base, pool.best)               # OURS: verified diff + roofline + $ + how-to
```

Data passed between steps is the `profile.json` schema (see PRD), populated from the
`Summary` table: `total_time` → latency, `mfu_estimated_percent` → % of peak (roofline),
`tensor_engine_active_time_percent` / `dma_active_time_percent` → engine utilization /
compute-vs-memory-bound.

## Prior art: AccelOpt (arXiv 2511.15915) — what we borrow

"AccelOpt: A Self-Improving LLM Agentic System for AI Accelerator Kernel Optimization"
is the closest research system to this idea: planner → executor (generate+verify) →
profiler → beam-search selection, on NKI/Trainium. Results: Trn1 49%→61% of peak,
Trn2 45%→59%; matches Claude Sonnet 4 quality at **26× cheaper** with open models.

Borrow:
- **Beam search** over variants beats independent sampling (step 3/7).
- **Optimization memory**: store slow→fast AND failed rewrites + LLM summaries; filter by
  speedup thresholds (their `t_pos=1.04`, `t_neg=1.15`); grow capacity over update
  frequency. Saved 16–17% cost (step 8).
- **Roofline / % of peak** as the headline metric, not just relative speedup — maps to
  `mfu_estimated_percent` (step 7); early-stop near ~80% peak (step 8).
- **Hardened correctness** — their key failure mode: *LLMs game a weak checker via partial
  results (fake speedup)*. Use multiple randomized inputs, full-output compare, hidden
  reference (step 5). This is why the gate is load-bearing, not a formality.
- **Profiling rigor**: warmup + multiple rounds + variance threshold (step 6).
- **Executor model quality matters more than planner**; randomize profiling-item order for
  diversity.

How we differ from AccelOpt: it's a research search-system (single-core, heavy sampling).
We are a **developer-facing Claude Code skill** that **composes AWS's official agentic-dev
skills**, runs on-instance, one command, with a verified shareable diff + cost + (later)
CI. We borrow the science; we ship the tool. (We will credit AccelOpt in the blog/how-to.)

## First e2e (no LLM needed yet) — validate the plumbing on a known pair

Use `nki-samples` matmul (`src/nki_samples/tutorials/matrix_multiplication/`):
- baseline = `nki_matmul_tiled_` (arithmetic intensity 102, memory-bound)
- "variant 2" = `nki_matmul_fully_optimized_` (arithmetic intensity 683, compute-bound)
- inputs: lhsT/rhs derived from 4096×1024 · 1024×2048 bf16; correctness via the tutorial's
  `torch.allclose(atol=1e-4, rtol=1e-2)`.

Run steps 1+5+6+7 on this fixed pair (skip generate/beam) to prove capture→verify→profile
→diff and confirm the predicted speedup with real `total_time` / `mfu_estimated_percent`.
That run becomes the first how-to/blog post.
