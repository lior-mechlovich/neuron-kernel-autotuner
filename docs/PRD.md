# PRD: Neuron Auto-Tuner (`/neuron-optimize`)

Status: READY-FOR-AGENT (v0)
Source design: `docs/DESIGN.md` (rev 2, 2026-06-06)
Repo: neuron-sdk-profiler (open-source Claude Code skill)

## Problem Statement

I write and tune kernels for AWS Neuron hardware (Trainium / Inferentia). Today, when a
profiler tells me "this op is your bottleneck, try X," I still have to do everything else
by hand: write the variant, recompile it, run it, manually check it still produces correct
results, capture a new profile, and eyeball whether it actually got faster. Every tool I
have — including AWS's Claude-powered Neuron Explorer and the open-source Neuron agentic
skills — stops at the *suggestion*. Nobody closes the loop. So optimization is slow,
error-prone (it's easy to ship a faster kernel that's subtly wrong), and the capture step
alone is ~7 manual steps and ~9 environment variables.

## Solution

A Claude Code skill, `/neuron-optimize <kernel>`, that closes the loop autonomously:
Claude generates optimization variants of my kernel, then the tool **auto-compiles each
one, verifies it is numerically correct against the original, profiles it on real Neuron
hardware, diffs the results, and returns the fastest variant that is still correct** —
with a proof it didn't break anything and a diff explaining why it won. AWS *suggests*;
this *applies, verifies, runs, and proves*. The correctness gate runs before the speed
gate, always — a fast-but-wrong kernel is never reported as a winner.

## User Stories

1. As a Neuron kernel author, I want to run one command against a kernel and get back a faster, verified-correct variant, so that I stop hand-iterating compile/run/check cycles.
2. As a kernel author, I want the tool to generate variants from a profiler's bottleneck suggestion, so that the optimization is informed by real hardware data, not guesswork.
3. As a kernel author, I want every generated variant compiled automatically, so that I never manually invoke `neuronx-cc` per attempt.
4. As a kernel author, I want a variant that fails to compile to be retried with the compiler error fed back to the model, so that transient codegen mistakes self-heal within a bounded number of attempts.
5. As a kernel author, I want every variant checked for numerical equivalence to the original within a tolerance, so that I never ship a kernel that is faster but wrong.
6. As a kernel author, I want an intentionally-wrong variant to be rejected by the tool, so that I can trust the correctness gate is real.
7. As a kernel author, I want each correct variant profiled on real Neuron hardware, so that the speed comparison reflects actual silicon, not estimates.
8. As a kernel author, I want a diff between the winning variant and my baseline (engine utilization, compute-vs-memory bound, top ops), so that I understand *why* it's faster.
9. As a kernel author, I want to point the tool at an existing instance or have it launch one, so that I control hardware cost.
10. As a kernel author, I want a ceiling on how many variants are tried and an early-stop when no improvement is found, so that a run can't silently burn hours of instance time.
11. As a kernel author, I want a clear "no improvement found" result when nothing beats the baseline, so that I'm not left guessing.
12. As a kernel author, I want to supply representative inputs for the correctness check, so that equivalence is judged on data shaped like my real workload.
13. As a kernel author, I want the capture step fully automated (env vars, artifact discovery, copy-back), so that I never SSH around hunting for NEFF/NTFF files.
14. As a kernel author, I want a portable `profile.json` for every run, so that results are inspectable, diffable, and reusable later.
15. As a kernel author, I want two runs flagged as non-comparable when their kernel/shape/dtype/instance differ, so that I'm never shown a misleading diff.
16. As a kernel author, I want run results persisted as baselines, so that I can compare a kernel against its own history over time.
17. As a kernel author, I want a per-variant cost figure (later milestone), so that I can rank by dollars, not just latency.
18. As a Neuron user already using AWS's agentic skills, I want this tool to reuse their capture/JSON output when present, so that I'm not running two competing capture stacks.
19. As an open-source user, I want to install the skill by cloning into `~/.claude/skills/` (or as a plugin), so that adoption is one step.
20. As a contributor, I want the value logic (verify/parse/diff/cost) to be standalone, importable scripts, so that it can later power a headless CLI and CI Action without a rewrite.
21. As a maintainer, I want the four pure modules unit-tested with fixtures, so that I can change them confidently without spinning up hardware.
22. As a cost-conscious user, I want a whole variant sweep to reuse a single warm instance, so that I'm not paying launch overhead per variant.
23. As a kernel author, I want the tool to tell me which measurement source it parsed (and its fidelity), so that I trust the numbers even as AWS deprecates older formats.

## Implementation Decisions

**Architecture: a closed loop over deep modules.** The orchestrating skill runs:
`seed → generate → compile-gate → correctness-gate → profile → diff/rank → report`. The
correctness gate sits before profiling; a variant that fails compile or correctness never
reaches the speed comparison.

**Deep modules (built/tested in isolation):**
- `verify` — correctness gate. Interface: `(reference_kernel, candidate_kernel, input_sets, rtol, atol) → {verified, max_abs_err, max_rel_err}`. Reuses the matmul tutorial's `torch.allclose(atol=1e-4, rtol=1e-2)` pattern (AWS's `neuron-framework-equivalence` skill is **model-level**, not kernel-level, so it does not apply here). **Hardened against reward-hacking** (AccelOpt finding: LLMs game a weak checker via partial results): use multiple randomized input sets per variant, compare the *full* output, and keep the reference hidden from the generator. Load-bearing.
- `parse` — normalize a profiler's raw output into the `profile.json` schema. Interface: `(raw_tool_output, meta) → profile.json`. Pure.
- `diff` — compare/rank profiles. Interface: `(baseline, candidate[]) → {ranked[], comparability_ok}`. Pure. Refuses on mismatched `comparability_key`.
- `cost` — attach dollars (later milestone). Interface: `(profile.json, price_table) → {usd_per_1m_units, usd_per_hr_basis}`. Pure. Uses **measured concurrent throughput**, not 1/latency; price table is pinned and region-aware.

**Shallow / IO / external modules (integration-tested):**
- `capture` — **compose AWS's `neuron-nki-profiling` skill** (do not reimplement). It runs **on-instance** (agent is co-located with hardware — no SSH/copy), sets `NEURON_RT_INSPECT_*` env vars, runs the kernel, finds the NKI NEFF, and emits `neuron-explorer view --output-format summary-json`. Our code wraps/parses that output.
- `compile` — wrap `neuronx-cc` (`--target trn2 --lnc 1`); return compiled artifact or structured error.
- `generate` — **compose AWS's `neuron-nki-writing` skill** and/or call Claude with `(seed, original kernel source, prior compile error?)` → variant source. Bounded retries on compile failure.
- `loop` — the skill orchestrator (**the net-new moat**); enforces `--max-variants`, beam-search selection, early-stop, optimization-memory reuse, and the gate ordering. AWS ships the blocks; nobody ships this loop.

**`profile.json` schema (provisional, locked after the step-1 hardware spike):** includes
`run_id`, `variant_id`, `kernel`, `instance_type`, `sdk_version`, `captured_at`, a
`comparability_key` (kernel + input_shapes + dtype + instance_type), a `metrics` block
(latency p50/p99, throughput, per-engine utilization, compute-vs-memory-bound, dma
throughput, top ops), and a `correctness` block (verified, rtol/atol, max_abs_err).

**Invocation contract (v0):**
`/neuron-optimize --instance i-… | --launch <type> --kernel <path> --inputs <path> [--seed <text> | --auto] [--max-variants 4] [--max-compile-retries 3] [--tol 1e-3]`

**Measurement source (prerequisite gate):** the parsed source must be confirmed on real
hardware before v0 is buildable — `neuron-profile --output-format json` is deprecated in
SDK 2.29; candidates are the agentic `neuron-nki-profiling` JSON, Neuron Explorer export,
or Perfetto traces (lower fidelity). Decision recorded after the hands-on spike.

**Reuse over rebuild:** seed variants from Neuron Explorer / agentic-skill suggestions
when available; reuse `neuron-nki-profiling` for capture/JSON rather than reimplementing.

**Cost control as a first-class concern:** one warm instance per sweep, bounded variant
count, early-stop when no correct variant beats the baseline.

**Extractability:** all four pure modules are standalone importable scripts so v3 (a
headless CLI + GitHub Action) is a lift, not a rewrite. (A Claude skill cannot power CI;
CI needs a binary + exit code.)

## Testing Decisions

**What makes a good test here:** assert external behavior, not internals. For the pure
modules, that means: given fixture inputs (sample raw profiler output, two recorded output
tensors, two `profile.json` files, a price table), assert the returned schema/verdict/
ranking/dollar figure — never assert how the function computed it.

**Modules tested (unit, fixture-driven, no hardware/LLM):**
- `verify` — **priority.** Tests: identical kernels → verified; within-tolerance noise → verified; an **intentionally-wrong variant → rejected** (the trust test, User Story 6); tolerance boundary behavior.
- `parse` — raw fixture output → expected `profile.json`; malformed/partial input → clear error.
- `diff` — known baseline+candidate → expected ranking; mismatched `comparability_key` → refuses/warns (User Story 15).
- `cost` — known throughput + price table → expected `$/1M-units`; uses concurrent throughput basis.

**Integration / smoke tests (need hardware or mocked external calls):** `capture`,
`compile`, `generate`, `loop` — covered by a single end-to-end smoke run against a real
instance once the measurement source is locked; not heavily mocked in v0 (avoid brittle
mocks before interfaces are proven).

**Prior art:** none in-repo yet (greenfield). Establish the fixture-based pure-module test
pattern as the project convention; mirror it for future modules.

## Out of Scope

- LLM-generated **model-level** variants (whole models / config / parallelism) — that's a
  later stretch milestone (v4); v0–v2 target single NKI kernels.
- CI regression gate / headless CLI / GitHub Action — exploratory v3, gated on adoption and
  the Neuron-in-CI runner story.
- Cost framing — deferred to v2 (after the loop and ranking work).
- A GUI / web report (HTML report) — deferred; output is terminal + `profile.json`.
- Competing with AWS on kernel *authoring* or *suggestions* — we reuse those; the product
  is the loop.
- Self-hosted CI runner infrastructure.

## Further Notes

- **Strategic posture:** the moat is the autonomous generate→verify→profile→rank loop, not
  capture (AWS gives capture away). Capture and diff are plumbing.
- **Audience shift (accepted):** leading with kernel code-gen targets NKI kernel authors
  (AWS's home turf); we win on the loop they don't have, not on authoring.
- **Biggest risks:** (1) correctness-gate fidelity for FP kernels, (2) LLM compile-failure
  rate, (3) sweep cost/time on $/hr hardware. All have explicit mitigations above.
- **Prerequisite before coding:** the hands-on hardware spike (one kernel, run AWS's
  profiling + agentic skills by hand, capture the JSON each emits) — resolves the
  measurement-source question and provides the v0 reference. Can be driven via AWS CLI.
- No GitHub remote on this repo yet, so this PRD is a local file. When a remote exists,
  publish it to the tracker with the `ready-for-agent` label and split into per-milestone
  issues (v0 loop, verify gate, schema, capture).

## Composition with neuron-agentic-development (decision)

We **compose AWS's open-source `neuron-agentic-development`** (agents + skills) as the
building blocks and ship only the **orchestrator loop** on top. Full step-by-step mapping
in `docs/ORCHESTRATION.md`. Key points:
- Runs **on-instance** (AWS co-locates agent + hardware; no SSH/copy — drops rev-2
  Approach A). Recommended dev box per AWS: `trn2.3xlarge` in `sa-east-1`/`ap-southeast-4`.
- Reuse: `neuron-nki-profiling` (capture), `neuron-nki-profile-querying` (`Summary` table →
  `total_time`, `mfu_estimated_percent`, engine %), `neuron-nki-writing`/writer-agent
  (generate), `neuron-nki-debugger-agent` (compile-fix), `neuron-nki-profile-analysis-agent`
  (bottleneck seed; also does a *single* before/after compare).
- Net-new (the moat): the automated multi-variant **loop**, hardened correctness gate,
  cross-variant **ranking + roofline % of peak**, **optimization memory**, cost framing,
  and (later) CI. AWS has no multi-variant loop, no memory, no cost, no CI.
- Data source RESOLVED: `neuron-explorer view --output-format summary-json` / the
  profile-querying DuckDB `Summary` table — not the deprecated `neuron-profile` JSON.

## Prior art: AccelOpt (arXiv 2511.15915)

Closest research system (LLM agentic NKI kernel optimization on Trainium). We borrow its
techniques and **credit it in the blog/how-to**: beam search over variants; an
optimization memory of slow→fast (and failed) rewrites with summaries (thresholds
`t_pos=1.04`/`t_neg=1.15`); roofline **% of peak** as the headline metric + early-stop
near ~80% peak; **hardened correctness checking** (their documented failure: LLMs game a
weak checker via partial results → fake speedup); profiling rigor (warmup, multiple rounds,
variance threshold). Results for context: Trn1 49%→61% of peak, Trn2 45%→59%; matched
Claude Sonnet 4 quality at 26× cheaper with open models. We differ by being a
developer-facing skill that **composes AWS's agentic-dev** and ships a verified, shareable
diff + cost + CI — not a research search harness.

## First e2e validation (no LLM yet)

Validate capture→verify→profile→diff on a known-good pair from `nki-samples` matmul:
baseline `nki_matmul_tiled_` (arithmetic intensity 102, memory-bound) vs "variant 2"
`nki_matmul_fully_optimized_` (683, compute-bound), 4096×1024·1024×2048 bf16, correctness
via the tutorial's `torch.allclose`. Confirms the predicted speedup with real `total_time`/
`mfu_estimated_percent` and becomes the first how-to/blog post. See `docs/ORCHESTRATION.md`.
