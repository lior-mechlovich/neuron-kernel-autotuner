# I built an AI that optimizes AWS Trainium kernels — and proved it on real silicon

*Every Neuron tool tells you what to fix. None of them fix it, run it, and prove it. So I built the loop that does — and learned that the way everyone measures kernels is lying to them.*

---

## The gap nobody is filling

If you write kernels for AWS Trainium or Inferentia, you use NKI — the Neuron Kernel Interface — and you lean on AWS's tooling to make them fast. That tooling is genuinely good. **Neuron Explorer** profiles your kernel and, since SDK 2.29, even uses Claude to suggest optimizations. AWS's open-source **neuron-agentic-development** ships Claude Code skills that write kernels, profile them, query the profile, and debug compile errors.

But notice what every one of these does: it **stops at the suggestion**. "Your kernel is memory-bound, try blocking the free dimension." Great — now *you* go write the variant, recompile it, check it still produces the right numbers, re-profile it, and eyeball whether it actually got faster. The loop is manual. And the riskiest part — *did my faster kernel stay correct?* — is on you.

So I built the missing piece: **[neuron-kernel-autotuner](https://github.com/lior-mechlovich/neuron-kernel-autotuner)**, a tool that closes the loop. You point it at a kernel; it generates variants, compiles them, **verifies they're numerically correct**, profiles each on real hardware, ranks them by actual speedup, and hands you back the fastest one that's still correct — with the proof.

It doesn't reinvent AWS's tools. It **orchestrates** them and adds the loop they don't have.

## How it works

The whole thing is a closed loop:

```
kernel ─► baseline ─► find bottleneck ─► GENERATE variant ─► COMPILE ✓
              ▲                                                   │
              │                                                   ▼
        keep top-K ◄── RANK ◄── PROFILE ◄── CORRECTNESS ✓ ◄──────┘
```

Each step maps onto a piece AWS already ships:

| Step | Who does it |
|---|---|
| Capture / profile | AWS `neuron-nki-profiling` + `neuron-nki-profile-querying` |
| Find the bottleneck | AWS `neuron-nki-profile-analysis-agent` |
| Generate a variant | AWS `neuron-nki-writing` |
| Fix compile errors | AWS `neuron-nki-debugging` |
| **Verify correctness** | **us — a hardened gate** |
| **Rank variants + memory + cost** | **us — the loop** |

The value isn't any single step. It's making them an automated, measured, ranked, *verified* loop you run with one command.

### The one rule that matters: correctness before speed

There's a paper called [AccelOpt](https://arxiv.org/abs/2511.15915) — a research system doing LLM-driven NKI kernel optimization. Buried in it is a warning that shaped my whole design: **LLMs will cheat the correctness check**. Given a weak verifier, the model learns to compute *part* of the output, pass a lazy check, and report a huge fake speedup.

So the correctness gate in this tool is deliberately hard to fool:

- it checks **multiple randomized input sets**, not one cherry-picked case
- it compares the **full output**, every element, within tolerance
- the reference answer is **never shown to the generator**

A kernel that's faster but wrong is never reported as a winner. Ever. That gate runs *before* anything gets profiled.

## Proving it: recreating a known speedup on real hardware

Claims are cheap. So I ran the whole thing end-to-end on a real Trainium instance, using a kernel pair where the answer is already known.

AWS's [nki-samples](https://github.com/aws-neuron/nki-samples) matmul tutorial ships five versions of the same matrix multiply, from a basic tiled one to a fully optimized one. Two of them make a perfect ground-truth test:

- **baseline:** `nki_matmul_tiled_` — arithmetic intensity ~102, **memory-bound**
- **optimized:** `nki_matmul_fully_optimized_` — arithmetic intensity ~683, **compute-bound**

Same math (4096×1024 · 1024×2048, bf16), same correct answer, very different performance. If my measure→verify→diff pipeline is any good, it should confirm the optimized one is faster *and* still correct — with numbers.

I launched a `trn1.2xlarge` in us-west-2 (NeuronCore-v2, about \$1.34/hr), copied the repo over, and ran one command:

```bash
bash scripts/bootstrap_trn1.sh
```

That verifies both kernels against a PyTorch reference over three input sets, benchmarks each on the NeuronCore, and writes the artifacts.

### The result

| variant | device latency | DMA-active | correct? |
|---|--:|--:|:--:|
| `nki_matmul_tiled_` (baseline) | **463 µs** | 16.0% | ✅ |
| `nki_matmul_fully_optimized_` | **273 µs** | 7.5% | ✅ |
| **speedup** | **1.70×** | | |

Both verified correct, and a real **1.70× speedup** on the chip. And the profile tells you *why*: DMA-active time dropped from 16% to 7.5% — the optimized kernel is far less memory-bound, exactly what the arithmetic-intensity jump (102 → 683) predicts.

## The plot twist: how everyone measures kernels is lying to them

Here's the part I didn't expect.

My first benchmark timed the kernels the obvious way — call it through the framework (torch_xla), wrap it in `time.perf_counter`, take the median. The result?

> Both kernels: **~57 milliseconds.** Speedup: **1.00×.**

The optimized kernel looked *identical* to the baseline. If I'd trusted that number, I'd have concluded the optimization did nothing.

It was wrong. Framework wall-clock is dominated by dispatch and host↔device transfer overhead — tens of milliseconds that are **the same for both kernels** and completely bury the sub-millisecond compute difference. The kernel that actually matters is a rounding error in that measurement.

Only when I measured **device latency** — what the NeuronCore actually spends executing — did the truth appear: 463 µs vs 273 µs, the real 1.70×.

This is the entire thesis of the project, proven by accident. **An autotuner cannot rank kernels on wall-clock; it has to rank on device metrics.** That's why the loop is built around the on-device profile, not a stopwatch. If your kernel benchmarking uses framework-level timing, it's probably hiding your wins (and your regressions).

## What's next

The matmul run proves the spine — measure, verify, diff — on ground truth. The roadmap builds out from there ([issues on GitHub](https://github.com/lior-mechlovich/neuron-kernel-autotuner/issues)):

- **v1:** swap the fixed pair for **LLM-generated** variants, beam search, ranking across many
- **v2:** an optimization memory that reuses slow→fast rewrites across runs, plus cost-per-token framing
- **v3:** a CI gate that fails a PR when a kernel regresses

## Try it

Everything is open source (Apache-2.0), the pure logic has unit tests, and the proof is reproducible with one command on any Trainium box:

**[github.com/lior-mechlovich/neuron-kernel-autotuner](https://github.com/lior-mechlovich/neuron-kernel-autotuner)**

Credits where due: this composes AWS's [neuron-agentic-development](https://github.com/aws-neuron/neuron-agentic-development), uses a test case from [nki-samples](https://github.com/aws-neuron/nki-samples), and borrows its core ideas — beam search, optimization memory, and the correctness-hardening lesson — from [AccelOpt](https://arxiv.org/abs/2511.15915).

*AWS gives you the parts. This is the loop.*
