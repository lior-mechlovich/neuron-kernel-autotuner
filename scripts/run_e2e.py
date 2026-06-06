#!/usr/bin/env python3
"""On-instance e2e: baseline vs optimized matmul, measured + verified + diffed.

This is the first end-to-end proof. It runs ON a Trainium/Inferentia instance and uses the
vendored nki-samples matmul kernels as a known-good baseline/optimized pair, so we validate
the measure -> verify -> diff plumbing BEFORE wiring up LLM variant generation.

What it does for each variant (default: nki_matmul_tiled_ vs nki_matmul_fully_optimized_):
  1. run the NKI kernel on device
  2. verify numerical correctness vs a torch reference over MULTIPLE input sets (hardened gate)
  3. measure device latency (median over N timed iters, after warmup)
  4. (best effort) capture a neuron-explorer profile for % of peak

Then it diffs optimized vs baseline and writes results/ artifacts.

Usage (on the instance, inside the Neuron venv):
    python scripts/run_e2e.py --target trn1 --iters 50 --input-sets 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "examples" / "matmul"))

from nka.diff import rank  # noqa: E402
from nka.profile_schema import make_profile  # noqa: E402
from nka.verify import verify_outputs  # noqa: E402

# M, K, N from the tutorial benchmark.
M, K, N = 4096, 1024, 2048
DTYPE_NAME = "bfloat16"


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _make_inputs(torch, device, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lhs = torch.rand((M, K), dtype=torch.bfloat16, generator=g)
    rhs = torch.rand((K, N), dtype=torch.bfloat16, generator=g)
    lhsT = lhs.t().contiguous()
    return lhs.to(device), rhs.to(device), lhsT.to(device)


def _to_np(torch, t):
    return t.to(torch.float32).cpu().numpy()


def _bench(fn, lhsT, rhs, torch, warmup, iters):
    """Mean device latency (us) via xla mark_step/wait_device_ops, NO d2h in the timed loop.

    Timing wall-clock with a host copy (.cpu()) per iter is dominated by the ~tens-of-ms d2h
    transfer, which is identical for both variants and hides the kernel difference. Instead we
    enqueue each call and flush it with mark_step (so every iter actually executes on device),
    then sync once with wait_device_ops and divide by iters.
    """
    import torch_xla.core.xla_model as xm

    for _ in range(warmup):
        out = fn(lhsT, rhs)
        xm.mark_step()
    xm.wait_device_ops()

    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(lhsT, rhs)
        xm.mark_step()
    xm.wait_device_ops()
    per_iter_us = (time.perf_counter() - t0) / iters * 1e6
    return per_iter_us, out


def run_variant(name, fn, torch, device, args):
    """Return (profile, ref_outputs, cand_outputs) for a kernel variant."""
    ref_sets, cand_sets = [], []
    last_out = None
    for i in range(args.input_sets):
        lhs, rhs, lhsT = _make_inputs(torch, device, seed=100 + i)
        ref = torch.matmul(lhs, rhs)
        out = fn(lhsT, rhs)
        ref_sets.append(_to_np(torch, ref))
        cand_sets.append(_to_np(torch, out))
        last_out = (lhsT, rhs)

    verdict = verify_outputs(ref_sets, cand_sets, rtol=1e-2, atol=1e-4,
                             min_input_sets=min(2, args.input_sets))

    lat_us, _ = _bench(fn, last_out[0], last_out[1], torch, args.warmup, args.iters)

    profile = make_profile(
        run_id=f"{name}-{int(time.time())}",
        variant_id=name,
        kernel="matmul",
        instance_type=args.instance_type,
        sdk_version=os.environ.get("NEURON_SDK_VERSION", "unknown"),
        input_shapes=[[M, K], [K, N]],
        dtype=DTYPE_NAME,
        metrics={"latency_us": round(lat_us, 3)},
        correctness=verdict,
        captured_at=_utcnow(),
    )
    return profile


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="trn1", help="neuronx-cc target (trn1/trn2/inf2)")
    ap.add_argument("--instance-type", default=os.environ.get("INSTANCE_TYPE", "trn1.2xlarge"))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--input-sets", type=int, default=3)
    ap.add_argument("--baseline", default="nki_matmul_tiled_")
    ap.add_argument("--optimized", default="nki_matmul_fully_optimized_")
    ap.add_argument("--out", default=str(REPO / "results"))
    args = ap.parse_args()

    os.environ.setdefault("NEURON_CC_FLAGS", f"--target {args.target}")

    import torch  # noqa: F401
    import torch_xla.core.xla_model as xm  # type: ignore
    import matrix_multiplication_nki_kernels as K_  # vendored

    device = xm.xla_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    variants = {args.baseline: getattr(K_, args.baseline),
                args.optimized: getattr(K_, args.optimized)}
    profiles = {}
    for name, fn in variants.items():
        print(f"[run] {name} ...", flush=True)
        p = run_variant(name, fn, torch, device, args)
        profiles[name] = p
        (out_dir / f"profile_{name}.json").write_text(json.dumps(p, indent=2))
        print(f"  verified={p['correctness']['verified']} latency_us={p['metrics']['latency_us']}")

    base = profiles[args.baseline]
    opt = profiles[args.optimized]
    result = rank(base, [opt])
    (out_dir / "diff.json").write_text(json.dumps(result, indent=2))

    best = result["best"]
    print("\n==== RESULT ====")
    print(f"baseline  {args.baseline}: {base['metrics']['latency_us']} us "
          f"(verified={base['correctness']['verified']})")
    print(f"optimized {args.optimized}: {opt['metrics']['latency_us']} us "
          f"(verified={opt['correctness']['verified']})")
    if best:
        print(f"SPEEDUP: {best['speedup_x']}x")
    else:
        print("No verified speedup (check skipped:", result["skipped"], ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
