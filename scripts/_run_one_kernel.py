#!/usr/bin/env python3
"""Run one vendored matmul kernel once via the jit+xla path to emit a NEFF (+ profile).

Run with NEURON_RT_INSPECT_* env set so the Neuron runtime dumps the NEFF/NTFF into
NEURON_RT_INSPECT_OUTPUT_DIR. One kernel per process (the inspect env must be set before
the runtime initializes). `neuron-bench` then measures device latency from the NEFF.
"""
import sys
from pathlib import Path

import torch
import torch_xla.core.xla_model as xm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "matmul"))
import matrix_multiplication_nki_kernels as K  # noqa: E402

M, Kk, N = 4096, 1024, 2048


def main() -> int:
    name = sys.argv[1]
    fn = getattr(K, name)
    d = xm.xla_device()
    lhs = torch.rand((M, Kk), dtype=torch.bfloat16)
    rhs = torch.rand((Kk, N), dtype=torch.bfloat16)
    lhsT = lhs.t().contiguous().to(d)
    rhs = rhs.to(d)
    out = fn(lhsT, rhs)
    xm.mark_step()
    xm.wait_device_ops()
    print(f"ran {name} -> {tuple(out.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
