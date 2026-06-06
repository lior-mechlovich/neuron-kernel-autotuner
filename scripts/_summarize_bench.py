#!/usr/bin/env python3
"""Summarize neuron-bench device-latency results for the matmul baseline vs optimized."""
import glob
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def med_us(variant: str):
    """Median NeuronCore (device) latency in us from neuron-bench's i1 (DataParallel=1) run."""
    cands = glob.glob(f"/tmp/bench_{variant}/*_i1_LIBMODE/nc_latency_data.json")
    if not cands:
        return None, 0
    data = json.load(open(cands[0]))["latency_data"]
    series = data["0"] if isinstance(data, dict) else data
    return statistics.median(series), len(series)


def main() -> int:
    base_us, base_n = med_us("tiled")
    opt_us, opt_n = med_us("fully_optimized")
    if base_us is None or opt_us is None:
        print("missing bench data:", base_us, opt_us)
        return 1
    speedup = round(base_us / opt_us, 3)
    out = {
        "instance_type": "trn1.2xlarge",
        "metric": "neuroncore_device_latency_us (neuron-bench, median)",
        "baseline": {"kernel": "nki_matmul_tiled_", "latency_us": round(base_us, 2), "samples": base_n},
        "optimized": {"kernel": "nki_matmul_fully_optimized_", "latency_us": round(opt_us, 2), "samples": opt_n},
        "speedup_x": speedup,
        "shapes": {"lhs": [4096, 1024], "rhs": [1024, 2048], "dtype": "bfloat16"},
    }
    (REPO / "results").mkdir(exist_ok=True)
    (REPO / "results" / "measured.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSPEEDUP (device, tiled/fully_optimized): {speedup}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
