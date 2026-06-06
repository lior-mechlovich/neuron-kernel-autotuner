"""The portable profile.json schema and helpers.

A profile.json is the single record passed between every step of the loop. It is produced
by `parse` from neuron-explorer's summary-json (or the profile-querying `Summary` table)
and consumed by `diff` and `cost`.

Schema is provisional (v0) and locked after the first hardware run. Two profiles are only
comparable when their `comparability_key` matches.
"""

from __future__ import annotations

from typing import Any

SCHEMA_VERSION = "0.1"

# Keys that define whether two runs may be diffed against each other.
COMPARABILITY_FIELDS = ("kernel", "input_shapes", "dtype", "instance_type")


def make_profile(
    *,
    run_id: str,
    kernel: str,
    instance_type: str,
    sdk_version: str,
    input_shapes: list[list[int]],
    dtype: str,
    metrics: dict[str, Any],
    variant_id: str | None = None,
    correctness: dict[str, Any] | None = None,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Build a normalized profile.json dict. `captured_at` is caller-supplied (ISO-8601)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "variant_id": variant_id,
        "kernel": kernel,
        "instance_type": instance_type,
        "sdk_version": sdk_version,
        "captured_at": captured_at,
        "comparability_key": {
            "kernel": kernel,
            "input_shapes": input_shapes,
            "dtype": dtype,
            "instance_type": instance_type,
        },
        "metrics": {
            "latency_us": metrics.get("latency_us"),
            "mfu_percent": metrics.get("mfu_percent"),  # % of peak (roofline)
            "tensor_engine_active_percent": metrics.get("tensor_engine_active_percent"),
            "dma_active_percent": metrics.get("dma_active_percent"),
            "compute_vs_memory_bound": metrics.get("compute_vs_memory_bound"),
            "throughput_units_s": metrics.get("throughput_units_s"),
        },
        "correctness": correctness or {"verified": None},
    }


def comparability_key(profile: dict[str, Any]) -> tuple:
    """Return a hashable comparability key for a profile."""
    ck = profile.get("comparability_key", {})
    return tuple(_freeze(ck.get(f)) for f in COMPARABILITY_FIELDS)


def comparable(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """True iff two profiles may be diffed (same kernel shape/dtype/instance)."""
    return comparability_key(a) == comparability_key(b)


def _freeze(v: Any) -> Any:
    if isinstance(v, list):
        return tuple(_freeze(x) for x in v)
    return v
