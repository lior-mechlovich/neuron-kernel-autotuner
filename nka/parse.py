"""parse: neuron-explorer summary-json / Summary-table row -> profile.json.

The exact summary-json field names are confirmed on the first hardware run; until then we
map defensively across candidate key names so the parser survives minor schema drift across
Neuron SDK releases. Pure: no hardware, no IO beyond an optional file read.
"""

from __future__ import annotations

import json
from typing import Any

from .profile_schema import make_profile

# Candidate source-key names -> our normalized metric name. First match wins.
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "latency_us": ("total_time", "total_time_us", "latency_us", "duration_us"),
    "mfu_percent": ("mfu_estimated_percent", "mfu_percent", "model_flops_utilization_percent"),
    "tensor_engine_active_percent": (
        "tensor_engine_active_time_percent",
        "pe_active_percent",
        "tensorengine_active_percent",
    ),
    "dma_active_percent": ("dma_active_time_percent", "dma_active_percent"),
    "throughput_units_s": ("throughput", "throughput_units_s", "ops_per_second"),
}


def _first(row: dict[str, Any], names: tuple[str, ...]) -> Any:
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None


def _classify_bound(tensor_pct: Any, dma_pct: Any) -> str | None:
    """Heuristic: more DMA-active than tensor-active => memory bound, and vice versa."""
    if tensor_pct is None or dma_pct is None:
        return None
    if dma_pct - tensor_pct > 10:
        return "memory"
    if tensor_pct - dma_pct > 10:
        return "compute"
    return "balanced"


def parse_summary(
    summary: dict[str, Any] | str,
    *,
    run_id: str,
    kernel: str,
    instance_type: str,
    sdk_version: str,
    input_shapes: list[list[int]],
    dtype: str,
    variant_id: str | None = None,
    correctness: dict[str, Any] | None = None,
    captured_at: str | None = None,
) -> dict[str, Any]:
    """Normalize one neuron-explorer Summary row (dict or JSON string) into profile.json."""
    if isinstance(summary, str):
        summary = json.loads(summary)
    # summary-json may wrap the row in a list or under a key; unwrap to a flat dict.
    row = _unwrap_summary_row(summary)

    tensor_pct = _first(row, _METRIC_ALIASES["tensor_engine_active_percent"])
    dma_pct = _first(row, _METRIC_ALIASES["dma_active_percent"])
    metrics = {
        "latency_us": _first(row, _METRIC_ALIASES["latency_us"]),
        "mfu_percent": _first(row, _METRIC_ALIASES["mfu_percent"]),
        "tensor_engine_active_percent": tensor_pct,
        "dma_active_percent": dma_pct,
        "throughput_units_s": _first(row, _METRIC_ALIASES["throughput_units_s"]),
        "compute_vs_memory_bound": _classify_bound(tensor_pct, dma_pct),
    }
    return make_profile(
        run_id=run_id,
        kernel=kernel,
        instance_type=instance_type,
        sdk_version=sdk_version,
        input_shapes=input_shapes,
        dtype=dtype,
        metrics=metrics,
        variant_id=variant_id,
        correctness=correctness,
        captured_at=captured_at,
    )


def _unwrap_summary_row(summary: dict[str, Any]) -> dict[str, Any]:
    """Find the flat metrics dict inside common summary-json wrappers."""
    for key in ("Summary", "summary", "data", "rows"):
        v = summary.get(key) if isinstance(summary, dict) else None
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return v[0]
        if isinstance(v, dict):
            return v
    return summary
