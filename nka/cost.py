"""cost: attach dollars to a profile.

The reason people choose Trainium/Inferentia is cost-per-unit, yet no first-party tool frames
profiling in dollars. We multiply measured throughput by the instance on-demand price.

Cost-per-1M-units uses MEASURED throughput (units/s), not 1/latency -- served economics depend
on achieved concurrency, not single-call latency (a distinction AccelOpt and the design doc
both flag). Pure: price table is passed in, not fetched.
"""

from __future__ import annotations

from typing import Any

# Indicative us-west-2 on-demand $/hr. Pinned, region-aware; override per call.
DEFAULT_PRICE_PER_HR = {
    "trn1.2xlarge": 1.34,
    "trn1.32xlarge": 21.50,
    "inf2.xlarge": 0.76,
    "inf2.8xlarge": 1.97,
    "trn2.48xlarge": 38.43,
}


def cost_of(
    profile: dict[str, Any],
    *,
    price_per_hr: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Return cost metrics for a single profile.

    `usd_per_1m_units` is None when throughput is unknown (we do NOT fake it from latency).
    """
    prices = price_per_hr or DEFAULT_PRICE_PER_HR
    inst = profile.get("instance_type")
    hr = prices.get(inst)
    m = profile.get("metrics") or {}
    tput = m.get("throughput_units_s")

    usd_per_hr = hr
    usd_per_1m_units = None
    basis = "throughput"
    if hr is not None and tput:
        usd_per_sec = hr / 3600.0
        usd_per_1m_units = round(usd_per_sec / tput * 1_000_000, 6)
    elif tput in (None, 0):
        basis = "unknown_throughput"

    return {
        "instance_type": inst,
        "usd_per_hr": usd_per_hr,
        "throughput_units_s": tput,
        "usd_per_1m_units": usd_per_1m_units,
        "basis": basis,
    }


def cost_delta(baseline: dict[str, Any], candidate: dict[str, Any], **kw: Any) -> dict[str, Any]:
    """Cost saving of candidate vs baseline (same instance assumed)."""
    b = cost_of(baseline, **kw)
    c = cost_of(candidate, **kw)
    saving = None
    if b["usd_per_1m_units"] and c["usd_per_1m_units"]:
        saving = round((b["usd_per_1m_units"] - c["usd_per_1m_units"]) / b["usd_per_1m_units"], 4)
    return {"baseline": b, "candidate": c, "fraction_cheaper": saving}
