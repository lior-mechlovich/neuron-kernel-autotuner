"""diff: rank verified variants against a baseline.

Headline metric is wall-clock speedup (baseline latency / candidate latency), with % of peak
(`mfu_percent`, the roofline metric AccelOpt emphasizes) reported alongside so a "faster"
result that is still far from peak is visible as such.

Refuses to compare profiles whose `comparability_key` differs (different shape/dtype/instance)
-- a silent cross-shape diff is worse than no diff. Pure.
"""

from __future__ import annotations

from typing import Any

from .profile_schema import comparable


def _latency(p: dict[str, Any]) -> float | None:
    return (p.get("metrics") or {}).get("latency_us")


def _verified(p: dict[str, Any]) -> bool:
    return bool((p.get("correctness") or {}).get("verified"))


def rank(
    baseline: dict[str, Any],
    candidates: list[dict[str, Any]],
    *,
    require_verified: bool = True,
) -> dict[str, Any]:
    """Rank candidates vs baseline by speedup. Returns a structured result.

    Only `comparable` and (optionally) verified candidates are ranked; the rest are reported
    under `skipped` with a reason so nothing is silently dropped.
    """
    base_lat = _latency(baseline)
    if base_lat is None or base_lat <= 0:
        raise ValueError("baseline has no positive latency_us")

    ranked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for c in candidates:
        cid = c.get("variant_id") or c.get("run_id")
        if not comparable(baseline, c):
            skipped.append({"variant": cid, "reason": "not_comparable"})
            continue
        if require_verified and not _verified(c):
            skipped.append({"variant": cid, "reason": "not_verified"})
            continue
        lat = _latency(c)
        if lat is None or lat <= 0:
            skipped.append({"variant": cid, "reason": "no_latency"})
            continue
        m = c.get("metrics") or {}
        ranked.append({
            "variant": cid,
            "latency_us": lat,
            "speedup_x": round(base_lat / lat, 4),
            "mfu_percent": m.get("mfu_percent"),
            "compute_vs_memory_bound": m.get("compute_vs_memory_bound"),
        })

    ranked.sort(key=lambda r: r["speedup_x"], reverse=True)
    best = ranked[0] if ranked else None
    base_m = baseline.get("metrics") or {}
    return {
        "baseline": {
            "variant": baseline.get("variant_id") or baseline.get("run_id"),
            "latency_us": base_lat,
            "mfu_percent": base_m.get("mfu_percent"),
            "compute_vs_memory_bound": base_m.get("compute_vs_memory_bound"),
        },
        "best": best,
        "ranked": ranked,
        "skipped": skipped,
    }
