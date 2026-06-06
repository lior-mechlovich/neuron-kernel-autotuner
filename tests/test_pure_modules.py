"""Unit tests for the four pure modules: parse, verify, diff, cost.

No hardware, no LLM -- fixture-driven. These are the value-bearing logic; the hardware/LLM
steps get integration smoke tests separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from nka.cost import cost_delta, cost_of
from nka.diff import rank
from nka.parse import parse_summary
from nka.profile_schema import comparable, make_profile


# ---------------------------------------------------------------- parse


def _summary(total_time, mfu, te, dma):
    return {"Summary": [{
        "total_time": total_time,
        "mfu_estimated_percent": mfu,
        "tensor_engine_active_time_percent": te,
        "dma_active_time_percent": dma,
    }]}


def test_parse_maps_summary_fields_and_classifies_bound():
    p = parse_summary(
        _summary(1000.0, 18.0, 20.0, 70.0),
        run_id="r1", kernel="matmul_tiled", instance_type="trn1.2xlarge",
        sdk_version="2.29", input_shapes=[[4096, 1024], [1024, 2048]], dtype="bfloat16",
    )
    assert p["metrics"]["latency_us"] == 1000.0
    assert p["metrics"]["mfu_percent"] == 18.0
    # dma(70) >> tensor(20) -> memory bound
    assert p["metrics"]["compute_vs_memory_bound"] == "memory"
    assert p["comparability_key"]["dtype"] == "bfloat16"


def test_parse_accepts_alias_keys_and_json_string():
    raw = '{"data": {"total_time_us": 500, "mfu_percent": 60, ' \
          '"tensor_engine_active_time_percent": 80, "dma_active_time_percent": 30}}'
    p = parse_summary(raw, run_id="r2", kernel="k", instance_type="trn1.2xlarge",
                      sdk_version="2.29", input_shapes=[[1, 1]], dtype="bf16")
    assert p["metrics"]["latency_us"] == 500
    assert p["metrics"]["compute_vs_memory_bound"] == "compute"  # tensor(80) >> dma(30)


# ---------------------------------------------------------------- verify


def _outs(seed, shape=(64, 64), scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.random(shape) * scale


def test_verify_passes_identical_outputs_over_multiple_sets():
    from nka.verify import verify_outputs
    ref = [_outs(1), _outs(2), _outs(3)]
    cand = [a.copy() for a in ref]
    v = verify_outputs(ref, cand)
    assert v["verified"] is True
    assert v["max_abs_err"] == 0.0


def test_verify_rejects_wrong_variant():
    from nka.verify import verify_outputs
    ref = [_outs(1), _outs(2)]
    cand = [ref[0].copy(), ref[1] + 1.0]  # second set is wrong
    v = verify_outputs(ref, cand)
    assert v["verified"] is False


def test_verify_rejects_single_input_set_even_if_matching():
    """Anti-reward-hack: one cherry-picked passing input set is not enough."""
    from nka.verify import verify_outputs
    ref = [_outs(7)]
    cand = [ref[0].copy()]
    v = verify_outputs(ref, cand)
    assert v["enough_input_sets"] is False
    assert v["verified"] is False


def test_verify_rejects_shape_mismatch():
    from nka.verify import verify_outputs
    ref = [_outs(1, (64, 64)), _outs(2, (64, 64))]
    cand = [_outs(1, (64, 32)), _outs(2, (64, 64))]
    v = verify_outputs(ref, cand)
    assert v["verified"] is False


# ---------------------------------------------------------------- diff


def _profile(vid, lat, mfu, verified=True, kernel="matmul", dtype="bf16",
             shapes=None, inst="trn1.2xlarge"):
    shapes = shapes or [[4096, 1024], [1024, 2048]]
    p = make_profile(run_id=vid, variant_id=vid, kernel=kernel, instance_type=inst,
                     sdk_version="2.29", input_shapes=shapes, dtype=dtype,
                     metrics={"latency_us": lat, "mfu_percent": mfu})
    p["correctness"] = {"verified": verified}
    return p


def test_rank_orders_by_speedup_and_picks_best():
    base = _profile("tiled", 1000.0, 18.0)
    cands = [_profile("hoist", 500.0, 35.0), _profile("fully_opt", 250.0, 68.0)]
    r = rank(base, cands)
    assert r["best"]["variant"] == "fully_opt"
    assert r["best"]["speedup_x"] == 4.0
    assert [c["variant"] for c in r["ranked"]] == ["fully_opt", "hoist"]


def test_rank_skips_unverified_and_noncomparable():
    base = _profile("tiled", 1000.0, 18.0)
    unverified = _profile("bad", 100.0, 90.0, verified=False)
    wrong_shape = _profile("xshape", 100.0, 90.0, shapes=[[2, 2]])
    r = rank(base, [unverified, wrong_shape])
    assert r["best"] is None
    reasons = {s["reason"] for s in r["skipped"]}
    assert reasons == {"not_verified", "not_comparable"}


def test_comparable_true_for_same_key():
    a = _profile("a", 1.0, 1.0)
    b = _profile("b", 2.0, 2.0)
    assert comparable(a, b) is True


# ---------------------------------------------------------------- cost


def test_cost_uses_throughput_not_latency():
    p = make_profile(run_id="r", kernel="k", instance_type="trn1.2xlarge",
                     sdk_version="2.29", input_shapes=[[1, 1]], dtype="bf16",
                     metrics={"latency_us": 1000, "throughput_units_s": 1000})
    c = cost_of(p)
    # $1.34/hr / 3600 / 1000 ops * 1e6 = 0.372...
    assert c["usd_per_1m_units"] == pytest.approx(1.34 / 3600 / 1000 * 1e6, rel=1e-6)
    assert c["basis"] == "throughput"


def test_cost_is_honest_when_throughput_unknown():
    p = make_profile(run_id="r", kernel="k", instance_type="trn1.2xlarge",
                     sdk_version="2.29", input_shapes=[[1, 1]], dtype="bf16",
                     metrics={"latency_us": 1000})
    c = cost_of(p)
    assert c["usd_per_1m_units"] is None
    assert c["basis"] == "unknown_throughput"


def test_cost_delta_reports_fraction_cheaper():
    base = make_profile(run_id="b", kernel="k", instance_type="trn1.2xlarge",
                        sdk_version="2.29", input_shapes=[[1, 1]], dtype="bf16",
                        metrics={"throughput_units_s": 1000})
    cand = make_profile(run_id="c", kernel="k", instance_type="trn1.2xlarge",
                        sdk_version="2.29", input_shapes=[[1, 1]], dtype="bf16",
                        metrics={"throughput_units_s": 4000})
    d = cost_delta(base, cand)
    assert d["fraction_cheaper"] == pytest.approx(0.75)
