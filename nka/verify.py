"""verify: the hardened correctness gate.

AccelOpt (arXiv:2511.15915) documents the key failure mode: an LLM can game a weak checker
by computing partial results and reporting a fake speedup. So the gate is deliberately hard
to fool:

  * MULTIPLE input sets (a variant must be correct on all of them, not one cherry-picked one)
  * FULL-output comparison (every element within tolerance, not a sampled subset)
  * the reference is computed independently and never shown to the generator

Pure: operates on numpy arrays. Hardware-free, so it is unit-tested with fixtures.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

MIN_INPUT_SETS = 2  # a single input set is too easy to overfit / game


def verify_outputs(
    reference_sets: Sequence[np.ndarray],
    candidate_sets: Sequence[np.ndarray],
    *,
    rtol: float = 1e-2,
    atol: float = 1e-4,
    min_input_sets: int = MIN_INPUT_SETS,
) -> dict[str, Any]:
    """Return a verdict dict for a candidate kernel against a reference.

    `reference_sets[i]` and `candidate_sets[i]` are the outputs for input set i.
    A candidate is `verified` only if every input set matches within tolerance AND there are
    at least `min_input_sets` of them.
    """
    if len(reference_sets) != len(candidate_sets):
        raise ValueError("reference_sets and candidate_sets must be the same length")

    n = len(reference_sets)
    enough = n >= min_input_sets
    per_set: list[dict[str, Any]] = []
    all_match = True

    for i, (ref, cand) in enumerate(zip(reference_sets, candidate_sets)):
        ref = np.asarray(ref, dtype=np.float64)
        cand = np.asarray(cand, dtype=np.float64)
        if ref.shape != cand.shape:
            per_set.append({"set": i, "match": False, "reason": "shape_mismatch",
                            "ref_shape": list(ref.shape), "cand_shape": list(cand.shape)})
            all_match = False
            continue
        match = bool(np.allclose(cand, ref, rtol=rtol, atol=atol))
        abs_err = np.abs(cand - ref)
        denom = np.maximum(np.abs(ref), 1e-12)
        rel_err = abs_err / denom
        per_set.append({
            "set": i,
            "match": match,
            "max_abs_err": float(abs_err.max()) if abs_err.size else 0.0,
            "max_rel_err": float(rel_err.max()) if rel_err.size else 0.0,
        })
        all_match = all_match and match

    verified = bool(all_match and enough)
    return {
        "verified": verified,
        "input_sets": n,
        "min_input_sets": min_input_sets,
        "enough_input_sets": enough,
        "rtol": rtol,
        "atol": atol,
        "max_abs_err": max((s.get("max_abs_err", 0.0) for s in per_set), default=0.0),
        "max_rel_err": max((s.get("max_rel_err", 0.0) for s in per_set), default=0.0),
        "per_set": per_set,
    }
