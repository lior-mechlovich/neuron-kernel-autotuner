"""neuron-kernel-autotuner: the orchestration layer over AWS neuron-agentic-development.

AWS ships the building blocks (write / profile / query / debug NKI kernels). This package
adds the closed auto-tuning loop they don't have: a hardened correctness gate, cross-variant
ranking by % of peak, cost framing, and an optimization memory.

The four pure modules here (parse, verify, diff, cost) have no hardware or LLM dependency
and are unit-tested with fixtures. The on-instance capture/compile/generate steps live in
scripts/ and compose AWS's skills.
"""

__version__ = "0.1.0"
