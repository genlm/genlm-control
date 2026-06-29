"""Counter-based (Threefry-2x32) picker checks.

The point of the counter-based picker is twofold and both are tested here:
  1. DEVICE-AGNOSTIC: the bitstream depends only on the (seed, slot, step) key, not the
     device -> CPU and CUDA produce identical uniforms (the test that needs the box).
  2. CORRECT: keyed-deterministic + order-independent, the uniforms are U(0,1), and
     ``threefry_gumbel`` draws exactly proportional to softmax(logits) (runs on CPU).

Local (CPU) covers everything but the CPU==CUDA identity, which skips without CUDA.
"""

import numpy as np
import pytest
import torch

from genlm.control.util import (
    threefry_2x32,
    threefry_uniform,
    threefry_gumbel,
    draw_key,
    set_draw_seed,
)


def test_keyed_determinism():
    """Same key -> identical uniforms; any field change -> different."""
    a = threefry_uniform(64, seed=1234, slot=3, step=7, device="cpu")
    b = threefry_uniform(64, seed=1234, slot=3, step=7, device="cpu")
    assert torch.equal(a, b)
    for kw in (dict(seed=1235), dict(slot=4), dict(step=8)):
        base = dict(seed=1234, slot=3, step=7)
        base.update(kw)
        c = threefry_uniform(64, device="cpu", **base)
        assert not torch.equal(a, c), f"key change {kw} did not change the stream"


def test_batched_draw_matches_per_row():
    """The on-device batched draw is byte-identical to N scalar per-row draws: the same
    (slot, step) keys produce the same uniforms (broadcast == per-row) and thus the same
    Gumbel-max id. The same `threefry_gumbel` serves both -- a batched `draw_key` (per-row
    slot/step tensors) draws the whole [N, V] population at once. No batched twin."""
    set_draw_seed(1234)
    V, N = 257, 6
    slots = torch.tensor([3, 0, 9, 2, 5, 1], dtype=torch.int64)
    steps = torch.tensor([7, 1, 4, 0, 12, 3], dtype=torch.int64)

    # 1) batched uniforms [N, V] are bit-identical to N scalar [V] calls
    ub = threefry_uniform(V, 1234, slots, steps, "cpu")
    assert ub.shape == (N, V)
    for r in range(N):
        us = threefry_uniform(V, 1234, int(slots[r]), int(steps[r]), "cpu")
        assert torch.equal(ub[r], us), f"uniform row {r} not bit-identical"

    # 2) the batched draw (threefry_gumbel under a batched draw_key) matches per-row scalar
    torch.manual_seed(0)
    logps = torch.randn(N, V, dtype=torch.float64)
    with draw_key(slots, steps):
        ids = threefry_gumbel(logps)
    for r in range(N):
        with draw_key(int(slots[r]), int(steps[r])):
            assert int(threefry_gumbel(logps[r])) == int(ids[r]), f"draw row {r} mismatch"


def test_uniformity():
    """threefry_uniform output passes a chi-square uniformity test over [0,1)."""
    # one uniform per (step) at a fixed seed/slot, many steps
    M = 200_000
    steps = torch.arange(M, dtype=torch.int64)
    x = threefry_2x32(torch.zeros_like(steps), steps, 1234, 0)
    u = (x.double() + 0.5) / 4294967296.0
    nb = 50
    counts = torch.histc(u.float(), bins=nb, min=0.0, max=1.0).numpy()
    expected = M / nb
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    # 50 bins -> 49 dof; chi2 0.999 quantile ~ 85. Comfortably below for a good hash.
    assert chi2 < 85.0, f"chi2={chi2:.1f} (49 dof) -- uniforms look non-uniform"


def test_gumbel_marginal_matches_softmax():
    """Vectorized Gumbel-max over counter noise draws proportional to softmax(logits)."""
    logits = torch.tensor([2.0, 1.0, 0.5, -1.0, 0.0], dtype=torch.float32)
    V = logits.numel()
    M = 300_000
    # row m uses counter (index i, step m); key (seed, slot)=(1234, 0)
    i = torch.arange(V, dtype=torch.int64)[None, :].expand(M, V)
    m = torch.arange(M, dtype=torch.int64)[:, None].expand(M, V)
    x = threefry_2x32(i, m, 1234, 0)
    u = (x >> 8).to(torch.float32) / 16777216.0  # 24-bit grid (matches threefry_uniform)
    g = -torch.log(-torch.log(u))
    draws = (logits + g).argmax(dim=-1)
    freq = torch.bincount(draws, minlength=V).double() / M
    want = torch.softmax(logits.double(), 0)
    err = (freq - want).abs().max().item()
    assert err < 0.005, f"empirical vs softmax max abs err {err:.4f}\n{freq}\nvs\n{want}"


def test_picker_keyed_and_fallback():
    """threefry_gumbel: deterministic under a scoped key, matches the explicit
    uniform->gumbel path, auto-advances the ordinal per draw within a scope; unkeyed it
    falls back to torch.rand (still a valid index)."""
    set_draw_seed(1234)
    logps = torch.log_softmax(torch.randn(100, dtype=torch.float32), 0)

    def pick(seed, slot, step):
        u = threefry_uniform(logps.shape[-1], seed, slot, step, logps.device)
        g = (-torch.log(-torch.log(u))).to(logps.dtype)
        return int((logps + g).argmax())

    # keyed -> deterministic across separate scopes at the same (slot, base)
    with draw_key(slot=2, base=5):
        t1 = int(threefry_gumbel(logps))
    with draw_key(slot=2, base=5):
        t2 = int(threefry_gumbel(logps))
    assert t1 == t2 == pick(1234, 2, 5)

    # within ONE scope the ordinal auto-advances: draw j uses step base+j
    with draw_key(slot=7, base=0):
        seq = [int(threefry_gumbel(logps)) for _ in range(4)]
    assert seq == [pick(1234, 7, j) for j in range(4)]

    # unkeyed -> torch.rand fallback still returns a valid index
    tok = int(threefry_gumbel(logps))
    assert 0 <= tok < 100


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPU==CUDA identity needs CUDA")
def test_cpu_cuda_bit_identical():
    """THE device-agnostic guarantee: identical (key, counter) -> bit-identical uniforms
    on CPU and CUDA. (The downstream log/argmax may differ by a transcendental ULP, far
    below the warm-KV residual; the RNG stream itself is exact.)"""
    for (seed, slot, step) in [(1234, 0, 0), (7, 3, 11), (2**31 - 1, 5, 99)]:
        cpu = threefry_uniform(257, seed, slot, step, "cpu")
        gpu = threefry_uniform(257, seed, slot, step, "cuda").cpu()
        assert torch.equal(cpu, gpu), f"CPU != CUDA at (seed={seed}, slot={slot}, step={step})"
    # raw integer words exactly equal too
    i = torch.arange(257, dtype=torch.int64)
    xc = threefry_2x32(i, torch.full_like(i, 99), 7, 3)
    xg = threefry_2x32(i.cuda(), torch.full_like(i, 99).cuda(), 7, 3).cpu()
    assert torch.equal(xc, xg)


if __name__ == "__main__":
    test_keyed_determinism()
    test_uniformity()
    test_gumbel_marginal_matches_softmax()
    test_picker_keyed_and_fallback()
    print("CPU checks passed.")
    if torch.cuda.is_available():
        test_cpu_cuda_bit_identical()
        print("CPU==CUDA bit-identity passed.")
    else:
        print("(skipped CPU==CUDA: no CUDA here)")
