"""Pinned-output tripwires for the SMC loop and the vendored resampling functions.

These pins freeze the CURRENT behavior of `smc_standard` (a full seeded run over a
deterministic mock potential) and of `resampling.py` (seeded index draws). They assert
nothing about correctness -- distributional correctness lives in the other test files.
Their job is to fail loudly when a change alters the algorithm's draws or weights, so
the change is made knowingly ("never change the SMC algorithm" is the bar).

An intentional change regenerates the pins: run this file as a script and paste the
printed blocks over the constants below.

    .venv/bin/python tests/sampler/test_snapshot_pin.py
"""

import asyncio

import numpy as np
import pytest
import torch

from genlm.control.constant import EOS
from genlm.control.potential import Potential
from genlm.control.sampler import resampling as R
from genlm.control.sampler.smc import SequenceModel, smc_standard
from genlm.control.sampler.token import DirectTokenSampler

# --------------------------------------------------------------------------- #
# Resampling pins: fn(weights) under np.random.seed(1234)                      #
# --------------------------------------------------------------------------- #

_GRIDS = {
    "peaked4": [0.97, 0.01, 0.01, 0.01],
    "uniform8": [1 / 8] * 8,
    "ramp5": [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15],
}

RESAMPLING_PINS = {
    ("peaked4", "multinomial"): [0, 0, 0, 0],
    ("peaked4", "stratified"): [0, 0, 0, 0],
    ("peaked4", "systematic"): [0, 0, 0, 0],
    ("peaked4", "residual"): [0, 0, 0, 0],
    ("uniform8", "multinomial"): [1, 4, 3, 6, 6, 2, 2, 6],
    ("uniform8", "stratified"): [0, 1, 2, 3, 4, 5, 6, 7],
    ("uniform8", "systematic"): [0, 1, 2, 3, 4, 5, 6, 7],
    ("uniform8", "residual"): [0, 1, 2, 3, 4, 5, 6, 7],
    ("ramp5", "multinomial"): [1, 3, 3, 4, 4],
    ("ramp5", "stratified"): [0, 2, 3, 4, 4],
    ("ramp5", "systematic"): [0, 2, 3, 3, 4],
    ("ramp5", "residual"): [2, 3, 4, 1, 3],
}


def _resampling_indices():
    out = {}
    for gname, w in _GRIDS.items():
        for method in ("multinomial", "stratified", "systematic", "residual"):
            np.random.seed(1234)
            idx = R.get_resampling_fn(method)(np.asarray(w))
            out[(gname, method)] = [int(i) for i in idx]
    return out


@pytest.mark.parametrize("key", sorted(RESAMPLING_PINS))
def test_resampling_pinned(key):
    gname, method = key
    np.random.seed(1234)
    idx = R.get_resampling_fn(method)(np.asarray(_GRIDS[gname]))
    assert [int(i) for i in idx] == RESAMPLING_PINS[key]


# --------------------------------------------------------------------------- #
# SMC loop pin: one seeded smc_standard run over a deterministic potential     #
# --------------------------------------------------------------------------- #


class _Ramp(Potential):
    """Deterministic toy: next-token weight rises with position; EOS overtakes."""

    def __init__(self):
        super().__init__([b"a", b"b", b"c"])

    async def complete(self, context):
        return -0.1 * len(context)

    async def prefix(self, context):
        return -0.5 * len(context)

    async def logw_next(self, context):
        n = len(context)
        row = [-1.0 - 0.1 * n, -2.0 + 0.2 * n, -3.0, -4.0 + 1.2 * n]  # a, b, c, EOS
        return self.make_lazy_weights(np.asarray(row))


def _run_pinned_smc():
    torch.manual_seed(7)
    np.random.seed(7)
    sampler = DirectTokenSampler(_Ramp(), autobatch=False)
    model = SequenceModel(
        unit_sampler=sampler,
        critic=None,
        max_tokens=5,
        twist_with_critic=False,
        terminate_when=None,
        verbosity=0,
    )
    particles = asyncio.run(
        smc_standard(model=model, n_particles=4, ess_threshold=0.5)
    )
    contexts = [
        [t if isinstance(t, bytes) else "EOS" for t in p.context] for p in particles
    ]
    weights = [float(p.weight) for p in particles]
    return contexts, weights


SMC_PIN_CONTEXTS = [
    [b"c", "EOS"],
    [b"a", b"a", b"a", "EOS"],
    [b"a", b"a", "EOS"],
    [b"a", b"b", b"b", b"b", "EOS"],
]
SMC_PIN_WEIGHTS = [
    -1.0561298112849253,
    -1.1229713476429621,
    -1.3374663045516892,
    -0.3229713476429623,
]


def test_smc_loop_pinned():
    contexts, weights = _run_pinned_smc()
    assert contexts == SMC_PIN_CONTEXTS
    np.testing.assert_allclose(weights, SMC_PIN_WEIGHTS, rtol=1e-10)
    for ctx in contexts:
        assert ctx[-1] == "EOS" or ctx[-1] is EOS


if __name__ == "__main__":
    print("RESAMPLING_PINS = {")
    for key, idx in sorted(_resampling_indices().items()):
        print(f"    {key!r}: {idx},")
    print("}")
    contexts, weights = _run_pinned_smc()
    print(f"SMC_PIN_CONTEXTS = {contexts!r}")
    print(f"SMC_PIN_WEIGHTS = {weights!r}")
