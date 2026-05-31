"""Shared sampler/critic definitions and the parity matrix.

Imported by both the snapshot generator (``_gen_parity_snapshot.py``) and the
parity gate (``test_per_token_parity.py``) so the reference and the controller run on
identical inputs.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genlm.control.constant import EOS  # noqa: E402
from genlm.control.sampler.token import (  # noqa: E402
    AWRS,
    DirectTokenSampler,
    SetTokenSampler,
)
from genlm.control.sampler import EagerSetSampler  # noqa: E402
from genlm.control.sampler.unit import (  # noqa: E402
    MultiTokenUnitSampler,
    TokenSetBoundary,
)

from conftest import MockPotential, WeightedSet  # noqa: E402

# Serialization + seeding are the shared harness (`_harness.seed_all`/`ctx_repr`/`num`),
# imported directly by the gate and the generator -- one definition, no drift.

N_PARTICLES = 16
MAX_TOKENS = 6
SEED = 1234
MATRIX = [0.0, 0.5, 1.0]


class _FlatteningMockPotential(MockPotential):
    """A MockPotential critic that flattens nested unit contexts before scoring,
    as a real multi-token critic would via `flatten_units` coercion."""

    def _logw(self, context):
        from genlm.control.sampler.unit import flatten_units

        return sum(
            self.next_token_logws[self.lookup[i]] for i in flatten_units(context)
        )


def _direct_sampler():
    p = WeightedSet(["0", "00", "1", "11"], [3.0, 2.0, 1.0, 4.0])
    return DirectTokenSampler(p), WeightedSet(
        ["0", "00", "1", "11"], [1.0, 2.0, 3.0, 1.5]
    )


def _mock_direct_sampler():
    vocab = [b"a", b"b", b"c"]
    logws = np.log([0.4, 0.3, 0.2, 0.1])
    p = MockPotential(vocab, logws)
    crit_logws = np.log([0.2, 0.3, 0.4, 0.1])
    return DirectTokenSampler(p), MockPotential(vocab, crit_logws)


def _multitoken_sampler():
    vocab = [b"a", b"b", b" "]
    logws = np.log([0.3, 0.3, 0.3, 0.1])
    p = MockPotential(vocab, logws)
    sub = DirectTokenSampler(p)
    boundary = TokenSetBoundary({b" ", EOS})
    sampler = MultiTokenUnitSampler(sub, boundary, max_subunits_per_unit=8)
    crit_logws = np.log([0.2, 0.4, 0.3, 0.1])
    return sampler, _FlatteningMockPotential(vocab, crit_logws)


def _awrs_sampler():
    # AWRS rejection over a Mock target + a boolean Mock condition (-inf rejects). Its own
    # seeded rng makes the rejection draws reproducible across both drivers (byte-exact).
    vocab = [b"a", b"b", b"c"]
    target = MockPotential(vocab, np.log([0.4, 0.3, 0.2, 0.1]))
    condition = MockPotential(vocab, np.array([0.0, float("-inf"), 0.0, 0.0]))  # reject "b"
    crit = MockPotential(vocab, np.log([0.2, 0.3, 0.4, 0.1]))
    return AWRS(target, condition, seed=42), crit


def _set_sampler():
    # EagerSetSampler over a Mock iter potential (byte-string sequences) + a Mock item
    # potential (the ints those bytes iterate into). The set draw consumes the global rng,
    # so byte-exactness needs the same consumption order in both drivers (gate-1 verifies).
    iter_vocab = [b"ab", b"cd", b"a"]
    item_vocab = list(dict.fromkeys(x for seq in iter_vocab for x in seq))  # ints
    mock_iter = MockPotential(iter_vocab, np.log([0.5, 0.3, 0.2, 0.1]))
    mock_item = MockPotential(item_vocab, np.log([0.3, 0.3, 0.2, 0.1, 0.1]))
    sampler = SetTokenSampler(
        EagerSetSampler(iter_potential=mock_iter, item_potential=mock_item)
    )
    return sampler, MockPotential(iter_vocab, np.log([0.25, 0.25, 0.25, 0.25]))


SAMPLER_BUILDERS = {
    "weighted_set_direct": _direct_sampler,
    "mock_direct": _mock_direct_sampler,
    "multitoken": _multitoken_sampler,
    "awrs": _awrs_sampler,
    "set": _set_sampler,
}

# Per-sampler ess set (default = full MATRIX). AWRS draws from its OWN seeded rng (never
# the global numpy stream), so its draw is byte-exact vs the original at ess=0 -- but with
# resampling (ess>0) the global-rng stream the resampler consumes is offset relative to the
# original (which advanced it during a global-rng draw), so the resample permutation
# diverges. That divergence is RNG-order, not bias (gate-2 covers no-bias), and the
# resample path itself is byte-pinned by the global-rng samplers (direct/multitoken). The
# Set draw likewise only needs its ess=0 pin to anchor gate-2's set case. So AWRS/Set are
# byte-pinned at ess=0 only -- exactly the regime gate-2 uses them in.
SAMPLER_ESS = {"awrs": [0.0], "set": [0.0]}


def matrix_combos():
    """The (sampler_name, use_critic, ess) cases -- shared by the gate and the generator so
    they enumerate identically. ess is per-sampler (SAMPLER_ESS), default the full MATRIX."""
    for sampler_name in SAMPLER_BUILDERS:
        for use_critic in (False, True):
            for ess in SAMPLER_ESS.get(sampler_name, MATRIX):
                yield sampler_name, use_critic, ess
