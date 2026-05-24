"""Shared sampler/critic definitions and the parity matrix.

Imported by both the snapshot generator (``_gen_parity_snapshot.py``) and the
parity gate (``test_per_token_parity.py``) so the reference and the hub run on
identical inputs.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from genlm.control.constant import EOS  # noqa: E402
from genlm.control.sampler.token import DirectTokenSampler  # noqa: E402
from genlm.control.sampler.unit import (  # noqa: E402
    MultiTokenUnitSampler,
    TokenSetBoundary,
)

from conftest import MockPotential, WeightedSet  # noqa: E402

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


SAMPLER_BUILDERS = {
    "weighted_set_direct": _direct_sampler,
    "mock_direct": _mock_direct_sampler,
    "multitoken": _multitoken_sampler,
}
