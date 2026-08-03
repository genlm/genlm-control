"""``Twist(contrast=..., temperature=...)``: the twist SMC actually holds."""

import pytest

from genlm.control.potential import Potential
from genlm.control.sampler.smc import Controller, Twist
from genlm.control.sampler.token import DirectTokenSampler


class _Flat(Potential):
    async def prefix(self, context):
        return 0.0

    async def complete(self, context):
        return 0.0


def _controller(**kwargs):
    p = _Flat([b"a", b"b"])
    return Controller(
        samplers=[DirectTokenSampler(p)],
        critics=[p],
        group_sizes=[2],
        ess_threshold=0.5,
        max_tokens=10,
        twist_with_critic=True,
        **kwargs,
    )


@pytest.mark.parametrize(
    "twist, expected",
    [
        (None, -1.5),  # bare critic score
        (Twist(contrast=True), 5.5),  # log-ratio against the proposal
        (Twist(contrast=True, temperature=0.25), 0.25 * 5.5),  # beta scales the ratio
        (Twist(contrast=True, temperature=0.0), 0.0),  # inert
    ],
)
def test_twist_value(twist, expected):
    c = _controller(twist=twist)
    p = c.particles[0]
    p.logp = -7.0
    assert c.twist.value(p, -1.5) == pytest.approx(expected)


def test_clip_requires_contrast():
    with pytest.raises(AssertionError):
        Twist(clip=(1.0, 1.0))
