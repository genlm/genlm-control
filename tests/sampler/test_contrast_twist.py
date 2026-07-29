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


def test_default_twist_is_the_bare_score():
    c = _controller()
    p = c.particles[0]
    p.logp = -7.0
    assert c.twist.value(p, -1.5) == pytest.approx(-1.5)


def test_contrast_subtracts_the_proposal_logp():
    c = _controller(twist=Twist(contrast=True))
    p = c.particles[0]
    p.logp = -7.0
    assert c.twist.value(p, -1.5) == pytest.approx(5.5)


def test_temperature_scales_the_contrasted_value():
    """beta multiplies the whole log-ratio, not just the critic's side."""
    c = _controller(twist=Twist(contrast=True, temperature=0.25))
    p = c.particles[0]
    p.logp = -7.0
    assert c.twist.value(p, -1.5) == pytest.approx(0.25 * 5.5)


def test_zero_temperature_is_inert():
    c = _controller(twist=Twist(contrast=True, temperature=0.0))
    p = c.particles[0]
    p.logp = -7.0
    assert c.twist.value(p, -1.5) == 0.0


def test_clip_requires_contrast():
    with pytest.raises(AssertionError):
        Twist(clip=(1.0, 1.0))
