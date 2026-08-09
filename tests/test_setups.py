import pytest
import numpy as np
from genlm.control import SMC
from genlm.control.potential import Potential, PromptedLLM, BoolFSA
from genlm.control.sampler import (
    AWRS,
    direct_token_sampler,
    eager_token_sampler,
    topk_token_sampler,
)
from genlm.control.sampler.token import TokenSampler
from unittest.mock import Mock


@pytest.fixture(scope="module")
def llm():
    return PromptedLLM.from_name("gpt2", backend="hf", temperature=0.5)


@pytest.fixture(scope="module")
def best_fsa():
    return BoolFSA.from_regex(r"\sthe\s(best|greatest).+")


async def assert_engine_run(engine, n_particles, max_tokens, ess_threshold, **kwargs):
    sequences = await engine(
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        max_tokens=max_tokens,
        **kwargs,
    )

    assert len(sequences) == n_particles
    assert all(len(seq) <= max_tokens for seq in sequences)

    print(sequences)

    return sequences


@pytest.mark.asyncio
async def test_with_llm(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = direct_token_sampler(mtl_llm)
    engine = SMC(sampler)

    sequences = await assert_engine_run(
        engine, n_particles=10, max_tokens=25, ess_threshold=0.5
    )

    assert all(b"." not in seq for seq, _ in sequences)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_product_llm(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    sampler = direct_token_sampler(mtl_llm * nyc_llm)
    engine = SMC(sampler)

    await assert_engine_run(
        engine, n_particles=10, max_tokens=25, ess_threshold=0.5, verbosity=1
    )

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_critic(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    sampler = direct_token_sampler(mtl_llm)
    engine = SMC(sampler, critic=nyc_llm)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_critic_no_twist(llm):
    # When the ess_threshold is 0, the critic is only applied at the end of the generation.
    # This is to avoid running the critic at each step for IS.
    # We test that the critic is applied the correct number of times.

    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    n_calls = 0

    class MockCritic(Potential):
        async def prefix(self, context):
            nonlocal n_calls
            n_calls += 1
            return 0

        async def complete(self, context):
            nonlocal n_calls
            n_calls += 1
            return 0

    sampler = direct_token_sampler(mtl_llm)
    engine = SMC(sampler, critic=MockCritic(mtl_llm.vocab))

    n_particles = 10

    await assert_engine_run(engine, n_particles, max_tokens=5, ess_threshold=0)

    assert n_calls == n_particles

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_critic_early_stop(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    n_calls = 0
    n_particles = 10

    class MockSampler(TokenSampler):
        async def sample(self, context):
            nonlocal n_calls
            n_calls += 1
            return b"a", float("-inf"), np.nan

    class MockPotential(Potential):
        async def prefix(self, context):
            return 0

        async def complete(self, context):
            return 0

    sampler = MockSampler(mtl_llm)
    engine = SMC(sampler, critic=MockPotential(mtl_llm.vocab))

    await assert_engine_run(engine, n_particles, max_tokens=5, ess_threshold=0)

    assert n_calls == n_particles

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_no_critic_early_stop(llm):
    mtl_llm = llm.spawn_new_eos([b"."])
    n_calls = 0
    n_particles = 10

    class MockSampler(TokenSampler):
        async def sample(self, context):
            nonlocal n_calls
            n_calls += 1
            return b"a", float("-inf"), np.nan

    sampler = MockSampler(mtl_llm)
    engine = SMC(sampler)

    await assert_engine_run(engine, n_particles, max_tokens=5, ess_threshold=0)

    assert n_calls == n_particles

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_fsa(llm, best_fsa):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = direct_token_sampler(mtl_llm)

    best_fsa = best_fsa.coerce(mtl_llm, f=b"".join)

    engine = SMC(sampler, critic=best_fsa)
    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")
    engine = SMC(sampler, critic=best_fsa * nyc_llm)
    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_sampler",
    [
        lambda llm, fsa: eager_token_sampler(llm, fsa),
        lambda llm, fsa: topk_token_sampler(llm, fsa, K=10),
        lambda llm, fsa: AWRS(llm, fsa.coerce(llm, f=b"".join)),
    ],
    ids=["eager", "topk", "awrs"],
)
async def test_with_llm_and_fsa_sampler(llm, best_fsa, make_sampler):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = make_sampler(mtl_llm, best_fsa)
    engine = SMC(sampler)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    engine = SMC(sampler, critic=nyc_llm)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


def test_invalids(llm, best_fsa):
    with pytest.raises(ValueError):
        SMC(llm)

    sampler = direct_token_sampler(llm)

    with pytest.raises(ValueError, match="must be a Potential"):
        SMC(sampler, critic="not a potential")

    with pytest.raises(ValueError):
        # Fail to coerce beforehand.
        SMC(sampler, critic=best_fsa)


def test_invalid_critic():
    # Create a mock TokenSampler
    mock_sampler = Mock(spec=TokenSampler)

    # Try to create SMC with an invalid critic (just a string)
    with pytest.raises(ValueError, match="`critic` must be a Potential"):
        SMC(unit_sampler=mock_sampler, critic="not a potential")
