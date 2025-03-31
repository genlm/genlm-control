import pytest
import numpy as np
from genlm_control import InferenceEngine
from genlm_control.potential import Potential, PromptedLLM, BoolFSA
from genlm_control.sampler import (
    direct_token_sampler,
    eager_token_sampler,
    topk_token_sampler,
)
from genlm_control.sampler.token import TokenSampler


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
    engine = InferenceEngine(sampler)

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
    engine = InferenceEngine(sampler)

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
    engine = InferenceEngine(sampler, critic=nyc_llm)

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
            return 0

        async def complete(self, context):
            return 0

        async def score(self, context):
            nonlocal n_calls
            n_calls += 1
            return 0

    sampler = direct_token_sampler(mtl_llm)
    engine = InferenceEngine(sampler, critic=MockCritic(mtl_llm.vocab))

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
    engine = InferenceEngine(sampler, critic=MockPotential(mtl_llm.vocab))

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
    engine = InferenceEngine(sampler)

    await assert_engine_run(engine, n_particles, max_tokens=5, ess_threshold=0)

    assert n_calls == n_particles

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_fsa(llm, best_fsa):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = direct_token_sampler(mtl_llm)

    best_fsa = best_fsa.coerce(mtl_llm, f=b"".join)

    engine = InferenceEngine(sampler, critic=best_fsa)
    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")
    engine = InferenceEngine(sampler, critic=best_fsa * nyc_llm)
    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_fsa_eager_sampler(llm, best_fsa):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = eager_token_sampler(mtl_llm, best_fsa)
    engine = InferenceEngine(sampler)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    engine = InferenceEngine(sampler, critic=nyc_llm)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


@pytest.mark.asyncio
async def test_with_llm_and_fsa_topk_sampler(llm, best_fsa):
    mtl_llm = llm.spawn_new_eos([b"."])
    mtl_llm.set_prompt_from_str("Montreal is")

    sampler = topk_token_sampler(mtl_llm, best_fsa, K=10)
    engine = InferenceEngine(sampler)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    nyc_llm = mtl_llm.spawn()
    nyc_llm.set_prompt_from_str("NYC is")

    engine = InferenceEngine(sampler, critic=nyc_llm)

    await assert_engine_run(engine, n_particles=10, max_tokens=25, ess_threshold=0.5)

    await engine.cleanup()


def test_invalids(llm, best_fsa):
    with pytest.raises(ValueError):
        InferenceEngine(llm)

    sampler = direct_token_sampler(llm)

    with pytest.raises(ValueError):
        InferenceEngine(llm, critic=sampler)

    sampler = direct_token_sampler(llm)
    with pytest.raises(ValueError):
        # Fail to coerce beforehand.
        InferenceEngine(sampler, critic=best_fsa)
