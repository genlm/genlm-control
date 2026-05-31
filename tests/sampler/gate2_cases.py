"""Shared gate-2 cases: ONE definition of the engine-native no-bias configs, imported by
both the gate (``test_engine_native.py``) and the original-reference generator
(``gen_original_reference.py``). Mirrors gate-1's ``parity_cases.py``.

Killing the test<->generator duplication matters most for the CRITICS: a critic edited in
the test but not the generator would make the cached reference (computed with the old
critic) silently mismatch -- a false comparison, not a loud error. Seeds/N/ess drift is
loud (a missing ``_ref`` key raises), but shared here too.

Engine creation is deferred: factories take ``llm`` (a ``PromptedLLM``); nothing here
imports ``genlm.backend``/vLLM at module load, so gate-1 collection on the mac is unaffected
and the generator can import this under main's ``genlm.control`` (PYTHONPATH=main).
"""

from dataclasses import dataclass
from typing import Callable, Optional

from genlm.control.constant import EndOfSequence
from genlm.control.potential import Potential
from genlm.control.potential.built_in.wfsa import BoolFSA
from genlm.control.sampler.token import AWRS, DirectTokenSampler, SetTokenSampler
from genlm.control.sampler.unit import BoundaryPredicate, MultiTokenUnitSampler
from genlm.control.sampler import EagerSetSampler

MODEL = "gpt2"
PROMPT = "The"
EOS_BYTES = [b"\n"]
CONFIG = {"model": MODEL, "prompt": PROMPT, "eos_hex": [b.hex() for b in EOS_BYTES]}


# --- critics / boundary (were byte-identical dupes in the test and the generator) ------


class TerminalContainsCritic(Potential):
    """Terminal 0/-inf indicator (the genlm-latent CoTCritic shape): the completed text
    must contain a space. ``prefix`` is 0 throughout; ``score`` is the indicator on any
    termination."""

    def __init__(self, vocab):
        super().__init__(vocabulary=vocab)

    async def _indicator(self, context):
        bs = [t for t in context if not isinstance(t, EndOfSequence)]
        try:
            text = b"".join(bs).decode("utf-8")
        except UnicodeDecodeError:
            return float("-inf")
        return 0.0 if " " in text else float("-inf")

    async def complete(self, context):
        return await self._indicator(context)

    async def prefix(self, context):
        return 0.0

    async def score(self, context):
        return await self._indicator(context)


class SoftVowelCritic(Potential):
    """Soft, content-dependent critic: -0.5 per vowel in the decoded text. Finite and
    non-trivial (a per-step twist accumulates real weight) and content-dependent (weights
    diverge -> ESS drops -> resample). Penalty deliberately MODERATE: stronger collapses
    ESS to ~1 effective particle every step, maximizing variance without testing more."""

    def __init__(self, vocab):
        super().__init__(vocabulary=vocab)

    def _pen(self, context):
        bs = [t for t in context if not isinstance(t, EndOfSequence)]
        try:
            text = b"".join(bs).decode("utf-8")
        except UnicodeDecodeError:
            return float("-inf")
        return -0.5 * sum(c in "aeiouAEIOU" for c in text)

    async def complete(self, context):
        return self._pen(context)

    async def prefix(self, context):
        return self._pen(context)


class ByteLengthBoundary(BoundaryPredicate):
    """A unit completes once its subunits span >= ``min_bytes`` bytes. Content-dependent +
    variable-length -> rows reach the boundary at different engine steps -> the staggered
    per-unit pop-out the burst must sync."""

    def __init__(self, min_bytes):
        self.min_bytes = min_bytes

    def __call__(self, unit_context, subunit_buffer):
        return (
            sum(len(t) for t in subunit_buffer if not isinstance(t, EndOfSequence))
            >= self.min_bytes
        )


# --- config table -----------------------------------------------------------------------
#
# ``make_sampler(llm, seed)`` is two-arg so AWRS gets its per-seed rng (every other factory
# ignores ``seed``); fresh per call (the async-trie Set sampler must bind to the run's loop).
# ``reference`` is data: "ref" => the generator emits a cached original key the test loads
# via ``_ref``; "steploop" => the test compares against a live StepLoop (no cached ref).


@dataclass(frozen=True)
class Case:
    label: str
    n_particles: int
    ess: float
    max_tokens: int
    seeds: tuple
    reference: str  # "ref" | "steploop"
    make_sampler: Callable  # (llm, seed) -> TokenSampler
    make_critic: Optional[Callable] = None  # (llm) -> Potential | None

    def sampler(self, llm, seed):
        return self.make_sampler(llm, seed)

    def critic(self, llm):
        return self.make_critic(llm) if self.make_critic is not None else None


# Coerced BoolFSAs are reused across seeds (the coerce over the 50k vocab is the expensive
# part); the sampler objects wrapping them are still minted fresh per call.
_COERCE_MEMO = {}


def boolfsa(llm, regex):
    key = (id(llm), regex)
    if key not in _COERCE_MEMO:
        _COERCE_MEMO[key] = llm * BoolFSA.from_regex(regex).coerce(llm, f=b"".join)
    return _COERCE_MEMO[key]


def _condition(llm, regex):
    key = ("cond", id(llm), regex)
    if key not in _COERCE_MEMO:
        _COERCE_MEMO[key] = BoolFSA.from_regex(regex).coerce(llm, f=b"".join)
    return _COERCE_MEMO[key]


S6 = (1234, 7, 99, 2024, 555, 31)
S12 = (1234, 7, 99, 2024, 555, 31, 8, 17, 42, 123, 271, 314)
S_AEIOU = (1234, 7, 99, 2024, 555, 31, 808, 42, 17, 6, 71, 900)

CASES = {
    c.label: c
    for c in [
        Case("unconstrained", 8, 0.0, 12, (1234,), "ref",
             lambda llm, seed: DirectTokenSampler(llm)),
        Case("constrained-boolfsa[a-z ]+", 16, 0.0, 12, (1234, 7), "steploop",
             lambda llm, seed: DirectTokenSampler(boolfsa(llm, r"[a-z ]+"))),
        Case("boolfsa[aeiou ]+", 16, 0.5, 10, S_AEIOU, "ref",
             lambda llm, seed: DirectTokenSampler(boolfsa(llm, r"[aeiou ]+"))),
        Case("terminal-critic", 16, 0.0, 12, S6, "ref",
             lambda llm, seed: DirectTokenSampler(llm),
             lambda llm: TerminalContainsCritic(llm.vocab)),
        Case("twist-critic", 16, 0.5, 12, S12, "ref",
             lambda llm, seed: DirectTokenSampler(llm),
             lambda llm: SoftVowelCritic(llm.vocab)),
        Case("multitoken-boolfsa[a-z ]+", 8, 0.5, 6, S12, "ref",
             lambda llm, seed: MultiTokenUnitSampler(
                 DirectTokenSampler(boolfsa(llm, r"[a-z ]+")),
                 ByteLengthBoundary(5), max_subunits_per_unit=6)),
        Case("awrs[a-z ]+", 16, 0.0, 12, S6, "ref",
             lambda llm, seed: AWRS(llm, _condition(llm, r"[a-z ]+"), seed=seed)),
        Case("set[a-z ]+", 8, 0.0, 8, (1234, 7, 99, 2024), "ref",
             lambda llm, seed: SetTokenSampler(
                 EagerSetSampler(iter_potential=llm,
                                 item_potential=BoolFSA.from_regex(r"[a-z ]+")))),
    ]
}
