from .token import DirectTokenSampler, SetTokenSampler, AWRS, TokenSampler
from .set import EagerSetSampler, TopKSetSampler
from .sequence import SMC, SequenceModel
from .unit import (
    UnitSampler,
    MultiTokenUnitSampler,
    BoundaryPredicate,
    TokenSetBoundary,
    FixedLengthBoundary,
    boundary_token_set,
    boundary_fixed_length,
)
from genlm.control.potential import Potential


def direct_token_sampler(potential):
    """Create a `DirectTokenSampler` that samples directly from a potential's vocabulary.

    See `DirectTokenSampler` for more details.

    Args:
        potential (Potential): The potential function to sample from. Should have an efficient logw_next method.

    Returns:
        (DirectTokenSampler): A sampler that directly samples tokens from the potential's vocabulary.
    """
    assert isinstance(potential, Potential)
    return DirectTokenSampler(potential)


def eager_token_sampler(iter_potential, item_potential):
    """Create a `SetTokenSampler` that uses the `EagerSetSampler` to sample a set of tokens.

    See `EagerSetSampler` for more details.

    Args:
        iter_potential (Potential): A potential function defined over a vocabulary of iterables.
        item_potential (Potential): A potential function defined over a vocabulary of items which are elements of the iterables.

    Returns:
        (SetTokenSampler): A sampler that wraps an `EagerSetSampler`.

    Note:
        This is the fastest sampler in most cases.
    """
    return SetTokenSampler(EagerSetSampler(iter_potential, item_potential))


def topk_token_sampler(iter_potential, item_potential, K):
    """Create a `SetTokenSampler` that uses the `TopKSetSampler` to sample a set of tokens.

    See `TopKSetSampler` for more details.

    Args:
        iter_potential (Potential): A potential function defined over a vocabulary of iterables.
        item_potential (Potential): A potential function defined over a vocabulary of items which are elements of the iterables.
        K (int|None): The `K` parameter for the `TopKSetSampler`.

    Returns:
        (SetTokenSampler): A sampler that wraps an `TopKSetSampler`.
    """
    return SetTokenSampler(TopKSetSampler(iter_potential, item_potential, K))


__all__ = [
    "AWRS",
    "direct_token_sampler",
    "eager_token_sampler",
    "topk_token_sampler",
    "DirectTokenSampler",
    "EagerSetSampler",
    "TopKSetSampler",
    "SetTokenSampler",
    "TokenSampler",
    "Importance",
    "SMC",
    "SequenceModel",
    "UnitSampler",
    "MultiTokenUnitSampler",
    "BoundaryPredicate",
    "TokenSetBoundary",
    "FixedLengthBoundary",
    "boundary_token_set",
    "boundary_fixed_length",
]
