"""Internal helpers shared by token samplers."""
from genlm.control.potential.base import Potential


def _validate_proposal_vocab(target_potential, proposal):
    """Require `proposal.vocab_eos` to match `target_potential.vocab_eos`
    token-for-token. Cross-tokenizer proposals are not yet supported."""
    if not isinstance(proposal, Potential):
        raise TypeError(
            f"`proposal` must be a Potential; got {type(proposal).__name__}."
        )
    if proposal.vocab_eos != target_potential.vocab_eos:
        raise ValueError(
            "Proposal must share the target potential's `vocab_eos` "
            "(token-for-token); cross-tokenizer proposals are not yet supported. "
            f"Target has {len(target_potential.vocab_eos)} tokens; proposal has "
            f"{len(proposal.vocab_eos)}."
        )
