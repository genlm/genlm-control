import numpy as np
from typing import List
from genlm.control.sampler.sequence import Sequences
from genlm.control.potential.base import Potential


def kl_divergence_direct(
    model_p, model_q, samples: List[str], batch_size: int = 32
) -> float:
    """Compute KL divergence directly using model log probabilities.

    This is the clean, efficient approach: KL(P||Q) = E_P[log P(x) - log Q(x)]

    Args:
        model_p: Model P with .log_prob(x) method
        model_q: Model Q with .log_prob(x) method
        samples (List[str]): Samples to evaluate (typically from model P)
        batch_size (int): Batch size for evaluation

    Returns:
        float: KL divergence estimate

    Example:
        >>> samples = model_p.sample(1000)
        >>> kl = kl_divergence_direct(model_p, model_q, samples)
    """
    log_ratios = []

    # Process in batches for memory efficiency
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]

        # Get log probabilities from both models
        logp = model_p.log_prob(batch)  # shape [batch_size]
        logq = model_q.log_prob(batch)  # shape [batch_size]

        # Compute log ratio for this batch
        log_ratio_batch = logp - logq
        log_ratios.extend(
            log_ratio_batch.cpu().numpy()
            if hasattr(log_ratio_batch, "cpu")
            else log_ratio_batch
        )

    # KL divergence is the expectation of log ratios
    return np.mean(log_ratios)


async def kl_divergence_potentials(
    potential_p: Potential, potential_q: Potential, samples: List[str]
) -> float:
    """Compute KL divergence using genlm-control Potential objects.

    Args:
        potential_p (Potential): First potential (P)
        potential_q (Potential): Second potential (Q)
        samples (List[str]): Samples to evaluate

    Returns:
        float: KL divergence estimate
    """
    log_ratios = []

    for sample in samples:
        # Skip empty samples
        if not sample or sample.strip() == "":
            print("Warning: Skipping empty sample")
            continue

        try:
            # Convert string to token sequence
            tokens = potential_p.tokenize(sample)

            # Skip if tokenization results in empty tokens
            if not tokens:
                print(f"Warning: Skipping sample with no tokens: '{sample}'")
                continue

            # Get log probabilities from both potentials
            logp = await potential_p.complete(tokens)
            logq = await potential_q.complete(tokens)

            log_ratios.append(logp - logq)

        except Exception as e:
            print(f"Warning: Error processing sample '{sample}': {e}")
            continue

    if not log_ratios:
        raise ValueError("No valid samples could be processed")

    return np.mean(log_ratios)


def kl_divergence_sequences(sequences1: Sequences, sequences2: Sequences) -> float:
    """Compute KL divergence between Sequences using their posteriors.

    Args:
        sequences1 (Sequences): First set of sequences (P)
        sequences2 (Sequences): Second set of sequences (Q)

    Returns:
        float: KL divergence D(P||Q)
    """
    # Get posterior distributions (already normalized)
    post1 = sequences1.decoded_posterior
    post2 = sequences2.decoded_posterior

    if not post1 or not post2:
        raise ValueError("No decodable sequences found")

    # Get union of all sequences
    all_sequences = set(post1.keys()) | set(post2.keys())

    kl_sum = 0.0
    for seq in all_sequences:
        p = post1.get(seq, 0.0)
        q = post2.get(seq, 0.0)

        # Only include sequences that appear in P
        if p > 0:
            # Add small smoothing to q if it's zero
            if q == 0:
                q = 1e-10

            kl_sum += p * np.log(p / q)

    return kl_sum


def effective_sample_size(sequences: Sequences) -> float:
    """Get the effective sample size from sequences."""
    return sequences.ess


def perplexity_from_kl(kl_div: float) -> float:
    """Convert KL divergence to perplexity."""
    return np.exp(kl_div)
