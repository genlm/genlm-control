# Vendored verbatim from llamppl.inference.resampling (llamppl>=0.2.2).
#
# Provenance: this is a verbatim copy of llamppl's pure-numpy resampling
# functions, vendored so that genlm-control can drop the llamppl runtime
# dependency while guaranteeing bit-for-bit parity with the resampling RNG
# behavior of the original `smc_standard` path. Do not "improve" these
# functions; the consumed-RNG order must match llamppl exactly.
import numpy as np


def systematic_resample(weights):
    """Systematic resampling with a single random offset.

    Generates N equally spaced probe points in [0, 1] with a single
    random offset between 0 and 1/N. Indices are derived by mapping
    these points through the inverse CDF of the categorical distribution
    defined by the weights. Each index i is resampled exactly
    floor(N * w_i) or ceil(N * w_i) times.

    Unlike stratified and residual resampling, systematic resampling
    is not provably lower-variance than multinomial in all cases;
    see Douc et al. (2005), Sec. 3.4: https://arxiv.org/abs/cs/0507025

    Adapted from FilterPy (R. Labbe):
    https://filterpy.readthedocs.io/en/latest/monte_carlo/resampling.html

    Args:
        weights (array-like): Normalized probability weights summing to 1.

    Returns:
        ndarray: Integer array of ancestor indices.
    """
    N = len(weights)
    positions = (np.random.random() + np.arange(N)) / N

    indexes = np.zeros(N, "i")
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0  # avoid round-off errors
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def stratified_resample(weights):
    """Stratified resampling with one random draw per stratum.

    Divides [0, 1] into N equal strata and draws one uniform point
    independently within each, so that consecutive points are between
    0 and 2/N apart. Indices are derived by mapping these points
    through the inverse CDF of the categorical distribution defined
    by the weights.

    Adapted from FilterPy (R. Labbe):
    https://filterpy.readthedocs.io/en/latest/monte_carlo/resampling.html

    Args:
        weights (array-like): Normalized probability weights summing to 1.

    Returns:
        ndarray: Integer array of ancestor indices.
    """
    N = len(weights)
    positions = (np.random.random(N) + np.arange(N)) / N

    indexes = np.zeros(N, "i")
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.0
    i, j = 0, 0
    while i < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes


def residual_resample(weights):
    """Residual resampling: deterministic floor copies + multinomial remainder.

    Takes floor(N * w_i) copies of each particle deterministically,
    then resamples the remaining slots from the fractional residuals
    using multinomial resampling.

    Adapted from FilterPy (R. Labbe):
    https://filterpy.readthedocs.io/en/latest/monte_carlo/resampling.html

    Args:
        weights (array-like): Normalized probability weights summing to 1.

    Returns:
        ndarray: Integer array of ancestor indices.
    """
    N = len(weights)
    weights = np.asarray(weights, dtype=float)
    indexes = np.zeros(N, "i")

    # Deterministic copies
    num_copies = np.floor(N * weights).astype(int)
    k = 0
    for i in range(N):
        for _ in range(num_copies[i]):
            indexes[k] = i
            k += 1

    # Multinomial resample on the residual
    if k < N:
        residual = weights * N - num_copies
        residual /= residual.sum()
        cumulative_sum = np.cumsum(residual)
        cumulative_sum[-1] = 1.0
        indexes[k:N] = np.searchsorted(cumulative_sum, np.random.random(N - k))

    return indexes


def multinomial_resample(weights):
    """Multinomial resampling: independent categorical draws.

    Each of the N ancestor indices is drawn independently from the
    categorical distribution defined by the weights.

    Args:
        weights (array-like): Normalized probability weights summing to 1.

    Returns:
        ndarray: Integer array of ancestor indices.
    """
    N = len(weights)
    return np.random.choice(N, size=N, replace=True, p=weights)


RESAMPLING_METHODS = {
    "systematic": systematic_resample,
    "stratified": stratified_resample,
    "residual": residual_resample,
    "multinomial": multinomial_resample,
}


def get_resampling_fn(method):
    """Get a resampling function by name.

    Args:
        method (str): One of 'systematic', 'stratified', 'residual', 'multinomial'.

    Returns:
        callable: Resampling function that takes weights and returns indices.

    Raises:
        ValueError: If method is not recognized.
    """
    if method not in RESAMPLING_METHODS:
        raise ValueError(
            f"Unknown resampling method '{method}'. "
            f"Must be one of: {', '.join(RESAMPLING_METHODS.keys())}"
        )
    return RESAMPLING_METHODS[method]
