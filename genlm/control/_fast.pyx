"""Cython kernel for ``logsumexp``: max-subtract trick in a tight C loop.

Wins ~5-9x over arsenal's numpy implementation on small (V<=128) arrays
where SMC spends most of its logsumexp time (resampling N particles,
per-token critic mixtures); negligible at full-vocab sizes.

Inputs are coerced to ``float64`` once at the Python/Cython boundary.
"""

cimport numpy as cnp
import numpy as np
from libc.math cimport exp, log, INFINITY

cnp.import_array()


def logsumexp(arr):
    """log(sum(exp(arr))) using the max-subtract trick for stability."""
    cdef cnp.ndarray[double, ndim=1, mode="c"] a = np.ascontiguousarray(arr, dtype=np.float64)
    return _logsumexp(a)


cdef double _logsumexp(double[::1] a) noexcept nogil:
    cdef Py_ssize_t n = a.shape[0]
    cdef Py_ssize_t i
    cdef double vmax, s

    if n == 0:
        return -INFINITY

    vmax = a[0]
    for i in range(1, n):
        if a[i] > vmax:
            vmax = a[i]

    if vmax == -INFINITY:
        return -INFINITY

    s = 0.0
    for i in range(n):
        s += exp(a[i] - vmax)
    return log(s) + vmax
