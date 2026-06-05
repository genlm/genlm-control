import contextlib
import contextvars
import warnings

import numpy as np
import torch
from genlm.grammar import Float, Log

from genlm.control.constant import EndOfSequence
from genlm.backend.tokenization import Token


def logsumexp(x):
    """Numpy log-sum-exp over a 1-D array; returns ``-inf`` (not ``nan``) on all-(-inf)
    input. CPU path; on-device weights use ``torch.logsumexp``."""
    x = np.asarray(x)
    if np.all(x == -np.inf):
        return -np.inf
    m = np.max(x)
    return np.log(np.sum(np.exp(x - m))) + m


def to_numpy(w):
    """Coerce a weight array to numpy regardless of backend (no-op on numpy)."""
    return w.cpu().numpy() if torch.is_tensor(w) else np.asarray(w)


def stack_weights(arrays):
    """Stack per-context weight arrays into one ``[N, V]`` batch, preserving the producer's
    backend (numpy stays numpy, torch stays torch) -- the batched-``LazyWeights`` weights."""
    return torch.stack(arrays) if torch.is_tensor(arrays[0]) else np.stack(arrays)


def _xp(w):
    """The array module (``torch`` or ``np``) backing ``w`` -- for backend-agnostic ops."""
    return torch if torch.is_tensor(w) else np


def _logsumexp(w):
    """Backend-dispatched log-sum-exp over a 1-D weight array."""
    return torch.logsumexp(w, 0) if torch.is_tensor(w) else logsumexp(w)


class LazyWeights:
    """
    A class to represent weights in a lazy manner, allowing for efficient operations
    on potentially large weight arrays without immediate materialization.

    Attributes:
        weights (np.ndarray): The weights associated with the tokens.
        encode (dict): A mapping from tokens to their corresponding indices in the weights array.
        decode (list): A list of tokens corresponding to the weights.
        is_log (bool): A flag indicating whether the weights are in log space.
    """

    def __init__(self, weights, encode, decode, log=True):
        """
        Initialize the LazyWeights instance.

        Args:
            weights (np.ndarray): The weights associated with the tokens.
            encode (dict): A mapping from tokens to their corresponding indices in the weights array.
            decode (list): A list of tokens corresponding to the weights.
            log (bool, optional): Indicates if the weights are in log space. Defaults to True.

        Raises:
            AssertionError: If the lengths of weights and decode do not match, or if encode has fewer entries than decode.
        """
        # ``weights`` keeps the producer's backend (LM->torch, grammar/FSA/trie->numpy; a
        # raw python sequence becomes numpy). Vocab is the LAST axis: ``[V]`` (one context)
        # or ``[N, V]`` (population); bulk ops reduce dim=-1, so one object serves both.
        if not (torch.is_tensor(weights) or isinstance(weights, np.ndarray)):
            weights = np.asarray(weights)
        assert weights.shape[-1] == len(decode)
        assert len(encode) == len(decode)

        self.weights = weights
        self.encode = encode
        self.decode = decode
        self.is_log = log

    def __getitem__(self, token):
        """
        Retrieve the weight for a given token.

        Args:
            token (Any): The token for which to retrieve the weight.
                Can be a Token object (direct lookup) or bytes (searches by byte_string).

        Returns:
            (float): The weight of the token, or -inf/0 if the token is not found.
        """
        if token in self.encode:
            return self.weights[self.encode[token]].item()

        # Fallback: if token is plain bytes (not Token), look up by byte_string content.
        # This supports old code that indexes by bytes; returns the first match.
        if Token.is_plain_bytes(token):
            if not hasattr(self, "_bytes_fallback"):
                self._bytes_fallback = {}
                for vocab_token in self.decode:
                    if isinstance(vocab_token, Token) and vocab_token.byte_string not in self._bytes_fallback:
                        self._bytes_fallback[vocab_token.byte_string] = vocab_token
            match = self._bytes_fallback.get(token)
            if match is not None:
                warnings.warn(
                    "Indexing LazyWeights by bytes is deprecated. "
                    "Use Token objects instead (e.g. from llm.tokenize()).",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return self.weights[self.encode[match]].item()

        return float("-inf") if self.is_log else 0

    def __len__(self):
        return self.weights.shape[-1]  # vocab size (last axis), batched or not

    def __array__(self):
        raise NotImplementedError(
            "LazyWeights cannot be converted to a numpy array. "
            "If you want to combine multiple LazyWeights, use their weights attribute directly."
        )

    def keys(self):
        """Return the list of tokens (keys) in the vocabulary."""
        return self.decode

    def values(self):
        """Return the weights associated with the tokens."""
        return self.weights

    def items(self):
        """Return a zip of tokens and weights."""
        return zip(self.keys(), self.values())

    def normalize(self):
        """
        Normalize the weights.

        Normalization is performed using log-space arithmetic when weights are logarithmic,
        or standard arithmetic otherwise.

        Returns:
            (LazyWeights): A new LazyWeights instance with normalized weights.
        """
        if self.is_log:
            return self.spawn(self.weights - _logsumexp(self.weights))
        else:
            return self.spawn(self.weights / self.weights.sum())

    def exp(self):
        """
        Exponentiate the weights. This operation can only be performed when weights are in log space.

        Returns:
            (LazyWeights): A new LazyWeights instance with exponentiated weights.

        Raises:
            AssertionError: If the weights are not in log space.
        """
        assert self.is_log, "Weights must be in log space to exponentiate"
        return self.spawn(_xp(self.weights).exp(self.weights), log=False)

    def log(self):
        """
        Take the logarithm of the weights. This operation can only be performed when weights are in regular space.

        Returns:
            (LazyWeights): A new LazyWeights instance with logarithmic weights.

        Raises:
            AssertionError: If the weights are already in log space.
        """
        assert not self.is_log, "Weights must be in regular space to take the logarithm"
        return self.spawn(_xp(self.weights).log(self.weights), log=True)

    def sum(self):
        """
        Sum the weights.

        Summation is performed using log-space arithmetic when weights are logarithmic,
        or standard arithmetic otherwise.

        Returns:
            (float): The sum of the weights, either in log space or regular space.
        """
        if self.is_log:
            return float(_logsumexp(self.weights))
        else:
            return float(self.weights.sum())

    def spawn(self, new_weights, log=None):
        """
        Create a new LazyWeights instance over the same vocabulary with new weights.

        Args:
            new_weights (np.ndarray): The new weights for the LazyWeights instance.
            log (bool, optional): Indicates if the new weights are in log space. Defaults to None.

        Returns:
            (LazyWeights): A new LazyWeights instance.
        """
        if log is None:
            log = self.is_log
        return LazyWeights(
            weights=new_weights, encode=self.encode, decode=self.decode, log=log
        )

    def materialize(self, top=None):
        """
        Materialize the weights into a chart.

        Args:
            top (int, optional): The number of top weights to materialize. Defaults to None.

        Returns:
            (Chart): A chart representation of the weights.
        """
        weights = self.weights
        order = weights.argsort()
        if top is not None:
            order = order[-int(top) :]

        semiring = Log if self.is_log else Float

        chart = semiring.chart()
        for i in reversed(order.tolist()):
            chart[self.decode[i]] = weights[i].item()

        return chart

    def __repr__(self):
        return repr(self.materialize())

    def assert_equal(self, other, **kwargs):
        """
        Assert that two LazyWeights instances are equal.

        This method asserts that the two LazyWeights instances have the same vocabulary
        (in identical order) and that their weights are numerically close.

        Args:
            other (LazyWeights): The other LazyWeights instance to compare.
            **kwargs (dict): Additional arguments for np.testing.assert_allclose (e.g., rtol, atol).
        """
        assert self.decode == other.decode
        np.testing.assert_allclose(
            to_numpy(self.weights), to_numpy(other.weights), **kwargs
        )

    def assert_equal_unordered(self, other, **kwargs):
        """
        Assert that two LazyWeights instances are equal, ignoring vocabularyorder.

        Args:
            other (LazyWeights): The other LazyWeights instance to compare.
            **kwargs (dict): Additional arguments for np.isclose (e.g., rtol, atol).
        """
        assert set(self.decode) == set(other.decode), "keys do not match"

        for x in self.decode:
            have, want = self[x], other[x]
            assert np.isclose(have, want, **kwargs), f"{x}: {have} != {want}"


def load_trie(V, backend=None, **kwargs):
    """
    Load a TokenCharacterTrie.

    Args:
        V (list[Token] | list[bytes] | list[str]): The vocabulary.
        backend (str, optional): The backend to use for trie construction. Defaults to None.
        **kwargs (dict): Additional arguments for the trie construction.

    Returns:
        (TokenCharacterTrie): A trie instance.
    """
    from genlm.backend.tokenization import Token  # lazy: backend absent on mac

    # Convert pure bytes/strings vocabularies to Token objects.
    # Skip if V already contains Token objects (Token subclasses bytes,
    # so we must check Token first).
    if (
        V
        and not isinstance(V[0], Token)
        and all(isinstance(item, (bytes, str)) for item in V)
    ):
        V = [
            Token(
                token_id=i,
                byte_string=item if isinstance(item, bytes) else item.encode("utf-8"),
            )
            for i, item in enumerate(V)
        ]

    if backend is None:
        backend = "parallel" if torch.cuda.is_available() else "sequential"

    if backend == "parallel":
        from genlm.backend.trie import ParallelTokenCharacterTrie

        return ParallelTokenCharacterTrie(V, **kwargs)
    else:
        from genlm.backend.trie import TokenCharacterTrie

        return TokenCharacterTrie(V, **kwargs)


def load_async_trie(V, backend=None, **kwargs):
    """
    Load an AsyncTokenCharacterTrie. This is a TokenCharacterTrie that
    automatically batches weight_sum and weight_max requests.

    Args:
        V (list): The vocabulary.
        backend (str, optional): The backend to use for trie construction. Defaults to None.
        **kwargs (dict): Additional arguments for the trie construction.

    Returns:
        (AsyncTokenCharacterTrie): An async trie instance.
    """
    from genlm.backend.trie import AsyncTokenCharacterTrie

    return AsyncTokenCharacterTrie(load_trie(V, backend, **kwargs))


# --- token-picker family ---
# Each maps a log-weight tensor -> drawn index over dim=-1 (1-D draw or batched [N, V]).


def gumbel_max(logps):
    """Argmax of ``logps + Gumbel noise`` -- the default picker."""
    g = -torch.log(-torch.log(torch.rand_like(logps)))
    return (logps + g).argmax(dim=-1)


def multinomial(logps):
    """Categorical draw over the last dim (scalar for ``[V]``, ``[N]`` for ``[N, V]``)."""
    p = (logps - torch.logsumexp(logps, dim=-1, keepdim=True)).exp()
    return torch.multinomial(p, 1).squeeze(-1)


def inverse_cdf(logps):
    """Single-uniform inverse-CDF draw over the last dim (scalar for ``[V]``, ``[N]`` for
    ``[N, V]``); one uniform per row, on ``logps``'s device."""
    cdf = (logps - torch.logsumexp(logps, dim=-1, keepdim=True)).exp().cumsum(dim=-1)
    u = torch.rand((*cdf.shape[:-1], 1), dtype=cdf.dtype, device=cdf.device)
    return torch.searchsorted(cdf, u).squeeze(-1).clamp_(max=cdf.shape[-1] - 1)


# --- counter-based (device/order-independent) noise ---
# Picker noise is a pure function of an explicit (seed, slot, step) key, not a shared RNG
# stream: threefry-2x32 in torch int64 is bit-identical CPU/CUDA, so burst (GPU) and StepLoop
# (CPU) draw the SAME noise. Key in scope via the ``draw_key`` ContextVar; unkeyed -> torch.rand.

_DRAW_KEY = contextvars.ContextVar("draw_key", default=None)  # (slot, [next_ordinal]) | None
_DRAW_SEED = 0  # base seed; set by set_draw_seed (mirror of seed_all's seed)

_TF_MASK = 0xFFFFFFFF
_TF_PARITY = 0x1BD11BDA
_TF_ROT = (13, 15, 26, 6, 17, 29, 16, 24)
_TF_ROUNDS = 20


def _rotl32(x, r):
    return ((x << r) | (x >> (32 - r))) & _TF_MASK


def threefry_2x32(c0, c1, k0, k1):
    """Threefry-2x32 keyed hash (``c*``/``k*`` int64 tensors/ints in [0, 2^32)); returns
    the first 32-bit output word. Pure int arithmetic -> bit-identical across devices."""
    ks0, ks1 = k0 & _TF_MASK, k1 & _TF_MASK
    ks2 = (_TF_PARITY ^ ks0 ^ ks1) & _TF_MASK
    ks = (ks0, ks1, ks2)
    x0 = (c0 + ks0) & _TF_MASK
    x1 = (c1 + ks1) & _TF_MASK
    for r in range(_TF_ROUNDS):
        x0 = (x0 + x1) & _TF_MASK
        x1 = _rotl32(x1, _TF_ROT[r % 8])
        x1 = x1 ^ x0
        if (r + 1) % 4 == 0:
            s = (r + 1) // 4
            x0 = (x0 + ks[s % 3]) & _TF_MASK
            x1 = (x1 + ks[(s + 1) % 3] + s) & _TF_MASK
    return x0


def threefry_uniform(n, seed, slot, step, device, dtype):
    """``n`` device-independent uniforms in (0,1) keyed by (seed, slot, step). ``slot``/
    ``step`` may be scalars (-> ``[n]``) or ``[N]`` tensors (-> ``[N, n]``, one keyed row each)."""
    i = torch.arange(n, device=device, dtype=torch.int64)  # counter word 0 (index)
    slot = torch.as_tensor(slot, device=device, dtype=torch.int64)
    step = torch.as_tensor(step, device=device, dtype=torch.int64)
    if slot.ndim:  # batched: [N] keys -> [N, 1] against [n] indices
        i, slot, step = i[None, :], slot[:, None], step[:, None]
    x = threefry_2x32(i & _TF_MASK, step & _TF_MASK, seed & _TF_MASK, slot & _TF_MASK)
    return ((x.double() + 0.5) / 4294967296.0).to(dtype)  # (0,1), bit-identical CPU/CUDA


def threefry_gumbel(logps):
    """Gumbel-max over counter-based noise when a ``draw_key`` is in scope (device/order-
    independent); torch.rand Gumbel fallback when unkeyed. A scalar key advances its ordinal
    per draw; a batched key (per-row ``slot``/``step`` tensors) draws every row of ``[N, V]``
    at once, byte-identical to the per-row scalar draws (noise is keyed, not streamed)."""
    key = _DRAW_KEY.get()
    if key is None:
        u = torch.rand_like(logps)
    else:
        slot, ctr = key
        if torch.is_tensor(slot):  # batched: one draw per row, ordinals fixed
            step = ctr
        else:  # scalar: advance the ordinal in this scope
            step = ctr[0]
            ctr[0] = step + 1
        u = threefry_uniform(logps.shape[-1], _DRAW_SEED, slot, step, logps.device, logps.dtype)
    g = -torch.log(-torch.log(u))
    return (logps + g).argmax(dim=-1)


def set_draw_seed(s):
    """Base seed for the counter-based picker (``threefry_gumbel``); mirror of seed_all."""
    global _DRAW_SEED
    _DRAW_SEED = int(s) & _TF_MASK


def get_draw_seed():
    """Current base seed for the counter-based streams (AWRS's default per-instance seed)."""
    return _DRAW_SEED


def awrs_gumbel_keys(logps, seed, step):
    """``logps + Gumbel`` over the threefry stream keyed by ``(seed, step)`` -- AWRS's OWN
    per-instance (seed, counter), so it is driver-independent yet device-identical."""
    u = threefry_uniform(logps.shape[-1], int(seed) & _TF_MASK, 0, int(step) & _TF_MASK,
                         logps.device, logps.dtype)
    return logps + (-torch.log(-torch.log(u)))


def draw_ordinal(context):
    """Flattened leaf count of a (possibly unit-nested) particle context -- the base draw
    ordinal. Token grain: ``len(context)``; unit grain: total subunits drawn."""
    n = 0
    for item in context:
        n += draw_ordinal(item) if isinstance(item, list) else 1
    return n


@contextlib.contextmanager
def draw_key(slot, base=0):
    """Scope the counter-based picker's key. Scalar ``slot``/``base`` (one particle): ``slot``
    = particle row, ``base`` = draw ordinal so far (advanced per draw). Tensor ``slot``/``base``
    (``[N]`` per-row): one batched draw over the population, ordinals fixed."""
    key = (slot, base) if torch.is_tensor(slot) else (int(slot), [int(base)])
    tok = _DRAW_KEY.set(key)
    try:
        yield
    finally:
        _DRAW_KEY.reset(tok)


DRAW_METHODS = {
    "gumbel_max": gumbel_max,
    "multinomial": multinomial,
    "inverse_cdf": inverse_cdf,
    "threefry_gumbel": threefry_gumbel,
}
# The picker ``select`` uses -- a process-wide setting (see ``set_draw_method``).
_picker = gumbel_max


def set_draw_method(method):
    """Set the token picker ``select`` uses: a name in ``DRAW_METHODS`` (``"gumbel_max"``
    default, ``"multinomial"``, ``"inverse_cdf"``) or a custom ``(logps_tensor) -> index``
    callable. Process-wide."""
    global _picker
    _picker = DRAW_METHODS[method] if isinstance(method, str) else method


def select(lazyweights):
    """Select a token from a log-space ``LazyWeights`` using the configured draw method
    (``set_draw_method``; default Gumbel-max). The single scalar-draw picker seam."""
    assert lazyweights.is_log
    return lazyweights.decode[int(picker_indices(lazyweights.weights))]


def picker_indices(weights):
    """Apply the configured picker to a (possibly batched) log-weight array, returning the
    drawn index/indices over dim=-1 (scalar for ``[V]``, ``[N]`` for ``[N, V]``). Lifts to
    torch (the picker family is pure-torch); a no-op on the common torch path."""
    return _picker(torch.as_tensor(weights))


def escape(x):
    if isinstance(x, EndOfSequence):
        return repr(x)
    elif isinstance(x, int):  # assume its a byte
        x = bytes([x])
    if isinstance(x, bytes):
        y = repr(x)[2:-1]
    else:
        y = repr(x)[1:-1]
    return y.replace(" ", "␣")
