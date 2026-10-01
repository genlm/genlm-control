import asyncio
import warnings
import weakref
from collections import defaultdict

import numpy as np
import torch
from genlm.grammar import Float, Log

from genlm.control.constant import EndOfSequence
from genlm.backend.tokenization import Token


def logsumexp(x, axis=-1, keepdims=False):
    """Log-sum-exp along `axis`, in the array's own backend. An all-`-inf` slice
    reduces to `-inf`, not `nan`."""
    if torch.is_tensor(x):
        return torch.logsumexp(x, axis, keepdim=keepdims)
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isneginf(m), 0.0, m)  # all -inf: shift by 0 so the sum is 0
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return out if keepdims else np.squeeze(out, axis=axis)


def to_numpy(w):
    """Return `w` as a numpy array."""
    return w.cpu().numpy() if torch.is_tensor(w) else np.asarray(w)


def stack_weights(arrays):
    """Stack per-context weight arrays into one `[N, V]` batch in their own backend."""
    return torch.stack(arrays) if torch.is_tensor(arrays[0]) else np.stack(arrays)


def _xp(w):
    """The array module (`torch` or `np`) backing `w`."""
    return torch if torch.is_tensor(w) else np


class LazyWeights:
    """
    A class to represent weights in a lazy manner, allowing for efficient operations
    on potentially large weight arrays without immediate materialization.

    Attributes:
        weights (np.ndarray | torch.Tensor): The weights associated with the tokens.
        encode (dict): A mapping from tokens to their corresponding indices in the weights array.
        decode (list): A list of tokens corresponding to the weights.
        is_log (bool): A flag indicating whether the weights are in log space.
    """

    def __init__(self, weights, encode, decode, log=True):
        """
        Initialize the LazyWeights instance.

        Args:
            weights (np.ndarray | torch.Tensor): The weights associated with the tokens.
            encode (dict): A mapping from tokens to their corresponding indices in the weights array.
            decode (list): A list of tokens corresponding to the weights.
            log (bool, optional): Indicates if the weights are in log space. Defaults to True.

        Raises:
            AssertionError: If the lengths of weights and decode do not match, or if encode has fewer entries than decode.
        """
        # Vocab is the last axis: `[V]` for one context, `[N, V]` for a batch.
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
            return self.spawn(self.weights - logsumexp(self.weights, keepdims=True))
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
            return float(logsumexp(self.weights))
        else:
            return float(self.weights.sum())

    def spawn(self, new_weights, log=None):
        """
        Create a new LazyWeights instance over the same vocabulary with new weights.

        Args:
            new_weights (np.ndarray | torch.Tensor): The new weights for the LazyWeights instance.
            log (bool, optional): Indicates if the new weights are in log space. Defaults to None.

        Returns:
            (LazyWeights): A new LazyWeights instance.
        """
        if log is None:
            log = self.is_log
        return LazyWeights(
            weights=new_weights, encode=self.encode, decode=self.decode, log=log
        )

    def materialize(self, top=None, sort=True):
        """
        Materialize the weights into a chart.

        Args:
            top (int, optional): The number of top weights to materialize. Defaults to None.
            sort (bool, optional): Order the chart by descending weight; ignored when `top` is set. Defaults to True.

        Returns:
            (Chart): A chart representation of the weights.
        """
        weights = self.weights
        if not sort and top is None:
            semiring = Log if self.is_log else Float
            chart = semiring.chart()
            for i, w in enumerate(weights.tolist()):
                chart[self.decode[i]] = w
            return chart
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


# Draw methods: each maps a log-weight tensor to a drawn index over the last dim
# (scalar for `[V]`, `[N]` for `[N, V]`). They draw from the global torch RNG, so
# `torch.manual_seed`, not `np.random.seed`, makes their draws reproducible. They
# stay at the row's dtype: MPS has no fp64.


def gumbel_max(logps):
    """Argmax of `logps + Gumbel noise`; the default draw method."""
    g = -torch.log(-torch.log(torch.rand_like(logps)))
    return (logps + g).argmax(dim=-1)


def multinomial(logps):
    """Categorical draw over the last dim (scalar for `[V]`, `[N]` for `[N, V]`)."""
    p = (logps - torch.logsumexp(logps, dim=-1, keepdim=True)).exp()
    return torch.multinomial(p, 1).squeeze(-1)


def inverse_cdf(logps):
    """Inverse-CDF draw over the last dim, one uniform per row (scalar for `[V]`, `[N]`
    for `[N, V]`)."""
    cdf = (logps - torch.logsumexp(logps, dim=-1, keepdim=True)).exp().cumsum(dim=-1)
    u = torch.rand((*cdf.shape[:-1], 1), dtype=cdf.dtype, device=cdf.device)
    return torch.searchsorted(cdf, u).squeeze(-1).clamp_(max=cdf.shape[-1] - 1)


def flatten_units(context):
    """Recursively flatten a (possibly unit-nested) context to a flat token list.

    Usage:
        potential.coerce(LLM, f=lambda ctx: b"".join(flatten_units(ctx)))
    """
    flattened = []
    for item in context:
        if isinstance(item, list):
            flattened.extend(flatten_units(item))
        else:
            flattened.append(item)
    return flattened


DRAW_METHODS = {
    "gumbel_max": gumbel_max,
    "multinomial": multinomial,
    "inverse_cdf": inverse_cdf,
}
# Process-wide draw method for `draw_from`; set it via `set_draw_method`.
_draw_method = gumbel_max


def set_draw_method(method):
    """
    Set the draw method used by `draw_from` and `draw_indices`, process-wide.

    Args:
        method (str | callable): A name in `DRAW_METHODS`, or a custom
            `(logps_tensor) -> index` callable.
    """
    global _draw_method
    _draw_method = DRAW_METHODS[method] if isinstance(method, str) else method


async def draw_from(lazyweights, draw=None, target=None):
    """
    Draw a token from a next-token distribution and weigh it.

    Concurrent callers on one event loop are drawn together, one batched reduction
    per backend, device and vocab size, each row with independent noise.

    Args:
        lazyweights (LazyWeights): The log-weight row to draw from.
        draw (callable, optional): Custom draw, taking the normalized probability
            chart and returning a token. Bypasses batching.
        target (LazyWeights, optional): A second row over the same vocabulary. Makes
            the draw an importance draw: `lazyweights` is the proposal, and the token
            is weighed under the target.

    Returns:
        (tuple): `(token, logw, logp)`, where `logw` is the row's normalizer `logZ`
            without `target` and `target[token] - logp` with it.
    """
    if draw is not None:
        logZ = lazyweights.sum()
        logps = lazyweights.spawn(lazyweights.weights - logZ)
        token = draw(logps.exp().materialize())
        logp = logps[token]
        if target is None:
            return token, logZ, logp
        return token, target[token] - logp, logp

    assert lazyweights.is_log
    future = asyncio.get_running_loop().create_future()
    batch = await join_batch(_DRAW_BATCHES, (lazyweights, target, future))
    if batch is not None:
        _flush_draws(batch)
    return await future


class BatchAbandoned(RuntimeError):
    """The caller holding a batch died before it could be flushed."""


class _Batch:
    """Per-event-loop request meeting point; must not outlive its loop."""

    __slots__ = ("queue", "armed")

    def __init__(self):
        self.queue = []
        self.armed = False


async def join_batch(store, entry):
    """
    Join the batch of concurrent callers on this event loop.

    The first caller holds the batch open until an event-loop pass adds no entry,
    then receives it to flush. If that leader dies first, every other queued future
    is failed with `BatchAbandoned`.

    Args:
        store (weakref.WeakKeyDictionary): Event loop to `_Batch` map, one per
            batched call site.
        entry (tuple): The request to add, ending in its future.

    Returns:
        (list | None): The drained batch for the holding caller, `None` for everyone
            else.
    """
    loop = asyncio.get_running_loop()
    batch = store.get(loop)
    if batch is None:
        batch = store[loop] = _Batch()
    batch.queue.append(entry)
    if batch.armed:
        return None
    batch.armed = True
    try:
        # Callers nested in a `gather` tree arrive one scheduler turn per level.
        while True:
            n = len(batch.queue)
            await asyncio.sleep(0)
            if len(batch.queue) == n:
                break
        queue, batch.queue = batch.queue, []
        return queue
    except BaseException as exc:
        queue, batch.queue = batch.queue, []
        # Skip this caller's own entry: it is unwinding, so an exception set on it
        # would go unretrieved.
        fail_futures([e for e in queue if e is not entry], batch_abandoned(exc))
        raise
    finally:
        batch.armed = False


def batch_abandoned(exc):
    """The failure handed to callers whose batch leader died.

    Never the cause itself: a `CancelledError` given to a caller who never asked for
    one leaves their task cancelled and skips their `except Exception`.
    """
    abandoned = BatchAbandoned(f"batch leader did not survive it: {exc!r}")
    abandoned.__cause__ = exc
    return abandoned


def fail_futures(entries, exc):
    """Resolve each entry's future -- its last element -- with `exc`."""
    for entry in entries:
        future = entry[-1]
        if not future.done():
            future.set_exception(exc)


_DRAW_BATCHES = weakref.WeakKeyDictionary()  # event loop -> _Batch


def _flush_draws(queue):
    """Resolve a batch of draws with one batched reduction per stackable group. Every
    future is resolved, with its draw or with a failure."""
    groups = defaultdict(list)
    for lw, target, future in queue:
        w = lw.weights
        key = (
            ("torch", w.device, w.shape[-1])
            if torch.is_tensor(w)
            else ("np", w.shape[-1])
        )
        groups[key].append((lw, target, future))
    for entries in groups.values():
        try:
            rows = torch.stack([torch.as_tensor(lw.weights) for lw, _, _ in entries])
            logZ = torch.logsumexp(rows, dim=-1)
            logps = rows.sub_(logZ.unsqueeze(-1))  # stack copied; safe in place
            idx = _draw_method(logps)
            logp = logps.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            ids = idx.tolist()
            # One float readback for the batch's normalizers and drawn log-probs.
            logZs, drawn_logps = torch.stack([logZ, logp]).tolist()
            # Importance draws: the drawn token's log-weight under each target row.
            target_logws = {}
            targeted = [k for k, (_, t, _) in enumerate(entries) if t is not None]
            if targeted:
                t_rows = torch.stack(
                    [torch.as_tensor(entries[k][1].weights) for k in targeted]
                )
                t_idx = idx[targeted]
                t_vals = t_rows.gather(-1, t_idx.unsqueeze(-1)).squeeze(-1).tolist()
                target_logws = dict(zip(targeted, t_vals))
        except Exception as exc:
            fail_futures(entries, exc)
            continue
        except BaseException as exc:
            # Unwinding: nothing else will resolve the futures still pending.
            for rest in groups.values():
                fail_futures(rest, batch_abandoned(exc))
            raise
        for k, ((lw, target, future), tok_id, z, p) in enumerate(
            zip(entries, ids, logZs, drawn_logps)
        ):
            if not future.done():
                logw = z if target is None else target_logws[k] - p
                future.set_result((lw.decode[tok_id], logw, p))


def draw_indices(weights):
    """Draw indices over the last dim of a log-weight array with the configured draw method
    (scalar for `[V]`, `[N]` for `[N, V]`)."""
    return _draw_method(torch.as_tensor(weights))


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
