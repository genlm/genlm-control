import warnings

import numpy as np
import torch
from genlm.grammar import Float, Log

from genlm.control.constant import EndOfSequence
from genlm.backend.draw import draw_from as backend_draw_from
from genlm.backend.tokenization import Token


def logsumexp(x, axis=-1, keepdims=False):
    """Log-sum-exp along `axis`, in the array's own backend. An all-`-inf` slice
    reduces to `-inf`, not `nan`. The default axis is the last one, so a batched
    `[N, V]` block reduces per row."""
    if torch.is_tensor(x):
        return torch.logsumexp(x, axis, keepdim=keepdims)
    x = np.asarray(x)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isneginf(m), 0.0, m)  # all -inf: shift by 0 so the sum is 0
    with np.errstate(divide="ignore"):
        out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return out if keepdims else np.squeeze(out, axis=axis)


def to_numpy(w):
    """Coerce a weight array to numpy regardless of backend (no-op on numpy)."""
    return w.cpu().numpy() if torch.is_tensor(w) else np.asarray(w)


def stack_weights(arrays):
    """Stack per-context weight arrays into one `[N, V]` batch, preserving the producer's
    backend (numpy stays numpy, torch stays torch)."""
    return torch.stack(arrays) if torch.is_tensor(arrays[0]) else np.stack(arrays)


def _xp(w):
    """The array module (`torch` or `np`) backing `w`, for backend-agnostic ops."""
    return torch if torch.is_tensor(w) else np


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
        # `weights` keeps the producer's backend (LM->torch, grammar/FSA/trie->numpy; a
        # raw python sequence becomes numpy). Vocab is the last axis: `[V]` for one
        # context, `[N, V]` for a population, and bulk ops reduce dim=-1.
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

        # Fallback: look up plain-bytes tokens by byte_string content (first match wins).
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

    def materialize(self, top=None, sort=True):
        """
        Materialize the weights into a chart.

        Args:
            top (int, optional): The number of top weights to materialize. Defaults to None.
            sort (bool, optional): Order the chart by descending weight. Defaults to True.
                Required by `top`; skip it when only the token-to-weight mapping is needed.

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


async def draw_from(lazyweights, draw=None, target=None):
    """Draw a token from a `LazyWeights` row through the backend draw window
    (`genlm.backend.draw`) and weigh it. A custom `draw` picker draws solo, outside
    the window.

    Args:
        lazyweights (LazyWeights): The log-weight row to draw from.
        draw (callable, optional): Picker taking a materialized normalized chart and
            returning a token.
        target (LazyWeights, optional): A second row over the same vocabulary, making
            this an importance draw: `lazyweights` is the proposal and the token is
            weighed under `target`.

    Returns:
        (tuple): `(token, logw, logp)`, `logw` being the row's log-normalizer, or
            `target[token] - logp` with a target.
    """
    assert lazyweights.is_log
    if draw is None:
        idx, logw, logp = await backend_draw_from(
            lazyweights.weights,
            target=None if target is None else target.weights,
        )
        return lazyweights.decode[idx], logw, logp

    logZ = lazyweights.sum()
    logps = lazyweights.spawn(lazyweights.weights - logZ)
    token = draw(logps.exp().materialize())
    logp = logps[token]
    if target is None:
        return token, logZ, logp
    return token, target[token] - logp, logp


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
