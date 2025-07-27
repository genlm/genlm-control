import pytest
import numpy as np
from genlm.control.constant import EOS
from genlm.control.util import LazyWeights, load_trie, escape


def test_lazy_weights_basic():
    # Test basic initialization and access
    weights = np.array([0.1, 0.2, 0.3])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=False)

    assert lw["a"] == 0.1
    assert lw["b"] == 0.2
    assert lw["c"] == 0.3
    assert lw["d"] == 0  # Non-existent token
    assert len(lw) == 3


def test_lazy_weights_log():
    # Test log-space weights
    weights = np.array([0.0, np.log(2), np.log(3)])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=True)

    assert lw["a"] == 0.0
    assert lw["b"] == np.log(2)
    assert lw["c"] == np.log(3)
    assert lw["d"] == float("-inf")  # Non-existent token


def test_lazy_weights_normalize():
    weights = np.array([0.1, 0.2, 0.3])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    # Test normal-space normalization
    lw = LazyWeights(weights, encode, decode, log=False)
    normalized = lw.normalize()
    np.testing.assert_allclose(np.sum(normalized.weights), 1.0)

    # Test log-space normalization
    lw_log = LazyWeights(np.log(weights), encode, decode, log=True)
    normalized_log = lw_log.normalize()
    np.testing.assert_allclose(np.exp(normalized_log.weights).sum(), 1.0)


def test_lazy_weights_exp_log():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    # Test exp
    lw_log = LazyWeights(np.log(weights), encode, decode, log=True)
    lw_exp = lw_log.exp()
    np.testing.assert_allclose(lw_exp.weights, weights)

    # Test log
    lw = LazyWeights(weights, encode, decode, log=False)
    lw_log = lw.log()
    np.testing.assert_allclose(lw_log.weights, np.log(weights))


def test_lazy_weights_assertions():
    with pytest.raises(NotImplementedError):
        weights = np.array([1.0, 2.0])
        lw = LazyWeights(weights, {"a": 0, "b": 1}, ["a", "b"])
        np.array(lw)

    with pytest.raises(AssertionError):
        lw = LazyWeights(np.log(weights), {"a": 0, "b": 1}, ["a", "b"], log=True)
        lw.log()  # Can't take log of log weights

    with pytest.raises(AssertionError):
        lw = LazyWeights(weights, {"a": 0, "b": 1}, ["a", "b"], log=False)
        lw.exp()  # Can't take exp of non-log weights


def test_lazy_weights_sum():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    # Test sum in normal space
    lw = LazyWeights(weights, encode, decode, log=False)
    assert lw.sum() == 6.0

    # Test sum in log space
    log_weights = np.log(weights)
    lw_log = LazyWeights(log_weights, encode, decode, log=True)
    np.testing.assert_allclose(lw_log.sum(), np.log(6.0))


def test_lazy_weights_assert_equal():
    w1 = np.array([1.0, 2.0, 3.0])
    w2 = np.array([1.0, 2.0, 3.0])
    w3 = np.array([1.1, 2.0, 3.0])  # Slightly different weights
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw1 = LazyWeights(w1, encode, decode, log=False)
    lw2 = LazyWeights(w2, encode, decode, log=False)
    lw3 = LazyWeights(w3, encode, decode, log=False)

    # Test exact equality
    lw1.assert_equal(lw2)

    # Test equality with tolerance
    with pytest.raises(AssertionError):
        lw1.assert_equal(lw3)
    lw1.assert_equal(lw3, rtol=0.2, atol=0.2)  # Should pass with higher tolerance


def test_lazy_weights_assert_equal_unordered():
    w1 = np.array([1.0, 2.0, 3.0])
    w2 = np.array([3.0, 1.0, 2.0])  # Same values, different order

    encode1 = {"a": 0, "b": 1, "c": 2}
    encode2 = {"c": 0, "a": 1, "b": 2}
    decode1 = ["a", "b", "c"]
    decode2 = ["c", "a", "b"]

    lw1 = LazyWeights(w1, encode1, decode1, log=False)
    lw2 = LazyWeights(w2, encode2, decode2, log=False)

    # Test unordered equality
    lw1.assert_equal_unordered(lw2)

    # Test with missing key
    encode3 = {"a": 0, "b": 1, "d": 2}
    decode3 = ["a", "b", "d"]
    lw3 = LazyWeights(w1, encode3, decode3, log=False)

    with pytest.raises(AssertionError, match="keys do not match"):
        lw1.assert_equal_unordered(lw3)

    with pytest.raises(AssertionError, match="keys do not match"):
        lw3.assert_equal_unordered(lw1)


def test_lazy_weights_keys():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=False)
    assert lw.keys() == ["a", "b", "c"]


def test_lazy_weights_values():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=False)
    assert list(lw.values()) == [1.0, 2.0, 3.0]


def test_lazy_weights_items():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=False)
    assert list(lw.items()) == [("a", 1.0), ("b", 2.0), ("c", 3.0)]


def test_load_trie():
    vocab = ["a", "b", "c"]
    trie = load_trie(vocab, backend="sequential")
    assert trie.decode == vocab

    trie = load_trie(vocab, backend="parallel")
    assert trie.decode == vocab

    trie = load_trie(vocab)
    assert trie.decode == vocab


def test_lazy_weights_repr():
    weights = np.array([1.0, 2.0, 3.0])
    encode = {"a": 0, "b": 1, "c": 2}
    decode = ["a", "b", "c"]

    lw = LazyWeights(weights, encode, decode, log=False)
    lw.__repr__()


def test_escape():
    assert escape(10) == "\\n"
    assert escape(b"hello") == "hello"
    assert escape("hello") == "hello"
    assert escape(EOS) == "EOS"
