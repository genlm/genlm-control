import pytest
from genlm.control.constant import EndOfSequence, EOS, EOT


def test_init_and_repr():
    eos = EndOfSequence("TEST")
    assert repr(eos) == "TEST"
    assert str(eos) == "TEST"


def test_equality():
    eos1 = EndOfSequence("TEST")
    eos2 = EndOfSequence("TEST")
    eos3 = EndOfSequence("OTHER")

    assert eos1 == eos2
    assert eos1 != eos3
    assert eos1 != "TEST"  # Compare with non-EndOfSequence


def test_radd():
    eos = EndOfSequence("TEST")

    # Test with string
    result = "hello" + eos
    assert isinstance(result, list)
    assert result == ["h", "e", "l", "l", "o", eos]

    # Test with bytes
    result = b"hello" + eos
    assert isinstance(result, list)
    assert result == [104, 101, 108, 108, 111, eos]

    # Test with list and verify type preservation
    input_list = [1, 2, 3]
    result = input_list + eos
    assert isinstance(result, list)
    assert result == [1, 2, 3, eos]
    assert type(result) is type(input_list)

    # Test with tuple and verify type preservation
    input_tuple = (1, 2, 3)
    result = input_tuple + eos
    assert isinstance(result, tuple)
    assert result == (1, 2, 3, eos)
    assert type(result) is type(input_tuple)


def test_radd_error():
    eos = EndOfSequence("TEST")
    with pytest.raises(
        TypeError,
        match=r"Cannot concatenate <class 'int'> with <class '.*EndOfSequence'>",
    ):
        _ = 42 + eos


def test_hash():
    eos1 = EndOfSequence("TEST")
    eos2 = EndOfSequence("TEST")
    eos3 = EndOfSequence("OTHER")

    # Same type should have same hash
    assert hash(eos1) == hash(eos2)
    # Different type should have different hash
    assert hash(eos1) != hash(eos3)

    # Test can be used in sets/dicts
    test_set = {eos1, eos2, eos3}
    assert len(test_set) == 2


def test_predefined_constants():
    assert isinstance(EOS, EndOfSequence)
    assert EOS.type_ == "EOS"
    assert isinstance(EOT, EndOfSequence)
    assert EOT.type_ == "EOT"
