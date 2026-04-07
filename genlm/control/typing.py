from dataclasses import dataclass
from collections.abc import Sequence as SequenceABC

from genlm.control.constant import EndOfSequence
from genlm.backend.tokenization import Token


@dataclass
class TokenType:
    """Base class representing the type of a token"""

    def check(self, value):
        """Check if a value matches this type"""
        raise NotImplementedError()  # pragma: no cover

    def is_iterable_of(self, element_type):
        """Check if this type can be interpreted as an iterable of element_type.

        Args:
            element_type (TokenType): The type to check if this is an iterable of

        Examples:
            >>> Sequence(Atomic(int)).is_iterable_of(Atomic(int))
            True
            >>> Atomic(bytes).is_iterable_of(Atomic(int))
            True
        """
        if isinstance(self, Sequence):
            return self.element_type == element_type

        if isinstance(self, Atomic):
            # Special cases for built-in iterables
            if (
                self.type is bytes
                and isinstance(element_type, Atomic)
                and element_type.type is int
            ):
                return True
            if (
                self.type is str
                and isinstance(element_type, Atomic)
                and element_type.type is str
            ):
                return True
            if (
                self.type is Token
                and isinstance(element_type, Atomic)
                and element_type.type is int
            ):
                return True

        return False


@dataclass
class Atomic(TokenType):
    """Represents a simple type like int or str"""

    type: type  # The Python type (int, str, etc.)

    def check(self, value):
        return isinstance(value, self.type) or isinstance(value, EndOfSequence)

    def convert(self, value):
        return self.type(value)

    def __repr__(self):
        return f"Atomic({self.type.__name__})"


@dataclass
class Sequence(TokenType):
    """Represents a list/sequence of another type"""

    element_type: TokenType  # The type of elements in the sequence

    def check(self, value):
        return isinstance(value, (list, tuple)) and all(
            self.element_type.check(x) for x in value
        )

    def convert(self, value):
        return tuple(self.element_type.convert(x) for x in value)

    def __repr__(self):
        return f"Sequence({self.element_type!r})"


def infer_type(value):
    """Infer the TokenType from a value.

    Args:
        value (Any): A sample value to infer type from

    Returns:
        (TokenType): The inferred type

    Examples:
        >>> infer_type(42)
        Atomic(type=int)
        >>> infer_type([1, 2, 3])
        Sequence(element_type=Atomic(type=int))
        >>> infer_type([[1, 2], [3, 4]])
        Sequence(element_type=Sequence(element_type=Atomic(type=int)))
    """
    if isinstance(value, SequenceABC) and not isinstance(value, (bytes, str)):
        if not value:
            raise ValueError("Cannot infer type from empty sequence")
        element_type = infer_type(value[0])
        if not all(element_type.check(x) for x in value):
            raise ValueError("Inconsistent types in sequence")
        return Sequence(element_type)

    return Atomic(type(value))


def infer_vocabulary_type(vocabulary):
    """Infer the TokenType from a vocabulary.

    Args:
        vocabulary (List[Any]): A list of tokens to infer type from

    Returns:
        (TokenType): The inferred type

    Raises:
        ValueError: If vocabulary is empty or contains inconsistent types

    Examples:
        >>> infer_vocabulary_type([1, 2, 3])
        Atomic(type=int)
        >>> infer_vocabulary_type([[1, 2], [3, 4]])
        Sequence(element_type=Atomic(type=int))
    """
    if not vocabulary:
        raise ValueError("Cannot infer type from empty vocabulary")

    token_type = infer_type(vocabulary[0])
    if not all(token_type.check(x) for x in vocabulary):
        raise ValueError("Inconsistent types in vocabulary")

    return token_type
