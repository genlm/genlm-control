class EndOfSequence:
    """End-of-sequence tokens."""

    def __init__(self, type_="EOS"):
        self.type_ = type_

    def __repr__(self):
        return self.type_

    def __eq__(self, other):
        return isinstance(other, EndOfSequence) and self.type_ == other.type_

    def __radd__(self, other):
        if isinstance(other, (str, bytes)):
            return [*list(other), self]
        elif isinstance(other, (list, tuple)):
            return type(other)(list(other) + [self])
        else:
            raise TypeError(f"Cannot concatenate {type(other)} with {type(self)}")

    def __hash__(self):
        return hash(self.type_)

    def __deepcopy__(self, memo):
        return self  # a value-like singleton

    def __iter__(self):
        return iter([self])

    def __len__(self):
        return 1


EOS = EndOfSequence("EOS")
EOT = EndOfSequence("EOT")
