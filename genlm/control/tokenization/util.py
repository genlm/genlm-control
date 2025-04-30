import numpy as np
from arsenal import colors
from collections import namedtuple


_encode_bytes_str = [
    'Ā', 'ā', 'Ă', 'ă', 'Ą', 'ą', 'Ć', 'ć', 'Ĉ', 'ĉ', 'Ċ', 'ċ', 'Č', 'č', 'Ď', 'ď',
    'Đ', 'đ', 'Ē', 'ē', 'Ĕ', 'ĕ', 'Ė', 'ė', 'Ę', 'ę', 'Ě', 'ě', 'Ĝ', 'ĝ', 'Ğ', 'ğ',
    'Ġ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?',
    '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_',
    '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'ġ',
    'Ģ', 'ģ', 'Ĥ', 'ĥ', 'Ħ', 'ħ', 'Ĩ', 'ĩ', 'Ī', 'ī', 'Ĭ', 'ĭ', 'Į', 'į', 'İ', 'ı',
    'Ĳ', 'ĳ', 'Ĵ', 'ĵ', 'Ķ', 'ķ', 'ĸ', 'Ĺ', 'ĺ', 'Ļ', 'ļ', 'Ľ', 'ľ', 'Ŀ', 'ŀ', 'Ł',
    'ł', '¡', '¢', '£', '¤', '¥', '¦', '§', '¨', '©', 'ª', '«', '¬', 'Ń', '®', '¯',
    '°', '±', '²', '³', '´', 'µ', '¶', '·', '¸', '¹', 'º', '»', '¼', '½', '¾', '¿',
    'À', 'Á', 'Â', 'Ã', 'Ä', 'Å', 'Æ', 'Ç', 'È', 'É', 'Ê', 'Ë', 'Ì', 'Í', 'Î', 'Ï',
    'Ð', 'Ñ', 'Ò', 'Ó', 'Ô', 'Õ', 'Ö', '×', 'Ø', 'Ù', 'Ú', 'Û', 'Ü', 'Ý', 'Þ', 'ß',
    'à', 'á', 'â', 'ã', 'ä', 'å', 'æ', 'ç', 'è', 'é', 'ê', 'ë', 'ì', 'í', 'î', 'ï',
    'ð', 'ñ', 'ò', 'ó', 'ô', 'õ', 'ö', '÷', 'ø', 'ù', 'ú', 'û', 'ü', 'ý', 'þ', 'ÿ',
]

# this is the inverse mapping of `_bytes_to_unicode`
_decode_str_bytes = {s: i for i, s in enumerate(_encode_bytes_str)}
_default_byte_decoder = _decode_str_bytes


def decode_hf_tokenizer(tokenizer):
    "Extract what we need from a 🤗 tokenizer."
    _merges = []
    V = tokenizer.get_vocab()
    if hasattr(tokenizer, 'bpe_ranks'):
        for (u,v) in tokenizer.bpe_ranks:
            _merges.append((V[u], V[v], V[u + v]))
    else:
        import json
        subtokenizer_dict = json.loads(tokenizer._tokenizer.to_str())
        for (u,v) in subtokenizer_dict["model"]["merges"]:
            _merges.append((V[u], V[v], V[u + v]))

    if hasattr(tokenizer, 'byte_decoder'):
        byte_decoder = tokenizer.byte_decoder
    else:
        byte_decoder = _default_byte_decoder

    _encode = {}
    _decode = [None]*len(V)
    for bs, token_id in V.items():
        b = bytes([byte_decoder[b] for b in bs])
        _encode[b] = token_id
        _decode[token_id] = b

    # map each byte (0-255) to token id (they are annoyingly not the same)
    _encode_byte = [None]*256
    for i in range(256):
        _encode_byte[i] = _encode[bytes([i])]

    return (_merges, _encode, _decode, _encode_byte)


class MyTree(namedtuple('MyTree', 'left, right')):
    def __repr__(self):
        return pretty(self)
    def to_nltk(self):
        import nltk
        if isinstance(self, tuple):
            return nltk.Tree('', [MyTree.to_nltk(y) for y in self])
        else:
            return escape(str(self))[2:-1]
    def _repr_html_(self):
        return self.to_nltk()._repr_svg_()


def pretty(x):
    if isinstance(x, tuple):
        y,z = x
        return (colors.dark.white % '(') + f'{pretty(y)}{pretty(z)}' + (colors.dark.white % ')')
    else:
        return escape(str(x)[2:-1])


def logsumexp(arr):
    """
    Compute `log(sum(exp(arr)))` without overflow.
    """
    arr = np.array(arr, dtype=np.float64)
    arr = arr[arr > -np.inf]
    if len(arr) == 0: 
        return -np.inf
    vmax = arr.max()
    arr -= vmax
    np.exp(arr, out=arr)
    out = np.log(arr.sum())
    out += vmax
    return out


def logmeanexp(xs):
    """
    Numerically stable implementation of log(mean(exp(xs))).

    Nptes:
      log(mean(exp(xs)))
      = log(sum(exp(xs))/n)
      = log(sum(exp(xs))) - log(n)
      = logsumexp(xs) - log(n)

    """
    return logsumexp(xs) - np.log(len(xs))


def escape(x):
    if isinstance(x, int):   # assume its a byte
        x = bytes([x])
    if isinstance(x, bytes):
        y = repr(x)[2:-1]
    else:
        y = repr(x)[1:-1]
    return y.replace(" ","␣")

def prefixes(z):
    """
    Return the prefixes of the sequence `z`

      >>> list(prefixes(''))
      ['']

      >>> list(prefixes('abc'))
      ['', 'a', 'ab', 'abc']

    """
    for p in range(len(z) + 1):
        yield z[:p]

def flatten(xs):
    if len(xs) == 0:
        return ()
    else:
        ys, y = xs
        return flatten(ys) + (y,)


def unflatten(ys):
    xs = ()
    for y in ys:
        xs = (xs, y)
    return xs


def longest_common_prefix(xs):
    if not xs:
        return ""

    # Sort the strings
    xs = sorted(xs)

    # Compare only the first and the last strings
    first = xs[0]
    last = xs[-1]

    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1

    # The longest common prefix will be the portion of the first string up to i
    return first[:i]


def lcp(xs, ys):
    "return the longest common prefix of `xs` and `ys` and the suffixes of `xs` and `ys` that are not common."
    i = 0
    N = len(xs)
    M = len(ys)
    while i < N and i < M and xs[i] == ys[i]:
        i += 1
    return xs[:i], xs[i:], ys[i:]


def prefix(xs, ys):
    assert isinstance(xs, str) and isinstance(ys, str)
    return ys.startswith(xs)


def strict_prefix(xs, ys):
    assert isinstance(xs, str) and isinstance(ys, str)
    return prefix(xs, ys) and xs != ys


def cons2str(ys):
    xs = []
    while ys != ():
        ys, y = ys
        xs.append(y)
    return ''.join(reversed(xs))


def covers(qs, ys):
    assert isinstance(qs, str) and isinstance(ys, tuple)
    return (qs == "") if ys == () else strict_prefix(cons2str(ys[0]), qs) and prefix(qs, cons2str(ys))
