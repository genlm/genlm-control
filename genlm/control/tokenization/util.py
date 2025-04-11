import html
import pylab as pl
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import HTML, SVG, display
from arsenal import colors
from arsenal.iterextras import batch
from collections import namedtuple


class posterior_encodings:
    """
    Quick method to inspect the posterior distribution over tokenizations of a given string.

    - C: CharacterBeam
    - bpe: BPE
    - xs: bytes

    """
    def __init__(self, C, bpe, xs):
        self.xs = xs
        self.encodings = Chart(-np.inf, {flatten(item.ys): item.ps for item in C.encodings(xs)})
        self.logZ = logsumexp(list(self.encodings.values()))
        self.canonical = bpe.encode_as_byte_chunks(xs)

    def show(self, top=None, highlight=None):
        if highlight is None: highlight = self.canonical
        for y, w in self.encodings.top(top).items():
            y = list(y)
            if highlight == y:
                print(colors.bold % np.exp(w - self.logZ), colors.bold % (y,))
            else:
                print(np.exp(w - self.logZ), y)


#def _bytes_to_unicode():
#    """Create a mapping from bytes to Unicode characters.
#
#    Returns:
#        dict: Mapping from byte values to Unicode characters
#    """
#
#    # https://chatgpt.com/share/6793c6f3-76cc-800c-97d0-0ae80e9df19f
#    #
#    # In language models like GPT, this mapping is used to tokenize raw input
#    # text into Unicode-compatible strings while preserving all possible byte
#    # sequences. Starting with printable characters ensures that frequent text
#    # remains human-readable, while less frequent or non-printable bytes are
#    # handled gracefully with unique Unicode characters.
#    #
#    # These characters overlap with standard encodings like ASCII and ISO-8859-1
#    # (Latin-1). Ensuring these byte values map directly to readable characters
#    # helps maintain compatibility with text encoded in those formats.
#    #
#    # The initial `bs` includes the following ranges:
#    #
#    # * ord("!") to ord("~"):
#    #
#    #   These correspond to the printable ASCII characters (e.g., !, A-Z, a-z,
#    #   0-9, and symbols like #, $, %).
#    #
#    #   These are among the most commonly used characters in plain text, so
#    #   mapping these bytes directly to themselves ensures intuitive and
#    #   readable output.
#    #
#    # * ord("¡") to ord("¬") and ord("®") to ord("ÿ"):
#    #
#    #   These are printable Latin-1 Supplement characters, including accented
#    #   letters (e.g., é, ü), punctuation marks (e.g., ¡, ¿), and symbols.
#    #
#    #   These characters are important for supporting European languages and are
#    #   frequently used in multilingual text.
#    #
#    # The function avoids including non-printable control characters (e.g.,
#    # 0x00--0x1F in ASCII) in the initial bs list. These characters are not
#    # typically useful for readable text and could cause display or processing
#    # issues if included.
#
#    bs = (
#        list(range(ord("!"), ord("~") + 1))
#        + list(range(ord("¡"), ord("¬") + 1))
#        + list(range(ord("®"), ord("ÿ") + 1))
#    )
#    cs = bs[:]
#    n = 0
#    for b in range(256):
#        if b not in bs:
#            bs.append(b)
#            cs.append(256 + n)
#            n += 1
#    cs = [chr(n) for n in cs]
#    return dict(zip(bs, cs))


# output of `_bytes_to_unicode`
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
    if len(arr) == 0: return -np.inf
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


def make_prefix_free(collection):
    """
    Make the collection prefix-free, i.e., remove any string that is a prefix of
    another.

    >>> make_prefix_free([])
    []
    >>> make_prefix_free(['aa','aa'])
    ['aa']
    >>> make_prefix_free(['aaaa','bbbb',''])
    ['']
    >>> make_prefix_free(['a','ab','abc','b'])
    ['a', 'b']
    >>> make_prefix_free(['ab','abc','b'])
    ['ab', 'b']

    """
    result = []
    for i, t in enumerate(sorted(collection)):
        if i == 0:
            result.append(t)
        else:
            prev = result[-1]
            if prev != t[:len(prev)]:
                result.append(t)
    return result


def complementary_prefix_set(context, V, eos):
    """
    Enumerate all the ways for a prefix to deviate from `context` under the
    vocabulary `V` and end-of-string `eos`.

    >>> [y + a for y, a in complementary_prefix_set('aaaa', {'a'}, '▪')]
    ['▪', 'a▪', 'aa▪', 'aaa▪']

    >>> [y + a for y, a in complementary_prefix_set('aaaa', {'a', 'b'}, '▪')]
    ['▪', 'b', 'a▪', 'ab', 'aa▪', 'aab', 'aaa▪', 'aaab']

    """
#    assert eos not in V and eos not in context
    # enumerate all the ways to make the prefix inconsistent with the context by
    # changing it by one character or EOS
    for p in range(len(context)):
        y = context[:p]     # proper prefixes only
        yield y, eos
        for a in V:
            #assert prefix(y + a, context) == (context[p] == a)
            if context[p] != a:
                yield y, a


class Chart(dict):
    def __init__(self, zero, vals=()):
        self.zero = zero
        super().__init__(vals)

    def __missing__(self, k):
        return self.zero

    def spawn(self):
        return Chart(self.zero)

    def __add__(self, other):
        new = self.spawn()
        for k, v in self.items():
            new[k] += v
        for k, v in other.items():
            new[k] += v
        return new

    def __mul__(self, other):
        new = self.spawn()
        for k in self:
            v = self[k] * other[k]
            if v == self.zero:
                continue
            new[k] += v
        return new

    def copy(self):
        return Chart(self.zero, self)

    def trim(self):
        return Chart(
            self.zero, {k: v for k, v in self.items() if v != self.zero}
        )

    def metric(self, other):
        assert isinstance(other, Chart)
        err = 0
        for x in self.keys() | other.keys():
            err = max(err, abs(self[x] - other[x]))
        return err

    def _repr_html_(self):
        return (
            '<div style="font-family: Monospace;">'
            + format_table(self.trim().items(), headings=['key', 'value'])
            + '</div>'
        )

    def __repr__(self):
        return repr({k: v for k, v in self.items() if v != self.zero})

    def __str__(self, style_value=lambda k, v: str(v)):
        def key(k):
            return -self[k]

        return (
            'Chart {\n'
            + '\n'.join(
                f'  {k!r}: {style_value(k, self[k])},'
                for k in sorted(self, key=key)
                if self[k] != self.zero
            )
            + '\n}'
        )

    def assert_equal(self, want, *, domain=None, tol=1e-5, verbose=False, throw=True):
        if not isinstance(want, Chart):
            want = Chart(self.zero, want)
        if domain is None:
            domain = self.keys() | want.keys()
        assert verbose or throw
        errors = []
        for x in domain:
            if abs(self[x] - want[x]) <= tol:
                if verbose:
                    print(colors.mark(True), x, self[x])
            else:
                if verbose:
                    print(colors.mark(False), x, self[x], want[x])
                errors.append(x)
        if throw:
            for x in errors:
                raise AssertionError(f'{x}: {self[x]} {want[x]}')

    def argmax(self):
        return max(self, key=self.__getitem__)

    def argmin(self):
        return min(self, key=self.__getitem__)

    def top(self, k):
        return Chart(
            self.zero,
            {k: self[k] for k in sorted(self, key=self.__getitem__, reverse=True)[:k]},
        )

    def max(self):
        return max(self.values())

    def min(self):
        return min(self.values())

    def sum(self):
        return sum(self.values())

    def sort(self, **kwargs):
        return Chart(self.zero, [(k, self[k]) for k in sorted(self, **kwargs)])

    def sort_descending(self):
        return Chart(self.zero, [(k, self[k]) for k in sorted(self, key=lambda k: -self[k])])

    def normalize(self):
        Z = self.sum()
        if Z == 0:
            return self
        return Chart(self.zero, [(k, v / Z) for k, v in self.items()])

    def filter(self, f):
        return Chart(self.zero, [(k, v) for k, v in self.items() if f(k)])

    def map_values(self, f):
        return Chart(f(self.zero), [(k, f(v)) for k, v in self.items()])

    def map_keys(self, f):
        return Chart(self.zero, [(f(k), v) for k, v in self.items()])

    def project(self, f):
        "Apply the function `f` to each key; summing when f-transformed keys overlap."
        out = self.spawn()
        for k, v in self.items():
            out[f(k)] += v
        return out

    # TODO: the more general version of this method is join
    def compare(self, other, *, domain=None):
        if not isinstance(other, Chart):
            other = Chart(self.zero, other)
        if domain is None:
            domain = self.keys() | other.keys()
        rows = []
        for x in domain:
            m = abs(self[x] - other[x])
            rows.append(dict(key=x, self=self[x], other=other[x], metric=m))
        return pd.DataFrame(rows)


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


class max_munch:
    def __init__(self, tokens):
        self._end = object()
        self.root = self.make_trie(tokens)

    def __call__(self, x):
        if len(x) == 0:
            return ()
        else:
            t, ys = self.munch(x)
            return (ys,) + self(x[t:])

    def munch(self, x):
        (t, ys) = next(self.traverse(x, 0, self.root))
        return (t, ys)

    def make_trie(self, words):
        root = dict()
        for word in words:
            curr = root
            for letter in word:
                curr = curr.setdefault(letter, {})
            curr[self._end] = self._end
        return root

    def traverse(self, query, t, node):
        """
        Enumerate (in order of longest to shortest) the strings in the trie matching
        prefixes of `query`.
        """
        if node == self._end:
            return
        if t < len(query):
            x = query[t]
            if x in node:
                yield from self.traverse(query, t + 1, node[x])
        if self._end in node:
            yield (t, query[:t])  # post order gives the longest match


def color_code_alignment(seq1, seq2):
    colored_seq1, colored_seq2 = format_alignment(seq1, seq2)
    print("Sequence 1:")
    print(colored_seq1)
    print("Sequence 2:")
    print(colored_seq2)


def format_alignment(seq1, seq2):
    import Levenshtein as lev
    alignment = lev.editops(seq1, seq2)
    colored_seq1 = []
    colored_seq2 = []
    seq1 = [f'{x}|' for x in seq1]
    seq2 = [f'{x}|' for x in seq2]
    idx1, idx2 = 0, 0
    for op, i, j in alignment:
        while idx1 < i:
            colored_seq1.append(colors.green % seq1[idx1])
            idx1 += 1
        while idx2 < j:
            colored_seq2.append(colors.green % seq2[idx2])
            idx2 += 1
        if op == 'replace':
            colored_seq1.append(colors.red % seq1[idx1])
            colored_seq2.append(colors.red % seq2[idx2])
            idx1 += 1
            idx2 += 1
        elif op == 'insert':
            colored_seq2.append(colors.blue % seq2[idx2])
            idx2 += 1
        elif op == 'delete':
            colored_seq1.append(colors.yellow % seq1[idx1])
            idx1 += 1
    while idx1 < len(seq1):
        colored_seq1.append(colors.green % seq1[idx1])
        idx1 += 1
    while idx2 < len(seq2):
        colored_seq2.append(colors.green % seq2[idx2])
        idx2 += 1
    return ''.join(colored_seq1), ''.join(colored_seq2)


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


def format_table(rows, headings=None):
    def fmt(x):
        if isinstance(x, (SVG, HTML)):
            return x.data
        elif hasattr(x, '_repr_html_'):
            return x._repr_html_()
        elif hasattr(x, '_repr_svg_'):
            return x._repr_svg_()
        elif hasattr(x, '_repr_image_svg_xml'):
            return x._repr_image_svg_xml()
        else:
            return f'<pre>{html.escape(str(x))}</pre>'

    return (
        '<table>'
        + (
            '<tr style="font-weight: bold;">'
            + ''.join(f'<td>{x}</td>' for x in headings)
            + '</tr>'
            if headings
            else ''
        )
        + ''.join(
            '<tr>' + ''.join(f'<td>{fmt(x)}</td>' for x in row) + ' </tr>' for row in rows
        )
        + '</table>'
    )


def display_table(*args, **kwargs):
    return display(HTML(format_table(*args, **kwargs)))


# Merge step to compare and display both lists
def merge_and_compare(reference, approx):

    from rich.console import Console
    from rich.table import Table
    from rich.text import Text

    # Initialize console
    console = Console()

    table = Table(show_lines=True)

    # Define columns
    table.add_column("Want", justify="left")
    table.add_column("Status", justify="center")
    table.add_column("Have", justify="left")

    i, j = 0, 0
    while i < len(reference) and j < len(approx):
        ref_key, ref_value = reference[i]
        appr_key, appr_value = approx[j]

        if ref_key == appr_key:
            # Both lists have the same key at this point
            table.add_row(
                Text(f"{ref_key}: {ref_value}", style=""),
                Text("✓", style="green"),
                Text(f"{appr_key}: {appr_value}", style="")
            )
            i += 1
            j += 1
        elif ref_key < appr_key:
            # Element in reference list but not in approx list
            table.add_row(
                Text(f"{ref_key}: {ref_value}", style="bold yellow"),
                Text("Missing", style="bold red"),
                ""
            )
            i += 1
        else:
            # Element in approx list but not in reference list
            table.add_row(
                "",
                Text("Extra", style="bold red"),
                Text(f"{appr_key}: {appr_value}", style="blue")
            )
            j += 1

    # Handle remaining elements in either list
    while i < len(reference):
        ref_key, ref_value = reference[i]
        table.add_row(
            Text(f"{ref_key}: {ref_value}", style="bold yellow"),
            Text("Missing", style="bold red"),
            ""
        )
        i += 1

    while j < len(approx):
        appr_key, appr_value = approx[j]
        table.add_row(
            "",
            Text("Extra", style="bold red"),
            Text(f"{appr_key}: {appr_value}", style="blue")
        )
        j += 1

    # Render the table in the terminal
    console.print(table)


# TODO: pad the last row so it doesn't look weird compared to the others.
def plot_surprisals(context, surprisals, batch_size=75):

    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")

    assert len(context) == len(surprisals)
    #N = len(surprisals)
    #T = batch_size

    context = np.array([escape(x) for x in context])
    surprisals = np.array(surprisals)
    for B in batch(batch_size, range(len(context))):

        fig = pl.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)

        sns.barplot(surprisals[B], ax=ax)

        #ax.set_title(repr(context))
        ax.set_xticks(range(len(context[B])))
        ax.set_xticklabels(list(context[B]))
        ax.set_ylabel('suprisal')

        sns.despine()


def plot_surprisals_paired(context, surprisals1, surprisals2, batch_size=75):
    sns.set_theme(style="whitegrid")
    sns.set_palette("pastel")

    # Ensure the number of tokens in context matches the number of surprisals in both lists
    assert len(context) == len(surprisals1) == len(surprisals2)
    #N = len(surprisals1)
    #T = batch_size

    context = np.array([escape(x) for x in context])
    surprisals1 = np.array(surprisals1)
    surprisals2 = np.array(surprisals2)

    for B in batch(batch_size, range(len(context))):

#        x = np.arange(len(labels))  # Label locations
        width = 0.3  # Reduced bar width to better suit the wide figure size

        _, ax = pl.subplots(figsize=(12, 3))  # Keep your original wide figure size

        x = np.arange(len(context[B]))  # Bar positions
        values1 = surprisals1[B]
        values2 = surprisals2[B]

        # Plot the first set of bars
        ax.bar(x - width/2, values1, width, label='1', color='lightblue')

        # Plot the second set of bars
        ax.bar(x + width/2, values2, width, label='2', color='orange')

        # Add labels and formatting
        ax.set_ylabel('Values')
        ax.set_title('Surprisal')
        ax.set_ylabel('suprisal')
        ax.set_xticks(x)
        ax.set_xticklabels(list(context[B]))

        # Ensure the legend doesn't overlap with the plot
        ax.legend()

        # Use tight_layout to adjust padding and fit everything within the plot area
        pl.tight_layout()

        sns.despine()
        pl.show()

    print(f'Overall: {sum(surprisals1):.2f}, {sum(surprisals2):.2f}, {sum(surprisals2)/sum(surprisals1):.2f}x')


def compare_canonical(C, context, ys=None):
    lm = C.llm
    sep = b'' if lm.byte_level else ''
    ys = lm.encode_prompt(context) if ys is None else ys

    lm_curve = []
    for x in prefixes(flatten(ys)):
        if len(x) == 0:
            continue
        elif len(x) == len(ys):
            lm_curve.append(-lm.logp_next(unflatten(x))[lm.eos])
        else:
            lm_curve += [-lm.logp_next(unflatten(x[:-1]))[x[-1]] / len(x[-1])] * len(x[-1])

    C_curve = []
    for x in prefixes(flatten(ys)):
        if len(x) == 0:
            continue
        elif len(x) == len(ys):
            C_curve.append(-C.logp_next(sep.join(x))[C.eos])
        else:
            C_curve += [-C.logp_next_seq(sep.join(x[:-1]), x[-1]) / len(x[-1])] * len(x[-1])

    curve = []
    for x in prefixes(context):
        if len(x) == 0:
            continue
        elif len(x) == len(context):
            curve.append(-C.logp_next(x)[C.eos])
        else:
            curve.append(-C.logp_next(x[:-1])[x[-1]])

    pl.plot(C_curve, label='char', alpha=0.5)
    pl.plot(lm_curve, label='token', alpha=0.5)
    pl.scatter(range(len(context)), curve, alpha=0.5)
    pl.legend()

    print('canonical llh:', sum(lm_curve))
    print('character llh:', sum(curve))

    #ax.set_title(repr(context))
    ax = pl.gca()
    ax.set_xticks(range(len(context)))
    if lm.byte_level:
        ax.set_xticklabels([escape(bytes([i])) for i in context])
    else:
        ax.set_xticklabels(list(map(escape, context)))
    ax.set_ylabel('suprisal')

    sns.despine()


class LocalLeakageInteractive:
    def __init__(self, local, reference, steps=None):
        self.local = local
        if steps is None:
            steps = [local.fancy_step([local.lm._encode[y] for y in ys])
                     for ys in prefixes(reference)]

        T = len(steps)
        logps = []; logqs = []; context = []
        for t in range(T):
            if t+1 < T:
                next_token = steps[t+1].prompt_bytes[-1]
            else:
                next_token =  local.lm.eos
            context.append(next_token)
            logqs.append(steps[t].new_logp_next[next_token])
            logps.append(steps[t].raw_logp_next[next_token])

        self.logps = np.array(logps)
        self.logqs = np.array(logqs)
        self.context = np.array(context)
        self.context_str = np.array([escape(y) for y in context])
        self.T = T
        self.steps = steps
        self.df = pd.DataFrame(dict(
            logps=logps,
            logqs=logqs,
            context=context,
            contex_str=self.context_str,
            leak=[step.total_blocked for step in steps],
            warp=[step.total_allowed for step in steps],
            num_blocked=[len(step.blocked) for step in steps],
        ))

    def show_ranks(self):
        # Create some sample figures but do NOT display them automatically
        figures = []

        for s in self.steps:
            fig, ax = pl.subplots()
            ax.set_title(s.prompt_bytes)
            s.show_ranks(ax=ax)
            figures.append(fig)
            pl.grid(False)
            pl.close(fig)  # Critical: Close the figure to prevent auto-display

        def display_function(x):
            x.canvas.draw()
            display(x)

        return show_steps(display_function, figures)

    def show_local_leakage(self, ax=None):
        if ax is None: ax = pl.figure(figsize=(18,4)).add_subplot(111)
        #ax.plot([np.exp(s.total) for s in self.steps], marker='x')
        sns.barplot([np.exp(s.total_blocked) for s in self.steps], ax=ax)
        ax.semilogy()
        ax.set_xticks(range(self.T))
        ax.set_xticklabels(self.context_str, rotation=90)
        ax.set_ylabel('local leakage')
        ax.grid(False)

    def show_local_warp(self, ax=None):
        if ax is None: ax = pl.figure(figsize=(18,4)).add_subplot(111)
        #ax.plot([np.exp(s.total_allowed) for s in self.steps], marker='x')
        sns.barplot(np.exp(self.df.warp), ax=ax)
        ax.semilogy()
        ax.set_xticks(range(self.T))
        ax.set_xticklabels(self.context_str, rotation=90)
        ax.set_ylabel('local warp')
        ax.grid(False)

    def show_pq_surprisals(self, ax=None, width=0.3):
        if ax is None: ax = pl.figure(figsize=(18,4)).add_subplot(111)

        x = np.arange(self.T)

        ax.bar(x - width/2, -self.logps, width, color='b', label='-log p')
        ax.bar(x + width/2, -self.logqs, width, color='r', label='-log q')

        ax.set_ylabel('suprisal')
        ax.set_xticks(x)
        ax.set_xticklabels(self.context_str, rotation=90)

        maxp = max(-self.logps[np.isfinite(self.logps)])
        maxq = max(-self.logqs[np.isfinite(self.logqs)])
        ymax = max(maxp, maxq) * 1.5  # pylint: disable=W3301
        for i, (p, q) in enumerate(zip(self.logps, self.logqs)):
            if not np.isfinite(p):
                ax.text(i, ymax, "∞", ha='center', va='bottom', fontsize=12)
                ax.bar(i - width/2, ymax, width, color='b')
            if not np.isfinite(q):
                ax.text(i, ymax, "∞", ha='center', va='bottom', fontsize=12)
                ax.bar(i + width/2, ymax, width, color='r')

        # Ensure the legend doesn't overlap with the plot
        ax.legend()

        # Use tight_layout to adjust padding and fit everything within the plot area
        pl.tight_layout()

        sns.despine()
        ax.grid(False)

    def show_step_info(self):
        return show_steps(self._show_step_info, self.steps)

    def _show_step_info(self, step):
        t = len(step.prompt_bytes)
        next_token = self.steps[t+1].prompt_bytes[-1]
        print('step:', t)
        print('next_token:', next_token)
        print(step.prompt_bytes[:], colors.bold % '_____')
        top_blocked = step.blocked[:5]
        top_allowed = step.allowed[:5]
        if next_token in step.blocked:
            top_blocked.append(next_token)
        if next_token in step.allowed:
            top_allowed.append(next_token)
        print(colors.light.red % f'{step.total_blocked:.3f}', f'{np.exp(step.total_blocked) * 100:.4f}%')
        for rank, x in enumerate(step.raw_logp_next.top(None)):
            if x in top_blocked:
                pct = np.exp(step.raw_logp_next[x] - step.total_blocked) * 100  # percentage of leak
                print(
                    f'{rank:6d}', (colors.light.red % colors.xmark),
                    f'{step.raw_logp_next[x]:6.2f}',
                    colors.light.red % f'{pct:6.2f}%',
                    f' {repr(x)}',
                    (colors.light.red % '<==== 💀') if x == next_token else ''
                )
            if x in top_allowed:
                print(
                    f'{rank:6d}',
                    (colors.light.green % colors.check),
                    f'{step.raw_logp_next[x]:6.2f}',
                    '       ',
                    f' {repr(x):20s}',
                    colors.light.green % '<====' if x == next_token else ''
                )

    def show_scatter(self):
        from arsenal.maths import compare
        compare(self.logps, self.logqs).show()
        #pl.scatter(self.logps, self.logqs)
        #pl.plot([self.logps.min(), self.logps.max()], [self.logps.min(), self.logps.max()], c='r')
        #pl.xlabel('log p')
        #pl.ylabel('log q')


class show_steps:
    def __init__(self, display_function, steps):
        from ipywidgets import Button, VBox, HBox, Output

        # Output widget to display one figure at a time
        self.steps = steps
        self.display_function = display_function
        self.output = Output()
        self.T = len(steps)

        # Index tracker
        self.current_index = 0

        # Create buttons
        self.btn_next = Button(description="Next")
        self.btn_prev = Button(description="Prev")

        # Assign callbacks
        self.btn_next.on_click(self.on_next_click)
        self.btn_prev.on_click(self.on_prev_click)

        # Initial display
        self.update_figure(self.current_index)

        # Arrange buttons and output widget
        self.navigation = HBox([self.btn_prev, self.btn_next])
        display(VBox([self.navigation, self.output]))

    # Function to update the displayed figure
    def update_figure(self, index):
        if 0 <= self.current_index < self.T:
            with self.output:
                self.current_index = index
                self.output.clear_output(wait=True)
                self.display_function(self.steps[self.current_index])

    def on_next_click(self, _):
        self.update_figure(self.current_index + 1)

    def on_prev_click(self, _):
        self.update_figure(self.current_index - 1)

    def _repr_html_(self):
        return
