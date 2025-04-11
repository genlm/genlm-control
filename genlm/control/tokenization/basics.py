import numpy as np
from collections import defaultdict
from genlm.control.tokenization.util import flatten, logsumexp, Chart


class Item:

    __slots__ = ('ps', 'ys', 'offset', 'parent')

    def __init__(self, ps, ys, offset, parent):
        self.ps = ps
        self.ys = ys
        self.offset = offset
        self.parent = parent

    def __repr__(self):
        return f'Item(ps={self.ps!r}, ys={self.ys!r})'

    def __eq__(self, other):
        return self.ps == other.ps and self.ys == other.ys

    def __hash__(self):
        return hash(self.ys)

    def __lt__(self, other):
        return self.ps < other.ps
class BeamItem(Item):
    def __init__(
        self,
        context,
        tokens,
        score,
        completed=False,
        stop_reason=None,
        pruned=False,
        debug=None,
        canonical=None,
        offset=None,
        parent=None,
    ):
        # If no offset is provided, default to the current token length.
        if offset is None:
            offset = len(tokens)
        # Initialize the base Item with:
        #   ps: the cumulative score,
        #   ys: the sequence of tokens,
        #   offset: (here, the current token length),
        #   parent: pointer to the parent BeamItem.
        super().__init__(ps=score, ys=tokens, offset=offset, parent=parent)
        self.context = context
        self.completed = completed
        self.stop_reason = stop_reason
        self.pruned = pruned
        self.debug = debug if debug is not None else []
        self.canonical = canonical

    def __repr__(self):
        return (f"BeamItem(tokens={self.ys!r}, score={self.ps!r}, "
                f"completed={self.completed}, canonical={self.canonical})")

class Beam(list):

    def __init__(self, *args):
        super().__init__(*args)
        self.parent = None

    @property
    def total(self):
        return self.logsumexp()

    def logsumexp(self):
        return logsumexp([x.ps for x in self])

    @property
    def fsa(self):
        from fsa import FSA
        tmp = FSA()
        for item in self:
            tmp += FSA.from_string(flatten(item.ys))
        return tmp.min()

    def __getitem__(self, arg):
        val = super().__getitem__(arg)
        return Beam(val) if isinstance(arg, slice) else val

    def sort(self):
        return Beam(sorted(self, key = lambda item: -item.ps))

    def top(self, K):
        return Beam(self.sort()[:K])

    def argmax(self):
        return max(self, key = lambda item: -item.ps)

    def __repr__(self):
        return '%s (%r/%s) {\n%s\n}' % (
            self.__class__.__name__,
            getattr(self, 'key', None),
            logsumexp([x.ps for x in self]) if len(self) > 0 else -np.inf,
            '\n'.join(f'  {x},' for x in self)
        )

    def depth(self):
        return max([len(item.ys) for item in self], default=0)

    def groupby(self, key):
        groups = defaultdict(Beam)
        for item in self:
            groups[key(item)].append(item)
        return groups

    def backup(self, qs):
        C = set()
        N = len(qs)
        for item in self:
            #if qs == ''.join(item.ys):  # no need to backup a perfect cover without any slop
            if N == item.offset:
                C.add(item)
            else:
                C.add(item.parent)
        return Beam(C)

    def to_latex(self, context, encode, fmt_p=lambda p: f'{p:.1f}'):
        N = len(context)
        space = r"{\textvisiblespace}"

        def fmt_cover(item):
            ys = flatten(item.ys)

            tokens = [
                fr'\tokenfont[{encode(x)}]{{{x.replace(" ", space)}}}'
                for x in ys
            ]
            x = ys[-1]
            matched = x[:N - item.parent.offset]
            unmatched = x[N - item.parent.offset:]
            fmt_x = matched + fr'\underline{{{unmatched}}}'
            fmt_x = fmt_x.replace(" ", space)

            tokens[-1] = fr'\tokenfont[{encode(x)}]{{{fmt_x}}}'

            return '[%s]' % (', '.join(tokens))


        lines = [r'\begin{align*}']
        for item in self:
            lines.append(fr'{fmt_p(item.ps)} &: {fmt_cover(item)} \\')
        lines.append(r'\end{align*}')

        return '\n'.join(lines)

    def compare(self, other):
        from tokenization.util import merge_and_compare
        # Call the function to display the merge comparison
        merge_and_compare([(flatten(x.ys), x.ps) for x in self],
                          [(flatten(x.ys), x.ps) for x in other])

    def to_chart(self):
        c = Chart(None)
        for p in self:
            c[p.ys] = p.ps
        return c
