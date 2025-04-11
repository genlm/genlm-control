import numpy as np
from collections import defaultdict
from genlm.control.tokenization.util import MyTree, decode_hf_tokenizer, flatten
from llist import dllist   # pylint: disable=E0611

from arsenal import Integerizer
from arsenal.datastructures.heap import LocatorMaxHeap
from collections import namedtuple
Value = namedtuple('Value', 'token_id, derivation')


class BPE:

    def __init__(self, _merges, _encode, _decode, _encode_byte):
        self._encode_byte = _encode_byte

        self._parent = {(u, v): uv for u, v, uv in _merges}
        self._merges = _merges
        self._encode = _encode
        self._decode = _decode
        self.V = len(_decode)          # token vocabulary size

        self._parent_l = {l: {} for l in range(self.V)}
        for u, v, uv in _merges:
            self._parent_l[u][v] = uv

        self.priority = {(u,v): -i for i, (u,v,_) in enumerate(self._merges)}
        self.make_derivation_table()
        self.__left_spine = self._left_spine_table()
        self.__right_spine = self._right_spine_table()
        self.overrides = {}

    @classmethod
    def from_merge_list(cls, merges):
        alphabet = Integerizer()

        # make the byte encoder the identity mapping
        _byte_encoder = [None]*256
        for i in range(256):
            assert alphabet(bytes([i])) == i
            _byte_encoder[i] = i

        _merges = []
        for u,v,uv in merges:
            _merges.append((alphabet(u) if len(u) > 1 else u[0],
                            alphabet(v) if len(v) > 1 else v[0],
                            alphabet(uv)))

        _encode = alphabet._map
        _decode = alphabet._list

        return cls(_merges, _encode, _decode, _byte_encoder)

    @classmethod
    def from_huggingface(cls, tokenizer):
        "Extract what we need from a 🤗 tokenizer."
        return cls(*decode_hf_tokenizer(tokenizer))

#    def build_tree_table(self):
#        trees = defaultdict(list)
#        for (u,v,uv) in _merges:   # NOTE: already ordered
#            u = _decode[u]
#            v = _decode[v]
#            uv = _decode[uv]
#            trees[uv].append((u, v))
#        assert all(len(ts) == 1 for ts in trees.values())
#        base = {bytes([x]) for y in _decode for x in y}
#        assert len(base) == 256
#        for u in base:
#            trees[u] = [u]
#        self.trees = dict(trees)

    def is_unambiguous(self, verbosity=0):
        unambig = True
        defined = set(self._encode_byte)
        for u,v,uv in self._merges:
            if uv in defined:
                unambig = False
                if verbosity > 0:
                    U, V, UV = self._decode[u], self._decode[v], self._decode[uv]
                    print('ambiguous', [u, v, uv], [U, V, UV])
            defined.add(uv)
        return unambig

    def is_proper(self, verbosity=0):
        proper = True
        defined = set(self._encode_byte)
        for u,v,uv in self._merges:
            if not (u in defined and v in defined):
                proper = False
                if verbosity > 0:
                    U, V, UV = self._decode[u], self._decode[v], self._decode[uv]
                    print('improper', [u, v, uv], [U, V, UV])
            defined.add(uv)
        return proper

#    # XXX: experimental; do not use.
#    def make_proper_and_unambiguous(self, proper=True, unambiguous=True):
#        # force the merge rules to be proper and unambiguous
#        defined = set(self._encode_byte)
#        _revised_merges = []
#        for u,v,uv in self._merges:
#            if unambiguous and uv in defined and (u in defined and v in defined):  # drop ambiguous
#                continue
#            if proper and not (u in defined and v in defined):  # drop any improper rule
#                continue
#            defined.add(uv)
#            _revised_merges.append((u,v,uv))
#        # TODO: does encode work correctly?
#        return BPE(_revised_merges, self._encode, self._decode, self._encode_byte)

    def _left_spine_table(self):
        "Closure of the left tables."
        left_spine = [None]*self.V
        left = self._left
        for i in range(self.V):
            spine = [np.inf, i]
            x = i
            while True:
                x = left[x]
                if x is None: break
                spine.append(x)
            spine.reverse()
            left_spine[i] = spine
        return left_spine

    def _right_spine_table(self):
        "Closure of the right tables."
        right_spine = [None]*self.V
        right = self._right
        for i in range(self.V):
            spine = [np.inf, i]
            x = i
            while True:
                x = right[x]
                if x is None: break
                spine.append(x)
            spine.reverse()
            right_spine[i] = spine
        return right_spine
    
    def encode_prompt(self, prompt: str) -> tuple[str, ...]:
        return unflatten(self.encode_as_byte_chunks(prompt.encode()))
    def nested_tuple_to_byte_chunks(self, s: tuple[str, ...]) -> list[bytes]: #takes nested tuple
        return [i.encode() for i in flatten(s)]
    def nested_tuple_to_token_ids(self, s: tuple[str, ...]) -> list[int]: #takes nested tuple
        return self.byte_chunks_to_token_ids(self.nested_tuple_to_byte_chunks(s))
    def decode_from_byte_chunks(self, ys: list[bytes]) -> bytes:
        return b''.join(ys)

    def decode_from_token_ids(self, token_ids: list[int]) -> bytes:
        return self.decode_from_byte_chunks(self.token_ids_to_byte_chunks(token_ids))

    def token_ids_to_byte_chunks(self, token_ids: list[int]) -> list[bytes]:
        return [self._decode[i] for i in token_ids]

    def byte_chunks_to_token_ids(self, byte_chunks: list[bytes]) -> list[int]:
        return [self._encode[y] for y in byte_chunks]

    def canonicalize_token_ids(self, token_ids: list[int]):
        return self.encode_as_token_ids(self.decode_from_token_ids(token_ids))

    def canonicalize_byte_chunks(self, chunks: list[bytes]):
        return self.encode_as_byte_chunks(self.decode_from_byte_chunks(chunks))

    @classmethod
    def load_model_by_name(cls, model_name: str):
        import transformers
        return cls.from_huggingface(transformers.AutoTokenizer.from_pretrained(model_name, use_fast=False))

    def encode_as_byte_chunks(self, x: bytes) -> list[bytes]:
        return self.token_ids_to_byte_chunks(self.encode_as_token_ids(x))

    def encode_as_token_ids(self, x: bytes) -> list[int]:
        assert isinstance(x, bytes)

        # Convert bytes to initial token IDs
        x = [self._encode_byte[i] for i in x]
        token_list = dllist(x)

        # Dictionary to track pairs and their positions
        pair_positions = defaultdict(list)
        current = token_list.first
        while current and current.next:
            pair = (current.value, current.next.value)
            pair_positions[pair].append(current)
            current = current.next

        # Apply each merge rule

        # TODO: this loop is inefficient because we might loop over merges that
        # do not appear in `x`.  We could use an agenda that is priortized by
        # the merge order as an alternative.
        for u, v, uv in self._merges:
            pair = (u, v)
            # TODO: use an agenda containing pair_positions, which is ordered by
            # the merge rule rank -- this is worked out in `fast_encode_with_derivation`.
            if pair not in pair_positions:
                continue

            for node in list(pair_positions[pair]):  # Use a copy of the list to avoid modification during iteration
                if not node.next or node.value != u or node.next.value != v:
                    continue  # Skip invalidated pairs

                # Merge (u, v) into uv
                node.value = uv
                token_list.remove(node.next)

                # Update neighbors
                if node.prev:
                    prev_pair = (node.prev.value, u)
                    new_prev_pair = (node.prev.value, uv)
                    if node.prev in pair_positions[prev_pair]:      # XXX: uh oh, this is linear time
                        pair_positions[prev_pair].remove(node.prev)
                    pair_positions[new_prev_pair].append(node.prev)

                if node.next:
                    next_pair = (v, node.next.value)
                    new_next_pair = (uv, node.next.value)
                    if node in pair_positions[next_pair]:       # XXX: uh oh, this is linear time
                        pair_positions[next_pair].remove(node)
                    pair_positions[new_next_pair].append(node)

            # Clear positions for the merged pair
            del pair_positions[pair]

        return list(token_list)

    # This method is very slow, it implements what Berglund calls the "SentencePiece" version of BPE
    def encode2(self, xs, callback=None) -> list[int]:
        "Rewrite until fixpoint; starting at a valid token sequence (or bytes)."
        if isinstance(xs, bytes): xs = [self._encode_byte[i] for i in xs]
        if callback is not None: callback(xs)
        while True:
            ys = self._rewrite_by_highest_priority_rule(xs)
            if ys == xs: break
            xs = ys
            if callback is not None: callback(xs)
        return xs

    def _rewrite_by_highest_priority_rule(self, x):
        for (u, v, uv) in self._merges:
            for t in range(len(x)-1):
                if (x[t], x[t+1]) == (u, v):
                    return x[:t] + [uv] + x[t+2:]
        return x

    def canonical_tree(self, p) -> MyTree:
        "Return the nested merge-pair representation of `p` (`bytes` or `int`)"
        if isinstance(p, bytes): p = self._encode[p]
        assert isinstance(p, int)
        if self._left[p] is None:
            return self._decode[p]
        else:
            return MyTree(self.canonical_tree(self._left[p]),
                          self.canonical_tree(self._right[p]))

    def is_canonical(self, token_ids: list[int]) -> bool:
        for i in range(1, len(token_ids)):
            if not self._incremental_canonicality(token_ids[i-1], token_ids[i]):
                return False
        return True

    def is_canonical_incremental(self, canonical_prefix: list[int], next_token_id: int) -> bool:
        """
        Efficiently determine if canonical_prefix + [next_token_id] is canonical.
        This method assumes `canonical_prefix` is a canonical seqeunce of token ids.
        """
        if len(canonical_prefix) == 0:   # every length-one token string is canonical
            return True
        else:
            return self._incremental_canonicality(canonical_prefix[-1], next_token_id)

    def _incremental_canonicality(self, left: int, right: int):
        "Checks whether the pair (left, right) is canonical."
        if left in self._noncanonical_token_ids: return False
        if right in self._noncanonical_token_ids: return False
        if right in self.overrides.get(left, ()): return True
        return (self._find_conflict(left, right) is None)  # no conflicting tree ⟺ canonical

    def _find_conflict(self, left: int, right: int) -> tuple:
        "Search for a conflicting tree with (left, right)."
        l = left
        L = np.inf
        while True:

            r = right
            R = np.inf
            while True:

                k = self._parent.get((l, r), np.inf)      # possible merge of l and r
                if k <= R and k < L:
                    # Note: The condition `k < lp` is a strict inequality (unlike `k <= rp`) so
                    # that the duplicate merge (`lp == k`) associates left as toes in BPE.
                    return (l, r)

                R = r
                r = self._left[r]
                if r is None: break

            L = l
            l = self._right[l]
            if l is None: break

    # This method is very slow
#    def _find_conflict_outer_loop(self, left: int, right: int) -> tuple:
#        "Search for a conflicting tree with (left, right)."
#        L = self._right_spine(left)
#        R = self._left_spine(right)
#        for (l,r), k in self._parent.items():
#            if r in R and l in L and k <= R[r] and k < L[l]:
#                return (l, r)

#    def _find_conflict2(self, left: int, right: int) -> tuple:
#        "Search for a conflicting tree with (left, right)."
#
#        spine_left = self.__right_spine[left]
#        spine_right = self.__left_spine[right]
#
#        L = len(spine_left) - 1    # inf padding
#        R = len(spine_right) - 1
#
#        # Cross product of left and right spines;
#        # the parent rank invariant can be violated by either the left or the right subtree.
#        for j in range(R):
#            r = spine_right[j]
#            rp = spine_right[j+1]     # r's parent in the right subtree
#            for i in range(L):
#                l = spine_left[i]
#                k = self._parent.get((l, r), np.inf)      # possible merge of l and r
#                if k < np.inf and k <= rp:
#                    lp = spine_left[i+1]   # l's parent in the left subtree
#                    # Note: The condition `k < lp` is a strict inequality (unlike `k <= rp`) so
#                    # that the duplicate merge (`lp == k`) associates left as does in BPE.
#                    if k < lp:
#                        return (l,r)

    # this is 8x slower than the baseline method.
#    def _find_conflict2(self, left: int, right: int) -> tuple:
#        "Search for a conflicting tree with (left, right)."
#        spine_left = self.__right_spine[left]
#        spine_right = self.__left_spine[right]
#        L = len(spine_left) - 1    # inf padding
#        R = len(spine_right) - 1
#
#        # For each right spine element r and its parent rp
#        for j in range(R):
#            r = spine_right[j]
#            rp = spine_right[j+1]     # r's parent in the right subtree
#
#            # Vectorized operations on left spine
#            l_array = spine_left[:-1]  # all elements except last
#            lp_array = spine_left[1:]  # all parents
#
#            # Get all possible merge ranks for current r with all l's
#            k_array = np.array([self._parent.get((l, r), np.inf) for l in l_array])
#
#            # Find conflicts where k <= rp and k < lp
#            conflicts = np.where((k_array < np.inf) & (k_array <= rp) & (k_array < lp_array))[0]
#
#            if len(conflicts) > 0:
#                # Return first conflict found
#                i = conflicts[0]
#                return (spine_left[i], r)
#
#        return None  # No conflict found

    def conflicting_next_tokens(self, left: int):
        """
        return the set of all conflicting next tokens following `left`.  Cache for efficiency.
        """
        return self._conflicting_next_tokens2(left)

    def _conflicting_next_tokens1(self, left: int):
        spine_left = self.__right_spine[left]
        spine_rights = self.__left_spine

        L = len(spine_left) - 1    # inf padding

        V = len(self._decode)
        conflicts = {}

        for i in range(L):
            l = spine_left[i]
            lp = spine_left[i+1]   #

            par_l = self._parent_l[l]    # possible merges `(l,r)` given `l`.

            for index in range(V):

                R = len(spine_rights[index]) - 1

                for j in range(R):
                    r = spine_rights[index][j]
                    rp = spine_rights[index][j+1]     # r's parent in the right subtree

                    k = par_l.get(r, np.inf)      # possible merge of l and r
                    if k < np.inf and k <= rp and k < lp:
                        conflicts[index] = (l,r)
        return conflicts

    def _conflicting_next_tokens2(self, left: int):
        """
        Return the set of all conflicting next tokens following `left`. Cache for efficiency.
        """
        spine_left = self.__right_spine[left]
        spine_rights = self.__left_spine

        L = len(spine_left) - 1  # inf padding
        V = len(self._decode)
        conflicts = {}

        for i in range(L):
            l = spine_left[i]
            lp = spine_left[i + 1]

            # Precompute parent relationships for l
            par_l = self._parent_l[l]  # possible merges `(l,r)` given `l`

            for index in range(V):
                spine_right = spine_rights[index]
                R = len(spine_right) - 1

                # Traverse spine_right once
                for j in range(R):
                    r = spine_right[j]
                    rp = spine_right[j + 1]

                    # Direct check for conflicts
                    if r in par_l:
                        k = par_l[r]
                        if k <= rp and k < lp:
                            conflicts[index] = (l, r)
                            break

        return conflicts

#    def _incremental_canonicality(self, left, right):
#        """
#        Fast implementation of `self.canonicalize_token_ids([left, right]) == [left, right]`.
#        """
#        #assert isinstance(left, int) and isinstance(right, int), [left, right, type(left), type(right)]
#        #return self.canonicalize_token_ids([left, right]) == [left, right]
#
#        spine_left = self._right_spine[left]
#        spine_right = self._left_spine[right]
#
#        L = len(spine_left) - 1    # inf padding
#        R = len(spine_right) - 1
#
#        # Cross product of left and right spines;
#        # the parent rank invariant can be violated by either the left or the right subtree.
#        for j in range(R):
#            r = spine_right[j]
#            rp = spine_right[j+1]     # r's parent in the right subtree
#            for i in range(L):
#                l = spine_left[i]
#                k = self._parent.get((l, r), np.inf)      # possible merge of l and r
#                if k < np.inf and k <= rp:
#                    lp = spine_left[i+1]   # l's parent in the left subtree
#                    # Note: The condition `k < lp` is a strict inequality (unlike `k <= rp`) so
#                    # that the duplicate merge (`lp == k`) associates left as toes in BPE.
#                    if k < lp:
#                        return False
#        return True

    #___________________________________________________________________________
    # Repair-based canonicalization algorithms

#    def repair_based_rewriting(self, context):
#        """
#        This algorithm implemented a naive fixpoint iteration procedure;
#        its efficiency can definitely be improved with semi-naive/agenda-based
#        change propagation.
#        """
#        while True:
#            changed = False
#            for t in range(1, len(context)):
#                x, y = context[t-1], context[t]
#                lr = self._find_conflict(x, y)
#                if lr is not None:
#                    l,r = lr
#                    context = context[:t-1] + self._surgery(x, y, l, r) + context[t+1:]
#                    changed = True
#                    break
#            if not changed:
#                return context

#    def _right_spine(self, left):
#        l = left
#        L = {l: np.inf}
#        while self._right[l] is not None:
#            L[self._right[l]] = l
#            l = self._right[l]
#        return L
#
#    def _left_spine(self, right):
#        r = right
#        R = {r: np.inf}
#        while self._left[r] is not None:
#            R[self._left[r]] = r
#            r = self._left[r]
#        return R
#
#    def _surgery(self, left, right, l, r):
#        C = []
#        # Tree surgery:
#        # 1) Flatten the spine of x until the conflict at `l`
#        ll = left
#        while ll != l:
#            C.append(self._left[ll])
#            ll = self._right[ll]
#        # 2) Apply the conflicting merge
#        C.append(self._parent[l,r])
#        # #) flatten the spine of y until the conflict at `r`
#        rr = right
#        rrs = []   # need to reverse these nodes
#        while rr != r:
#            rrs.append(self._right[rr])
#            rr = self._left[rr]
#        C.extend(reversed(rrs))
#        return C

    #___________________________________________________________________________
    # Alternative implementation of the `encode_as_token_ids`

    def rewrite(self, xs, callback=None):
        "Rewrite until fixpoint; starting at a valid token sequence."
        if isinstance(xs, bytes): xs = [self._encode_byte[i] for i in xs]
        if callback is not None: callback(xs)
        for (u, v, uv) in self._merges:
            while True:
                ys = self._rewrite(u, v, uv, xs)
                if ys == xs: break
                xs = ys
                if callback is not None: callback(xs)
        return xs

    def _rewrite(self, u, v, uv, x):
        # replace uv in x, taking the first match
        for t in range(len(x)-1):
            if (x[t], x[t+1]) == (u, v):
                return x[:t] + [uv] + x[t+2:]
        return x

    #___________________________________________________________________________
    # Alternative implementation of `repair_based_rewriting`

#    def repair_based_rewriting_ordered(self, context, callback=None):
#        if callback is not None: callback(context)
#        for l,r,k in self._merges:
#            while True:
#                new = self._repair_based_rewriting_ordered(context, l, r, k)
#                if new is None:
#                    break
#                context = new
#                if callback is not None: callback(context)
#        return context
#
#    def _repair_based_rewriting_ordered(self, context, l, r, k):
#        for t in range(1, len(context)):
#            left, right = context[t-1], context[t]
#            L = self._right_spine(left)
#            R = self._left_spine(right)
#            if r in R and l in L and k <= R[r] and k < L[l]:
#                return context[:t-1] + self._surgery(left, right, l, r) + context[t+1:]

    #___________________________________________________________________________
    #

    def fast_encode_with_derivation(self, x):
        assert isinstance(x, bytes)

        # Convert bytes to initial token IDs
        _x = x
        x = [self._encode_byte[i] for i in x]
        token_list = dllist([Value(i, bytes([j])) for i, j in zip(x, _x)])

        agenda = LocatorMaxHeap()

        # Dictionary to track pairs and their positions
        pair_positions = defaultdict(list)
        current = token_list.first
        while current and current.next:
            pair = (current.value.token_id, current.next.value.token_id)
            pair_positions[pair].append(current)
            current = current.next
            if pair in self.priority:
                agenda[pair] = self.priority[pair]

        # Apply each merge rule
        while agenda:
            pair, _ = agenda.pop()
            (u, v) = pair
            uv = self._parent[u,v]

            if pair not in pair_positions:
                continue

            for node in list(pair_positions[pair]):  # Use a copy of the list to avoid modification during iteration
                if not node.next or node.value.token_id != u or node.next.value.token_id != v:
                    continue  # Skip invalidated pairs

                # Merge (u, v) into uv
                node.value = Value(uv, MyTree(node.value.derivation, node.next.value.derivation))
                token_list.remove(node.next)

                # Update neighbors
                if node.prev:
                    prev_pair = (node.prev.value.token_id, u)
                    new_prev_pair = (node.prev.value.token_id, uv)
                    if node.prev in pair_positions[prev_pair]:      # XXX: uh oh, this is linear time
                        pair_positions[prev_pair].remove(node.prev)
                    pair_positions[new_prev_pair].append(node.prev)
                    if new_prev_pair in self.priority:
                        agenda[new_prev_pair] = self.priority[new_prev_pair]

                if node.next:
                    next_pair = (v, node.next.value.token_id)
                    new_next_pair = (uv, node.next.value.token_id)
                    if node in pair_positions[next_pair]:       # XXX: uh oh, this is linear time
                        pair_positions[next_pair].remove(node)
                    pair_positions[new_next_pair].append(node)
                    if new_next_pair in self.priority:
                        agenda[new_next_pair] = self.priority[new_next_pair]

            # Clear positions for the merged pair
            del pair_positions[pair]

        return list(token_list)

    # This method is very slow, it implements what Berglund calls the "SentencePiece" version of BPE
    def slow_encode_with_derivation(self, xs, callback=None) -> list[int]:
        "Rewrite until fixpoint; starting at a valid token sequence (or bytes)."
        assert isinstance(xs, bytes)
        ds = [bytes([x]) for x in xs]
        xs = [self._encode_byte[i] for i in xs]
        if callback is not None: callback(xs)
        while True:
            [ys, ds] = self._slow_encode_with_derivation(xs, ds)
            if ys == xs: return (xs, ds)
            xs = ys
            if callback is not None: callback(xs)

    def _slow_encode_with_derivation(self, xs, ds):
        for (u, v, uv) in self._merges:
            for t in range(len(xs)-1):
                if (xs[t], xs[t+1]) == (u, v):
                    return (
                        xs[:t] + [uv]                     + xs[t+2:],
                        ds[:t] + [MyTree(ds[t], ds[t+1])] + ds[t+2:],
                    )
        return xs, ds

    def make_derivation_table(self):
        self._noncanonical_token_ids = set()
        self._left = [None]*self.V
        self._right = [None]*self.V
        for x in self._decode:
            if x.startswith(b'<|'):
                self._noncanonical_token_ids.add(self._encode[x])
                continue   # skip special/added tokens
            # Note: Some tokens are never canonical, so we filter them below
            try:
                [(_, t)] = self.fast_encode_with_derivation(x)
            except ValueError:
                self._noncanonical_token_ids.add(self._encode[x])
            self._update_derivation_table(t)

    # TODO: we are doing more work than necessary because we are doing the
    # updates for subtree trees that we have already been done.  There is
    # probably a more bototm-up approach that will fill in the table more
    # efficiently. We can circle back later to figure that out.
    def _update_derivation_table(self, t):
        if isinstance(t, MyTree):
            l, r = t
            L = self._update_derivation_table(l)
            R = self._update_derivation_table(r)
            T = self._parent[L,R]
            # sanity check: clobbering should not happen if each token has a
            # canonical derivation.
            assert self._left[T] is None or self._left[T] == L
            assert self._right[T] is None or self._right[T] == R
            self._left[T] = L
            self._right[T] = R
            return T
        else:
            assert isinstance(t, bytes)
            return self._encode[t]
