import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from genlm.control.tokenization.util import MyTree, decode_hf_tokenizer
from llist import dllist
from collections import namedtuple

from arsenal.datastructures.heap import LocatorMaxHeap

Value = namedtuple('Value', 'token_id, derivation')
VERYLARGE = 10000000

class FastCanonicalityFilterBPE:

    def __init__(self, _merges, _encode, _decode, _encode_byte, eos_token_id):
        self._encode_byte = _encode_byte

        self._parent = {(u, v): uv for u, v, uv in _merges}
        self._merges = _merges
        self._encode = _encode
        self._decode = _decode
        self.V = len(_decode)          # token vocabulary size

        self.priority = {(u,v): -i for i, (u,v,_) in enumerate(self._merges)}
        self.make_derivation_table()

        self.__left_spine, max_left_spine_width = self._left_spine_table()
        self.__right_spine, max_right_spine_width = self._right_spine_table()

        self.left_spine_vector = self.vectorize_spine(self.__left_spine, max_left_spine_width)
        self.right_spine_vector = self.vectorize_spine(self.__right_spine, max_right_spine_width)

        self.indices = np.array([(index, j) for index in range(self.V)
                                 for j in range(len(self.__left_spine[index])-1)])

        self.vector_r = self.left_spine_vector[self.indices[:,0], self.indices[:,1]]
        self.vector_rp = self.left_spine_vector[self.indices[:,0], self.indices[:,1]+1]

        tmp = sp.dok_matrix((self.V, self.V), dtype=np.int32)
        for u, v, uv in _merges:
            tmp[u, v] = uv+1 # +1 to avoid zero-indexing

        self.parent_l_matrix = tmp.tocsr()
        self.parent_l_matrix = self.parent_l_matrix[:, self.vector_r]

        self.eos_token_id = eos_token_id
        self.overrides = defaultdict(lambda: set())

    def __call__(self, context):
        if context == ():
            mask = np.ones(self.V, dtype=bool)
        else:
            (_, last_token) = context
            mask = self._vectorized_conflicting_next_tokens2(self._encode[last_token])
        mask[self.eos_token_id] = True
        return mask

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
            left, right = t
            L = self._update_derivation_table(left)
            R = self._update_derivation_table(right)
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

    def vectorize_spine(self, spine, max_spine_width):
        new_spine = [
            [s[i] if i < len(s) else -VERYLARGE for i in range(max_spine_width)]
            for s in spine
        ]
        return np.array(new_spine, dtype=np.int32)

    def _left_spine_table(self):
        "Closure of the left tables."
        max_width = 0
        left_spine = [None]*self.V
        left = self._left
        for i in range(self.V):
            spine = [VERYLARGE, i]
            x = i
            while True:
                x = left[x]
                if x is None: 
                    break
                spine.append(x)
            spine.reverse()
            left_spine[i] = spine
            max_width = max(max_width, len(spine))
        return left_spine, max_width

    def _right_spine_table(self):
        "Closure of the right tables."
        max_width = 0
        right_spine = [None]*self.V
        right = self._right
        for i in range(self.V):
            spine = [VERYLARGE, i]
            x = i
            while True:
                x = right[x]
                if x is None: 
                    break
                spine.append(x)
            spine.reverse()
            right_spine[i] = spine
            max_width = max(max_width, len(spine))
        return right_spine, max_width

    def set_overrides(self, model_name):
        if "gpt2" in model_name:
            for (left, right) in [(198, 198), (2637, 82)]:
                self.overrides[left].add(right)
                print(f"adding override {self._decode[left]} <-> {self._decode[right]}")

    def _vectorized_conflicting_next_tokens(self, left: int):
        spine_left = self.__right_spine[left]

        L = len(spine_left) - 1    # inf padding
        conflicts = set()

        np_matrix = self.parent_l_matrix[spine_left[:L]].toarray()

        for i in range(L):
            lp = spine_left[i+1]

            vector_k = np_matrix[i]
            # convert 0 in vector_k to VERYLARGE
            vector_k = np.where(vector_k != 0, vector_k-1, VERYLARGE)

            conflict_mask = (vector_k < VERYLARGE)
            conflict_mask &= (vector_k <= self.vector_rp)
            conflict_mask &= (vector_k < lp)
            conflicts.update(self.indices[conflict_mask][:,0])
        conflicts.update(self.overrides[left])

        return conflicts

    def _vectorized_conflicting_next_tokens2(self, left: int):
        spine_left = self.__right_spine[left]

        L = len(spine_left) - 1    # inf padding

        mask = np.ones(self.V, dtype=bool)

        np_matrix = self.parent_l_matrix[spine_left[:L]].toarray()

        for i in range(L):
            lp = spine_left[i+1]

            vector_k = np_matrix[i]
            # convert 0 in vector_k to VERYLARGE
            vector_k = np.where(vector_k != 0, vector_k-1, VERYLARGE)

            conflict_mask = (vector_k < VERYLARGE)
            conflict_mask &= (vector_k <= self.vector_rp)
            conflict_mask &= (vector_k < lp)
            mask[self.indices[conflict_mask][:,0]] = False

        return mask

    @classmethod
    def from_huggingface(cls, tokenizer):
        "Extract what we need from a 🤗 tokenizer."
        return cls(*decode_hf_tokenizer(tokenizer), eos_token_id=tokenizer.eos_token_id)
