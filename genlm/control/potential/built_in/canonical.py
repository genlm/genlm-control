import numpy as np
from arsenal import colors
from llist import dllist
from genlm.control.potential.base import Potential
import scipy.sparse as sp
from collections import namedtuple
from collections import defaultdict
from arsenal.datastructures.heap import LocatorMaxHeap

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

def escape(x):
    if isinstance(x, int):   # assume its a byte
        x = bytes([x])
    if isinstance(x, bytes):
        y = repr(x)[2:-1]
    else:
        y = repr(x)[1:-1]
    return y.replace(" ","␣")

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

class CanonicalTokenization(Potential):
    """
    A custom potential that enforces canonical BPE tokenization.
    
    This potential ensures that tokens follow the canonical tokenization rules
    by using the FastCanonicalityFilterBPE under the hood.
    """
    
    def __init__(self, tokenizer, model_name): 
        """
        Initialize the Canonical Potential
        
        Args:
            tokenizer: The HuggingFace tokenizer to use
            model_name: The name of the model (used for setting overrides)
        """
        self.canonicality_filter = FastCanonicalityFilterBPE.from_huggingface(tokenizer)
        self.canonicality_filter.set_overrides(model_name)
        self.tokenizer = tokenizer
        # IMPORTANT: In the base Potential class, EOS will be added to vocab automatically
        # So we should NOT add it ourselves to the vocabulary we pass to super().__init__
        vocabulary = self.canonicality_filter._decode
        
        super().__init__(vocabulary)
    
    async def complete(self, context):
        """
        Assess if a complete sequence follows canonical tokenization.
        
        Args:
            context: Sequence of tokens
            
        Returns:
            float: 0.0 if canonical, float('-inf') otherwise
        """
        # Empty sequences are considered canonical
        if not context:
            return 0.0
        
        # Check if the sequence is canonical
        
        is_canonical = self._check_canonicality(context)
        return 0.0 if is_canonical else float('-inf')
    
    async def prefix(self, context):
        """
        Assess if a prefix sequence could potentially extend to a canonical sequence.
        For canonicality, this is the same as complete.
        
        Args:
            context: Sequence of tokens
            
        Returns:
            float: 0.0 if potentially canonical, float('-inf') otherwise
        """
        return await self.complete(context)
    
    async def logw_next(self, context):
        """
        Compute weights for each possible next token given the context.
        
        Args:
            context: Sequence of tokens
            
        Returns:
            LazyWeights: Weights for each token in the vocabulary and EOS
        """
        # Get the prefix weight (to check if context itself is canonical)
        ctx_log_w = await self.prefix(context)
        
        if ctx_log_w == float("-inf"):
            logws = np.full((len(self.vocab_eos),), float("-inf"), dtype=np.float32)
            #always allow eos
            logws[-1] = 0.0
        else:
            if context:
                t = (None, context[-1])
                filter_mask = self.canonicality_filter(t)
            else:
                filter_mask = np.ones(len(self.canonicality_filter._decode), dtype=bool)
                
            # Create log weights directly instead of using np.log(filter_mask)
            # This is more efficient, avoids torch (with torch can't combine with other potentials!)
            logws_no_eos = np.where(filter_mask, 0.0, float("-inf")).astype(np.float32)
            
            #append eos to the logws, always allow eos. 
            # NOTE: concat is because ._decode does not include eos while .vocab_eos does
            logws = np.concatenate([logws_no_eos, np.array([0.0], dtype=np.float32)])
        
        return self.make_lazy_weights(logws)
    
    def _check_canonicality(self, context):
        """
        Check if a sequence follows canonical tokenization.
        
        Args:
            context: Sequence of tokens
            
        Returns:
            bool: True if the sequence is canonical, False otherwise
        """
        # If we're checking a single token, it's always canonical
        if len(context) == 1:
            return True
        
        # Check all adjacent token pairs for canonicality
        for i in range(1, len(context)):
            prev_token = context[i-1]
            current_token = context[i]
            
            # Format expected by the filter: (None, previous_token)
            t = (None, prev_token)
            mask = self.canonicality_filter(t)
            # print("percent of mask: ", np.sum(mask)*100 / len(mask))
            
            # Find token_id in the canonicality filter's vocabulary
            token_id = self.canonicality_filter._encode[current_token]
            if not mask[token_id]:
                return False
        
        return True
    