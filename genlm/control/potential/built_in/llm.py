import contextvars
import weakref
import torch
import warnings
from typing import NamedTuple
from genlm.control.potential.base import Potential, _burst_logw_next_overrides
from genlm.backend.tokenization import Token


_prompt_ids_overrides: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_prompt_ids_overrides", default=None
)


def _walk_leaves(potential):
    """Yield the leaf potentials (no ``children``), recursing composites' ``children``."""
    children = potential.children
    if not children:
        yield potential
        return
    for child in children:
        yield from _walk_leaves(child)


def _is_burst_lm(p):
    return isinstance(p, PromptedLLM) and p.model.supports_burst


def find_engine_lm(potential):
    """The single burst-capable engine LM leaf, or ``None`` if not exactly one."""
    lms = [lf for lf in _walk_leaves(potential) if _is_burst_lm(lf)]
    return lms[0] if len(lms) == 1 else None


def lm_leaves(potential):
    """All ``PromptedLLM`` leaves."""
    return [lf for lf in _walk_leaves(potential) if isinstance(lf, PromptedLLM)]


def constraint_leaf_ids(potential):
    """``id``s of the non-(burst-LM) leaves -- the constraint identity a batched burst's
    groups must share (counts a non-burst ``PromptedLLM``)."""
    return frozenset(id(lf) for lf in _walk_leaves(potential) if not _is_burst_lm(lf))


def _compat_eos_tokens(eos_byte_strings, kwargs):
    """Handle deprecated ``eos_tokens`` kwarg, forwarding to ``eos_byte_strings``."""
    old = kwargs.pop("eos_tokens", None)
    if old is not None:
        if eos_byte_strings is not None:
            raise TypeError(
                "Cannot specify both 'eos_byte_strings' and the deprecated 'eos_tokens'."
            )
        warnings.warn(
            "'eos_tokens' is deprecated, use 'eos_byte_strings' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return old
    return eos_byte_strings


class _TokenEncodeDict(dict):
    """A dict mapping Token→int that also accepts bytes keys (deprecated).

    Primary keys are Token objects (hashed by token_id). When a plain bytes key
    is used, falls back to a cached bytes→token_id lookup and emits a
    DeprecationWarning.
    """

    def __init__(self, token_dict):
        super().__init__(token_dict)
        self._bytes_fallback = None

    def _build_bytes_fallback(self):
        if self._bytes_fallback is None:
            self._bytes_fallback = {}
            for token, idx in super().items():
                if isinstance(token, Token):
                    bs = token.byte_string
                    if bs not in self._bytes_fallback:
                        self._bytes_fallback[bs] = idx

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if Token.is_plain_bytes(key):
                self._build_bytes_fallback()
                if key in self._bytes_fallback:
                    warnings.warn(
                        "Indexing token_maps.encode by bytes is deprecated. "
                        "Use Token objects instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    return self._bytes_fallback[key]
            raise

    def __contains__(self, key):
        if super().__contains__(key):
            return True
        if Token.is_plain_bytes(key):
            self._build_bytes_fallback()
            return key in self._bytes_fallback
        return False


def load_model_by_name(name, backend, **kwargs):
    if backend == "vllm":
        from genlm.backend.llm import AsyncVirtualLM  # pragma: no cover

        model_cls = AsyncVirtualLM  # pragma: no cover
    elif backend == "hf":
        from genlm.backend.llm import AsyncTransformer

        model_cls = AsyncTransformer
    elif backend == "mlx":
        from genlm.backend.llm import AsyncMlxLM

        model_cls = AsyncMlxLM
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Must be one of ['vllm', 'hf', 'mlx']"
        )  # pragma: no cover

    return model_cls.from_name(name, **kwargs)


class TokenMappings(NamedTuple):
    """
    Container for token mappings in a language model.

    Attributes:
        decode: All Token objects in the vocabulary (indexed by token_id)
        encode: Mapping from Token to its position in decode (for backwards compat,
            also accepts bytes lookup via Token's bytes subclassing)
        eos_idxs: Token IDs for EOS tokens
        eos_byte_strings: EOS tokens as byte strings
        eos_token_objs: Actual EOS Token objects
        potential_vocab: Vocabulary excluding EOS tokens
    """

    decode: list[Token]
    encode: dict[Token, int]
    eos_idxs: list[int]
    eos_byte_strings: list[bytes]
    eos_token_objs: list[Token]
    potential_vocab: list[Token]

    @classmethod
    def create(cls, decode, eos_byte_strings=None, **kwargs):
        """Create TokenMappings from a vocabulary and EOS tokens.

        Args:
            decode (list[Token]): List of Token objects representing the full vocabulary.
            eos_byte_strings (list[bytes]): List of byte strings representing EOS tokens.
        """
        eos_byte_strings = _compat_eos_tokens(eos_byte_strings, kwargs)
        # Coerce to bytes -> make spawn_new_eos accept tokens, bytes, or any
        # mix without the caller having to remember.
        eos_byte_strings = [bytes(bs) for bs in eos_byte_strings]
        if len(set(eos_byte_strings)) != len(eos_byte_strings):
            raise ValueError("Duplicate eos byte strings")

        # Collect ALL tokens whose byte_string matches any EOS byte_string.
        # When multiple tokens share the same byte_string (duplicate byte
        # representations), all of them must be treated as EOS — otherwise the
        # model could emit a duplicate token that the sampler wouldn't
        # recognise as end-of-sequence.
        #
        # Order: first the primary matches (one per eos_byte_string, in input
        # order), then any extra duplicates found in decode order.
        eos_bs_to_tokens = {bs: [] for bs in eos_byte_strings}
        for token in decode:
            if token.byte_string in eos_bs_to_tokens:
                eos_bs_to_tokens[token.byte_string].append(token)

        # Verify all requested EOS byte_strings were found
        missing = {bs for bs, toks in eos_bs_to_tokens.items() if not toks}
        if missing:
            raise ValueError("EOS token not in language model vocabulary")

        # Primary matches first (preserves input order), then duplicates
        seen = set()
        eos_token_objs = []
        eos_idxs = []
        for bs in eos_byte_strings:
            for token in eos_bs_to_tokens[bs]:
                if token.token_id not in seen:
                    seen.add(token.token_id)
                    eos_token_objs.append(token)
                    eos_idxs.append(token.token_id)

        # Build potential_vocab excluding all EOS tokens
        eos_token_ids = set(eos_idxs)
        potential_vocab = [
            token for token in decode if token.token_id not in eos_token_ids
        ]

        encode = _TokenEncodeDict({token: i for i, token in enumerate(decode)})

        return cls(
            decode=decode,
            encode=encode,
            eos_idxs=eos_idxs,
            eos_byte_strings=eos_byte_strings,
            eos_token_objs=eos_token_objs,
            potential_vocab=potential_vocab,
        )


class PromptedLLM(Potential):
    """A potential representing a language model conditioned on a fixed prompt prefix.

    `PromptedLLM`s operate on byte sequences.

    Notes on EOS Token Handling:\n
    - Tokens to treat as end-of-sequence tokens are specified via the `eos_byte_strings` argument.\n
    - These tokens are excluded from the potential's vocabulary and as such do not appear in the `vocab` attribute.\n
        This means they cannot appear in any input contexts to the potential nor in the output of `logw_next`. They can be used in the prompt however.\n
    - The log probability assigned to the `genlm.control`'s reserved `EOS` token is the sum of the log probabilities of all the specified EOS tokens.\n

    This class wraps an `AsyncLM` instance.
    """

    def __init__(
        self,
        llm,
        prompt_ids=None,
        eos_byte_strings=None,
        temperature=1.0,
        token_maps=None,
        lora_name=None,
        **kwargs,
    ):
        """`
        Initializes the PromptedLLM potential.

        Args:
            llm (AsyncLM): The language model to use.
            prompt_ids (list[int], optional): Optional prompt to use as a prompt prefix for all input contexts.
                Must be a list of token IDs. Defaults to None. The prompt ids can be set post-init via `prompt` or `prompt_ids`.
            eos_byte_strings (list[bytes], optional): List of tokens to treat as end-of-sequence tokens.
                Defaults to the EOS token of the language model's tokenizer.
            temperature (float, optional): The temperature to apply to the language model's logits. Defaults to 1.
            token_maps (TokenMappings, optional): A precomputed mapping of tokens to token IDs with the potential's vocabulary.
                If provided, `eos_byte_strings` must not be provided. Defaults to None, which constructs a TokenMappings from the language model's byte vocabulary and the EOS tokens.
        """
        eos_byte_strings = _compat_eos_tokens(eos_byte_strings, kwargs)
        self.model = llm
        self._default_prompt_ids = list(prompt_ids or [])
        self.temperature = temperature
        self.lora_name = lora_name  # property setter derives self._fwd

        if token_maps is not None:
            if eos_byte_strings is not None:
                raise ValueError(
                    "eos_byte_strings must not be provided when token_maps is provided."
                )
            self.token_maps = token_maps
        else:
            byte_vocab = self.model.byte_vocab
            default_eos = byte_vocab[self.model.tokenizer.eos_token_id].byte_string
            self.token_maps = TokenMappings.create(
                decode=byte_vocab,
                eos_byte_strings=eos_byte_strings or [default_eos],
            )

        super().__init__(vocabulary=self.token_maps.potential_vocab)

    @classmethod
    def from_name(
        cls,
        name,
        backend=None,
        eos_byte_strings=None,
        prompt_ids=None,
        temperature=1.0,
        **kwargs,
    ):
        """Create a `PromptedLLM` from a Hugging Face model name.

        Args:
            name (str): Name of the model to load
            backend (str, optional): `AsyncLM` backend to use:\n
                * 'vllm' to instantiate an `AsyncVirtualLM`; ideal for GPU usage\n
                * 'hf' for an `AsyncTransformer`; ideal for CPU usage\n
                * 'mlx' for an `AsyncMlxLM`; ideal for Apple silicon usage\n
                * 'mock' for a `MockAsyncLM`; ideal for testing.\n
                Defaults to 'vllm' if CUDA is available, otherwise 'hf'.
            eos_byte_strings (list[bytes], optional): List of tokens to treat as end-of-sequence tokens.
                Defaults to the EOS token of the language model's tokenizer.
            prompt_ids (list[int], optional): Optional prompt to use as a prompt prefix for all input contexts.
                Must be a list of token IDs. Defaults to None. The prompt ids can be set post-init via `set_prompt_from_str` or `prompt_ids`.
            temperature (float, optional): The temperature to apply to the language model's logits. Defaults to 1.
            **kwargs (dict): Additional arguments passed to AsyncLM constructor

        Returns:
            (PromptedLLM): An instance of PromptedLLM
        """
        eos_byte_strings = _compat_eos_tokens(eos_byte_strings, kwargs)
        backend = backend or ("vllm" if torch.cuda.is_available() else "hf")
        model = load_model_by_name(name, backend=backend, **kwargs)
        return cls(
            model, prompt_ids=prompt_ids, eos_byte_strings=eos_byte_strings, temperature=temperature
        )

    @property
    def eos_byte_strings(self):
        return self.token_maps.eos_byte_strings

    @eos_byte_strings.setter
    def eos_byte_strings(self, value):
        raise ValueError(
            "Cannot reset eos_byte_strings after initialization. "
            "Use spawn_new_eos(new_eos_byte_strings) instead."
        )

    @property
    def lora_name(self):
        """LoRA adapter this view forwards under (``None`` = base model). The burst
        tags each substream with it; the slow lane forwards through ``_fwd`` (an
        adapter-bound view of the engine), so both lanes apply the adapter
        consistently. Assigning rebinds the forward handle — rebind between SMC
        runs, never mid-burst (the burst snapshots adapter names at start)."""
        return self._lora_name

    @lora_name.setter
    def lora_name(self, lora_name):
        self._lora_name = lora_name
        self._fwd = self.model.lora_view(lora_name)

    @property
    def prompt_ids(self):
        """The currently active prompt token ids.

        Backed by a module-level ``contextvars.ContextVar`` holding a
        ``WeakKeyDictionary`` keyed by the instance itself. Concurrent
        asyncio tasks each see their own writes via copy-on-write; sibling
        tasks see the parent context's value or the instance default.
        Entries vanish automatically when an instance is garbage-collected.
        """
        overrides = _prompt_ids_overrides.get()
        if overrides is None:
            return self._default_prompt_ids
        return overrides.get(self, self._default_prompt_ids)

    @prompt_ids.setter
    def prompt_ids(self, value):
        current = _prompt_ids_overrides.get()
        new = weakref.WeakKeyDictionary() if current is None else weakref.WeakKeyDictionary(current)
        new[self] = list(value)
        _prompt_ids_overrides.set(new)

    @property
    def prompt(self):
        """
        Get the current prompt as Token objects.

        Returns:
            (list[Token]|None): The current prompt as Token objects, or None if no prompt_ids are set.
        """
        if not self.prompt_ids:
            return  # pragma: no cover
        return [self.token_maps.decode[x] for x in self.prompt_ids]

    def set_prompt_from_str(self, prompt_str):
        """Set the fixed prompt from a string.

        Modifies `prompt_ids` to be the token IDs of the input prompt according to the language model's tokenizer.

        Args:
            prompt_str (str): The prompt to set.
        """
        if not isinstance(prompt_str, str):
            raise ValueError(
                f"Prompt must a string got {type(prompt_str)}. "
                f"To set the prompt from a list of token IDs, use prompt_ids."
            )

        if prompt_str.endswith(" "):
            warnings.warn(
                "Prompt ends with whitespace, which may affect tokenization. "
                "Consider removing trailing whitespace.",
                stacklevel=2,
            )

        self.prompt_ids = self.model.tokenizer.encode(prompt_str)

    def _find_token_id_for_bytes(self, byte_string):
        """Find token_id for a byte_string (first match for duplicates).

        Uses a lazily-built cache for O(1) lookup. For duplicate byte strings,
        returns the first token_id encountered in the vocabulary.
        """
        if not hasattr(self, "_bytes_to_token_id"):
            # Build reverse map: bytes → first token_id. Later entries don't
            # overwrite, so the first match wins (consistent with old behavior).
            self._bytes_to_token_id = {}
            for token in self.token_maps.decode:
                if token.byte_string not in self._bytes_to_token_id:
                    self._bytes_to_token_id[token.byte_string] = token.token_id
        return self._bytes_to_token_id.get(byte_string)

    def encode_tokens(self, tokens):
        """Encode a list of Token objects to token IDs.

        Args:
            tokens (list[Token]): List of Token objects

        Returns:
            (list[int]): A list of token IDs corresponding to the input tokens.

        Raises:
            ValueError: If any token is not in the vocabulary.

        Note:
            Passing bytes is deprecated. Use Token objects from llm.tokenize().
        """
        if not tokens:
            return []

        result = []
        warned = False
        for item in tokens:
            if isinstance(item, Token):
                result.append(item.token_id)
            else:
                if not warned:
                    warnings.warn(
                        "Passing bytes to encode_tokens is deprecated. "
                        "Use Token objects for precise control. ",
                        DeprecationWarning,
                        stacklevel=3,
                    )
                    warned = True
                token_id = self._find_token_id_for_bytes(item)
                if token_id is None:
                    raise ValueError(f"Token {item!r} not in vocabulary")
                result.append(token_id)
        return result

    def decode_tokens(self, ids):
        """
        Decode a list of token IDs to Token objects.

        Args:
            ids (list[int]): A list of token IDs in the language model's vocabulary.

        Returns:
            (list[Token]): Token objects corresponding to the input token IDs.
        """
        return [self.token_maps.decode[x] for x in ids]

    def tokenize(self, context_str):
        """Tokenize a string to a list of Token objects.

        Uses the language model's tokenizer to map `context_str` to token IDs,
        then returns the corresponding Token objects.

        Args:
            context_str (str): A string to encode

        Returns:
            (list[Token]): Token objects corresponding to the input string.
        """
        return self.decode_tokens(self.model.tokenizer.encode(context_str))

    async def log_probability(self, context):
        """
        Compute the log probability of `context` given the prompt.

        Args:
            context (list[bytes] | list[Token]): A sequence of byte tokens or Token objects.

        Returns:
            (float): The log probability of `context`.
        """
        if not context:
            return 0

        context_ids = self.encode_tokens(context)
        return await self._log_probability(context_ids)

    async def _log_probability(self, context_ids):
        prefixes = [self.prompt_ids + context_ids[:i] for i in range(len(context_ids))]
        log_ps = self._maybe_temper(
            await self._fwd.batch_next_token_logprobs(prefixes)
        )
        target_ids = torch.tensor(context_ids, device=log_ps.device)
        with torch.no_grad():
            token_logprobs = torch.gather(log_ps, 1, target_ids.unsqueeze(1))
            total_logprob = token_logprobs.sum().item()

        return total_logprob

    def _maybe_temper(self, logps):
        if self.temperature == 1:
            return logps
        return torch.log_softmax(logps / self.temperature, dim=-1)

    async def prefix(self, context):
        """
        Compute the log probability of `context` given the prompt.

        Args:
            context (list[bytes] | list[Token]): A sequence of byte tokens or Token objects.

        Returns:
            (float): The log probability of `context`.
        """
        return await self.log_probability(context)

    async def complete(self, context):
        """
        Compute the log probability of `context` and the eos tokens given the prompt.

        If the model has multiple eos tokens, their probabilities will be summed.

        Args:
            context (list[bytes] | list[Token]): A sequence of byte tokens or Token objects.

        Returns:
            (float): The log probability of the context.
        """
        context_ids = self.encode_tokens(context)
        logp_context = await self._log_probability(context_ids)
        logp_next = self._maybe_temper(
            await self._fwd.next_token_logprobs(self.prompt_ids + context_ids)
        )
        logp_eos = torch.logsumexp(logp_next[self.token_maps.eos_idxs], dim=0).item()
        return logp_context + logp_eos

    def _eos_index_tensors(self, device):
        """Cache (and return) the EOS / non-EOS column-index tensors used to fold
        the engine vocab into the control vocab (EOS kept last). Shared by the
        per-row :meth:`_process_logw_next` and the batched
        :meth:`_process_logw_next_batch`."""
        if (
            not hasattr(self, "_eos_idxs_tensor")
            or not hasattr(self, "_non_eos_indices")
            or self._eos_idxs_tensor.device != device
        ):
            self._eos_idxs_tensor = torch.tensor(
                self.token_maps.eos_idxs, device=device
            )
            all_indices = torch.arange(len(self.token_maps.decode), device=device)
            self._non_eos_indices = all_indices[
                ~torch.isin(all_indices, self._eos_idxs_tensor)
            ]
        return self._eos_idxs_tensor, self._non_eos_indices

    def _verify_logit_padding(self, n_logits):
        """Guard the tail-padding done when the model emits fewer logits than
        ``len(token_maps.decode)`` (the tokenizer added tokens beyond the embedding
        matrix, e.g. Gemma's <image_soft_token>). Padding the tail with -inf is only
        correct if the model's logit indices are contiguous ``0..n_logits-1``;
        verify that once. Shared by the per-row :meth:`_process_logw_next` and the
        batched :meth:`_process_logw_next_batch` so both raise (rather than silently
        mis-fold columns) on a model that violates the assumption."""
        if not hasattr(self, "_logit_padding_verified"):
            for i in range(n_logits):
                if self.token_maps.decode[i].token_id != i:
                    raise ValueError(
                        f"Token ID / index mismatch at position {i}: "
                        f"decode[{i}].token_id={self.token_maps.decode[i].token_id}. "
                        f"Padding assumes added tokens are at indices >= vocab_size."
                    )
            self._logit_padding_verified = True

    def _process_logw_next_batch(self, logits):
        """Vectorized, on-device analog of :meth:`_process_logw_next`: maps a
        ``[N, n_logits]`` batch of raw engine logits to ``[N, V+1]`` control-vocab
        log-weights (EOS folded into the last column) entirely on device, no host
        transfer. Returns a ``torch.Tensor`` (not a ``LazyWeights``); the caller samples
        on-device and transfers only the N drawn ids back."""
        eos_idxs, non_eos = self._eos_index_tensors(logits.device)
        n_decode = len(self.token_maps.decode)
        n_logits = logits.shape[1]
        if n_logits < n_decode:
            self._verify_logit_padding(n_logits)  # same guard as the per-row path
            pad = torch.full(
                (logits.shape[0], n_decode - n_logits),
                float("-inf"),
                dtype=logits.dtype,
                device=logits.device,
            )
            logits = torch.cat([logits, pad], dim=1)
        logits = logits[:, :n_decode].log_softmax(dim=1)  # [N, n_decode]
        out = torch.full(
            (logits.shape[0], len(self.vocab) + 1),
            float("-inf"),
            dtype=logits.dtype,
            device=logits.device,
        )
        out[:, : len(self.vocab)] = logits[:, non_eos]
        if eos_idxs.numel() == 1:
            # On-device single-EOS gather: index with the 0-dim tensor, NOT
            # `eos_idxs.item()`, so the EOS column needs no host sync.
            out[:, -1] = logits[:, eos_idxs[0]]
        else:
            out[:, -1] = torch.logsumexp(logits[:, eos_idxs], dim=1)
        return out

    def _process_logw_next(self, logw_next):
        """Process the log probabilities for the next tokens.

        This function rearranges the log probabilities such that the end-of-sequence (EOS) token's log probability
        is the sum of the log probabilities of `self.eos_byte_strings`.

        Args:
            logw_next (torch.tensor): The log probabilities for the next tokens.

        Returns:
            (LazyWeights): Processed log probabilities for the next tokens.
        """
        # N=1 adapter over the batched on-device fold: same pad / slice / log_softmax /
        # EOS fold, kept as a CPU torch tensor for the slow lane's LazyWeights.
        out = self._process_logw_next_batch(logw_next.unsqueeze(0))
        return self.make_lazy_weights(out[0].float().cpu())

    async def logw_next(self, context):
        """Get log probabilities for next tokens given the prompt and `context`.

        Args:
            context (list[bytes] | list[Token]): A sequence of byte tokens or Token objects.

        Returns:
            (LazyWeights): Log probabilities for next tokens and EOS. Keys are Token objects.
        """
        override = _burst_logw_next_overrides.get()
        if override is not None and self in override:
            return override[self]  # burst: serve the engine's warm logits, no forward
        context_ids = self.encode_tokens(context)
        logw_next = self._maybe_temper(
            await self._fwd.next_token_logprobs(self.prompt_ids + context_ids)
        )
        return self._process_logw_next(logw_next)

    async def batch_logw_next(self, contexts):
        """Next-token log-weights for a batch of contexts, as ONE batched `LazyWeights`
        (`.weights` shape `[N, V+1]`). In a batched burst the engine's warm `[N, V+1]` batch
        is injected via the `burst_logw_next` override (same ContextVar as the scalar path,
        value batched) -- served directly, no forward.

        Args:
            contexts (list[list[bytes]] | list[list[Token]]): A list of token sequences.

        Returns:
            (LazyWeights): batched log-weights, `.weights` shape `[N, V+1]`. Keys are Tokens.
        """
        override = _burst_logw_next_overrides.get()
        if override is not None and self in override:
            return override[self]  # burst: the engine's warm [N, V+1] batch, no forward
        context_ids_batch = [self.encode_tokens(context) for context in contexts]
        logw_nexts = self._maybe_temper(
            await self._fwd.batch_next_token_logprobs(
                [self.prompt_ids + context_ids for context_ids in context_ids_batch]
            )
        )
        # One on-device fold over the [N, n_logits] batch -> [N, V+1] (== stacking the
        # per-row `_process_logw_next`), wrapped as one batched LazyWeights.
        return self.make_lazy_weights(self._process_logw_next_batch(logw_nexts).float().cpu())

    def __repr__(self):
        return f"PromptedLLM(prompt={self.prompt!r})"

    def spawn(self, prompt_ids=None, eos_byte_strings=None, temperature=None, **kwargs):
        """
        Spawn a new PromptedLLM.

        Args:
            prompt_ids (optional, list[int]): The prompt to use as a prompt prefix for all input contexts.
                Defaults to the same prompt_ids as `self`.
            eos_byte_strings (optional, list[bytes]): A list of tokens to treat as end-of-sequence tokens.
                Defaults to the same eos_byte_strings as `self`.
            temperature (optional, float): The temperature with which to rescale logprobs.
                Defaults to the same temperature as `self`.

        Returns:
            (PromptedLLM): A new PromptedLLM with the same prompt and eos tokens.

        Note:
            This is a shallow copy. The new PromptedLLM will share the underlying AsyncLM instance.
        """
        eos_byte_strings = _compat_eos_tokens(eos_byte_strings, kwargs)
        prompt_ids = prompt_ids if prompt_ids is not None else self.prompt_ids.copy()
        temperature = temperature if temperature is not None else self.temperature

        if (eos_byte_strings is None) or (eos_byte_strings == self.token_maps.eos_byte_strings):
            # If the eos tokens don't change, we don't need to recompute the token maps or vocabulary.
            return PromptedLLM(
                self.model,
                prompt_ids=prompt_ids,
                temperature=temperature,
                token_maps=self.token_maps,
                lora_name=self.lora_name,
            )

        return PromptedLLM(
            self.model,
            prompt_ids=prompt_ids,
            eos_byte_strings=eos_byte_strings,
            temperature=temperature,
            lora_name=self.lora_name,
        )

    def spawn_new_eos(self, eos_byte_strings=None, **kwargs):
        """
        Create a new PromptedLLM with a different set of end-of-sequence tokens.

        Args:
            eos_byte_strings (list[bytes]): A list of tokens to treat as end-of-sequence tokens.

        Returns:
            (PromptedLLM): A new PromptedLLM with the specified end-of-sequence tokens.
                The new model will have the same prompt_ids as `self`.
        """
        eos_byte_strings = _compat_eos_tokens(eos_byte_strings, kwargs)
        return self.spawn(eos_byte_strings=eos_byte_strings)

    def to_autobatched(self):
        raise ValueError("PromptedLLMs are autobatched by default.")
