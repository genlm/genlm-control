import torch
import warnings
from typing import NamedTuple
from genlm.control.potential.base import Potential
from genlm.backend.tokenization import Token


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
        if len(set(eos_byte_strings)) != len(eos_byte_strings):
            raise ValueError("Duplicate eos byte strings")

        eos_byte_strings_set = set(eos_byte_strings)

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
        self.prompt_ids = prompt_ids or []
        self.temperature = temperature

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
        # TODO: Handle race condition where prompt_ids reset concurrently.
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
            await self.model.batch_next_token_logprobs(prefixes)
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
            await self.model.next_token_logprobs(self.prompt_ids + context_ids)
        )
        logp_eos = torch.logsumexp(logp_next[self.token_maps.eos_idxs], dim=0).item()
        return logp_context + logp_eos

    def _process_logw_next(self, logw_next):
        """Process the log probabilities for the next tokens.

        This function rearranges the log probabilities such that the end-of-sequence (EOS) token's log probability
        is the sum of the log probabilities of `self.eos_byte_strings`.

        Args:
            logw_next (torch.tensor): The log probabilities for the next tokens.

        Returns:
            (LazyWeights): Processed log probabilities for the next tokens.
        """
        # This is ugly, but it's useful for all potentials to adhere to the convention
        # of keeping the EOS token at the end of the weights array.

        # Cache eos_idxs_tensor and non_eos_indices on first use
        if (
            not hasattr(self, "_eos_idxs_tensor")
            or not hasattr(self, "_non_eos_indices")
            or self._eos_idxs_tensor.device != logw_next.device
        ):
            self._eos_idxs_tensor = torch.tensor(
                self.token_maps.eos_idxs, device=logw_next.device
            )
            all_indices = torch.arange(
                len(self.token_maps.decode), device=logw_next.device
            )
            self._non_eos_indices = all_indices[
                ~torch.isin(all_indices, self._eos_idxs_tensor)
            ]

        # The model may produce fewer logits than len(token_maps.decode) when
        # the tokenizer has added tokens beyond the model's embedding matrix
        # (e.g. Gemma's <image_soft_token>). Pad with -inf so these tokens
        # are unscorable but still present in the vocabulary.
        # We assert that HF models always produce logits for token indices
        # 0..vocab_size-1, and added tokens are at indices >= vocab_size.
        n_decode = len(self.token_maps.decode)
        n_logits = len(logw_next)
        if n_logits < n_decode:
            # Verify (once) that token IDs in the model's logit range are
            # contiguous 0..n_logits-1, so padding the tail is safe.
            if not hasattr(self, "_logit_padding_verified"):
                for i in range(n_logits):
                    if self.token_maps.decode[i].token_id != i:
                        raise ValueError(
                            f"Token ID / index mismatch at position {i}: "
                            f"decode[{i}].token_id={self.token_maps.decode[i].token_id}. "
                            f"Padding assumes added tokens are at indices >= vocab_size."
                        )
                self._logit_padding_verified = True
            pad = torch.full(
                (n_decode - n_logits,),
                float("-inf"),
                dtype=logw_next.dtype,
                device=logw_next.device,
            )
            logw_next = torch.cat([logw_next, pad])

        logw_next = logw_next[:n_decode]
        logw_next = logw_next.log_softmax(dim=0)
        _logw_next = torch.full(
            (len(self.vocab) + 1,),
            float("-inf"),
            dtype=logw_next.dtype,
            device=logw_next.device,
        )
        _logw_next[: len(self.vocab)] = logw_next[self._non_eos_indices]

        # Special case: if only one EOS idx, just assign directly (avoids cost of logsumexp)
        if self._eos_idxs_tensor.numel() == 1:
            _logw_next[-1] = logw_next[self._eos_idxs_tensor]
        else:
            _logw_next[-1] = torch.logsumexp(logw_next[self._eos_idxs_tensor], dim=0)

        return self.make_lazy_weights(_logw_next.float().cpu().numpy())

    async def logw_next(self, context):
        """Get log probabilities for next tokens given the prompt and `context`.

        Args:
            context (list[bytes] | list[Token]): A sequence of byte tokens or Token objects.

        Returns:
            (LazyWeights): Log probabilities for next tokens and EOS. Keys are Token objects.
        """
        context_ids = self.encode_tokens(context)
        logw_next = self._maybe_temper(
            await self.model.next_token_logprobs(self.prompt_ids + context_ids)
        )
        return self._process_logw_next(logw_next)

    async def batch_logw_next(self, contexts):
        """Get log probabilities for next tokens given the prompt and `context`, for a batch of contexts.

        Args:
            contexts (list[list[bytes]] | list[list[Token]]): A list of sequences of byte tokens or Token objects.

        Returns:
            (list[LazyWeights]): Log probabilities for next tokens and EOS for each context. Keys are Token objects.
        """
        context_ids_batch = [self.encode_tokens(context) for context in contexts]
        logw_nexts = self._maybe_temper(
            await self.model.batch_next_token_logprobs(
                [self.prompt_ids + context_ids for context_ids in context_ids_batch]
            )
        )
        return [self._process_logw_next(logw_next) for logw_next in logw_nexts]

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
            )

        return PromptedLLM(
            self.model,
            prompt_ids=prompt_ids,
            eos_byte_strings=eos_byte_strings,
            temperature=temperature,
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
