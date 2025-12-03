import torch
import warnings
from typing import NamedTuple
from genlm.control.potential.base import Potential


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
    Container for token mappings between bytes and tokens IDs in a language model.

    The `decode` and `encode` mappings are generally different from the `PromptedLLM` class (see notes on EOS token handling).
    """

    decode: list[bytes]  # token_id -> bytes
    encode: dict[bytes, int]  # bytes -> token_id
    eos_idxs: list[int]  # IDs of EOS tokens
    eos_tokens: list[bytes]  # EOS tokens
    potential_vocab: list[bytes]  # tokens in the potential's vocabulary

    @classmethod
    def create(cls, decode, eos_tokens):
        if len(set(eos_tokens)) != len(eos_tokens):
            raise ValueError("Duplicate eos tokens")
        encode = {x: i for i, x in enumerate(decode)}
        if not all(eos in encode for eos in eos_tokens):
            raise ValueError("EOS token not in language model vocabulary")
        eos_idxs = [encode[eos] for eos in eos_tokens]
        eos_tokens_set = set(eos_tokens)
        potential_vocab = [x for x in decode if x not in eos_tokens_set]
        return cls(
            decode=decode,
            encode=encode,
            eos_idxs=eos_idxs,
            eos_tokens=eos_tokens,
            potential_vocab=potential_vocab,
        )


class PromptedLLM(Potential):
    """A potential representing a language model conditioned on a fixed prompt prefix.

    `PromptedLLM`s operate on byte sequences.

    Notes on EOS Token Handling:\n
    - Tokens to treat as end-of-sequence tokens are specified via the `eos_tokens` argument.\n
    - These tokens are excluded from the potential's vocabulary and as such do not appear in the `vocab` attribute.\n
        This means they cannot appear in any input contexts to the potential nor in the output of `logw_next`. They can be used in the prompt however.\n
    - The log probability assigned to the `genlm.control`'s reserved `EOS` token is the sum of the log probabilities of all the specified EOS tokens.\n

    This class wraps an `AsyncLM` instance.
    """

    def __init__(
        self,
        llm,
        prompt_ids=None,
        eos_tokens=None,
        temperature=1.0,
        token_maps=None,
    ):
        """`
        Initializes the PromptedLLM potential.

        Args:
            llm (AsyncLM): The language model to use.
            prompt_ids (list[int], optional): Optional prompt to use as a prompt prefix for all input contexts.
                Must be a list of token IDs. Defaults to None. The prompt ids can be set post-init via `prompt` or `prompt_ids`.
            eos_tokens (list[bytes], optional): List of tokens to treat as end-of-sequence tokens.
                Defaults to the EOS token of the language model's tokenizer.
            temperature (float, optional): The temperature to apply to the language model's logits. Defaults to 1.
            token_maps (TokenMappings, optional): A precomputed mapping of tokens to token IDs with the potential's vocabulary.
                If provided, `eos_tokens` must not be provided. Defaults to None, which constructs a TokenMappings from the language model's byte vocabulary and the EOS tokens.
        """
        self.model = llm
        self.prompt_ids = prompt_ids or []
        self.temperature = temperature

        if token_maps is not None:
            if eos_tokens is not None:
                raise ValueError(
                    "eos_tokens must not be provided when token_maps is provided."
                )
            self.token_maps = token_maps
        else:
            self.token_maps = TokenMappings.create(
                decode=self.model.byte_vocab,
                eos_tokens=eos_tokens
                or [self.model.byte_vocab[self.model.tokenizer.eos_token_id]],
            )

        super().__init__(vocabulary=self.token_maps.potential_vocab)

    @classmethod
    def from_name(
        cls,
        name,
        backend=None,
        eos_tokens=None,
        prompt_ids=None,
        temperature=1.0,
        **kwargs,
    ):
        """Create a `PromptedLLM` from a HuggingFace model name.

        Args:
            name (str): Name of the model to load
            backend (str, optional): `AsyncLM` backend to use:\n
                * 'vllm' to instantiate an `AsyncVirtualLM`; ideal for GPU usage\n
                * 'hf' for an `AsyncTransformer`; ideal for CPU usage\n
                * 'mlx' for an `AsyncMlxLM`; ideal for Apple silicon usage\n
                * 'mock' for a `MockAsyncLM`; ideal for testing.\n
                Defaults to 'vllm' if CUDA is available, otherwise 'hf'.
            eos_tokens (list[bytes], optional): List of tokens to treat as end-of-sequence tokens.
                Defaults to the EOS token of the language model's tokenizer.
            prompt_ids (list[int], optional): Optional prompt to use as a prompt prefix for all input contexts.
                Must be a list of token IDs. Defaults to None. The prompt ids can be set post-init via `set_prompt_from_str` or `prompt_ids`.
            temperature (float, optional): The temperature to apply to the language model's logits. Defaults to 1.
            **kwargs (dict): Additional arguments passed to AsyncLM constructor

        Returns:
            (PromptedLLM): An instance of PromptedLLM
        """
        backend = backend or ("vllm" if torch.cuda.is_available() else "hf")
        model = load_model_by_name(name, backend=backend, **kwargs)
        return cls(
            model, prompt_ids=prompt_ids, eos_tokens=eos_tokens, temperature=temperature
        )

    @property
    def eos_tokens(self):
        return self.token_maps.eos_tokens

    @eos_tokens.setter
    def eos_tokens(self, value):
        raise ValueError(
            "Cannot reset eos_tokens after initialization. "
            "Use spawn_new_eos(new_eos_tokens) instead."
        )

    @property
    def prompt(self):
        """
        Get the current prompt as a list of byte sequences corresponding to the prompt token IDs.

        Returns:
            (list[bytes]|None): The current prompt as a list of bytes sequences or None if no prompt_ids are set.
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

    def encode_tokens(self, tokens):
        """Encode a list of byte tokens to a list of token IDs in
        the underlying language model's vocabulary.

        Args:
            tokens (list[bytes]): List of byte tokens to encode

        Returns:
            (list[int]): A list of token IDs corresponding to the input tokens.

        Raises:
            ValueError: If any token is not in the vocabulary
        """
        try:
            return [self.token_maps.encode[x] for x in tokens]
        except KeyError as e:
            raise ValueError(f"Token {e.args[0]} not in vocabulary") from e

    def decode_tokens(self, ids):
        """
        Decode a list of token IDs in the language model's vocabulary to a list of byte tokens.

        Args:
            ids (list[int]): A list of token IDs in the language model's vocabulary.

        Returns:
            (list[bytes]): A list of byte tokens corresponding to the input token IDs.
        """
        return [self.token_maps.decode[x] for x in ids]

    def tokenize(self, context_str):
        """Tokenize a string to a list of `bytes` objects, each corresponding to a token in the vocabulary.

        Uses the language model's tokenizer to map `context_str` to a list of token IDs, and then decodes the token IDs to bytes.

        Args:
            context_str (str): A string to encode

        Returns:
            (List[bytes]): A list of byte tokens corresponding to the input string.
        """
        return self.decode_tokens(self.model.tokenizer.encode(context_str))

    async def log_probability(self, context):
        """
        Compute the log probability of `context` given the prompt.

        Args:
            context (list[bytes]): A sequence of bytes tokens.

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
            context (list[bytes]): A sequence of bytes tokens.

        Returns:
            (float): The log probability of `context`.
        """
        return await self.log_probability(context)

    async def complete(self, context):
        """
        Compute the log probability of `context` and the eos tokens given the prompt.

        If the model has multiple eos tokens, their probabilities will be summed.

        Args:
            context (list[bytes]): A sequence of bytes tokens.

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
        is the sum of the log probabilities of `self.eos_tokens`.

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

        logw_next = logw_next[: len(self.token_maps.decode)]
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
            context (List[bytes]): A sequence of bytes tokens.

        Returns:
            (LazyWeights): Log probabilities for next tokens and EOS.
        """
        logw_next = self._maybe_temper(
            await self.model.next_token_logprobs(
                self.prompt_ids + self.encode_tokens(context)
            )
        )
        return self._process_logw_next(logw_next)

    async def batch_logw_next(self, contexts):
        """Get log probabilities for next tokens given the prompt and `context`, for a batch of contexts.

        Args:
            contexts (list[list[bytes]]): A list of sequences of bytes tokens.

        Returns:
            (List[LazyWeights]): Log probabilities for next tokens and EOS for each context.
        """
        logw_nexts = self._maybe_temper(
            await self.model.batch_next_token_logprobs(
                [self.prompt_ids + self.encode_tokens(context) for context in contexts]
            )
        )
        return [self._process_logw_next(logw_next) for logw_next in logw_nexts]

    def __repr__(self):
        return f"PromptedLLM(prompt={self.prompt!r})"

    def spawn(self, prompt_ids=None, eos_tokens=None, temperature=None):
        """
        Spawn a new PromptedLLM.

        Args:
            prompt_ids (optional, list[int]): The prompt to use as a prompt prefix for all input contexts.
                Defaults to the same prompt_ids as `self`.
            eos_tokens (optional, list[bytes]): A list of tokens to treat as end-of-sequence tokens.
                Defaults to the same eos_tokens as `self`.
            temperature (optional, float): The temperature with which to rescale logprobs.
                Defaults to the same temperature as `self`.

        Returns:
            (PromptedLLM): A new PromptedLLM with the same prompt and eos tokens.

        Note:
            This is a shallow copy. The new PromptedLLM will share the underlying AsyncLM instance.
        """
        prompt_ids = prompt_ids if prompt_ids is not None else self.prompt_ids.copy()
        temperature = temperature if temperature is not None else self.temperature

        if (eos_tokens is None) or (eos_tokens == self.token_maps.eos_tokens):
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
            eos_tokens=eos_tokens,
            temperature=temperature,
        )

    def spawn_new_eos(self, eos_tokens):
        """
        Create a new PromptedLLM with a different set of end-of-sequence tokens.

        Args:
            eos_tokens (list[bytes]): A list of tokens to treat as end-of-sequence tokens.

        Returns:
            (PromptedLLM): A new PromptedLLM with the specified end-of-sequence tokens.
                The new model will have the same prompt_ids as `self`.
        """
        return self.spawn(eos_tokens=eos_tokens)

    def to_autobatched(self):
        raise ValueError("PromptedLLMs are autobatched by default.")
