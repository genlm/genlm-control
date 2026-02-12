import asyncio
from collections import Counter
from typing import Any, Union

import numpy as np
import regex

from genlm.control import EOS
from genlm.control.potential import Potential
from genlm.control.potential.built_in.llm import TokenMappings
from genlm.control.util import LazyWeights
from genlm.backend import decode_vocab


class HarmonyChat:
    """Encodes the structure of the "assistant" field of the Harmony chat format.

    Provides methods to extract the "harmony channels" (analysis, final, commentary)
    from it. Since it operates on the byte representation of tokens, it also provides
    methods to convert between token IDs and byte representations.

    Attributes:
        tokenizer: The tokenizer used to encode and decode tokens.
        token_maps (TokenMappings): Mappings between token IDs and byte representations.
        potential_vocab (list[bytes]): The byte vocabulary used by potentials.
        end_token (bytes): Byte representation of the ``<|end|>`` token.
        message_token (bytes): Byte representation of the ``<|message|>`` token.
        channel_token (bytes): Byte representation of the ``<|channel|>`` token.
        analysis_tokens (list[bytes]): Byte representation of the ``"analysis"`` string.
        final_tokens (list[bytes]): Byte representation of the ``"final"`` string.
        commentary_tokens (list[bytes]): Byte representation of the ``"commentary"`` string.
    """

    def __init__(self, tokenizer: Any) -> None:
        """
        Initialize HarmonyChat with a tokenizer.

        Args:
            tokenizer: A tokenizer that supports the harmony chat format.
                The tokenizer must be able to encode the harmony chat tokens
                as single tokens.

        """
        # Check that the tokenizer object has the minimum required methods.
        assert hasattr(tokenizer, "encode"), "Tokenizer is missing the 'encode' method."
        assert hasattr(tokenizer, "decode"), "Tokenizer is missing the 'decode' method."
        assert hasattr(tokenizer, "apply_chat_template"), (
            "Tokenizer is missing the 'apply_chat_template' method."
        )

        # Check that the tokenizer supports the special tokens of the harmony chat format
        # (in which case they should all be encoded as single tokens).
        for token in [
            "<|start|>",
            "<|channel|>",
            "<|message|>",
            "<|end|>",
            "<|return|>",
        ]:
            assert len(tokenizer.encode(token)) == 1, (
                f"Token {token!r} is not encoded as a single token. "
                "The tokenizer does not appear to support the harmony chat format."
            )

        self.tokenizer = tokenizer
        _byte_vocab, _ = decode_vocab(
            tokenizer
        )  # Byte representation of each token. Follows the same schema as PromptedLLM.
        _eos_tokens = [
            _byte_vocab[
                tokenizer.eos_token_id
            ]  # for gpt-oss, this is the <|return|> token.
        ]

        self.token_maps = TokenMappings.create(
            decode=_byte_vocab, eos_tokens=_eos_tokens
        )
        self.potential_vocab = self.token_maps.potential_vocab

        # Store the byte representation of special tokens needed for harmony channel parsing.
        self.end_token = self.decode_tokens(self.tokenizer.encode("<|end|>"))[0]
        self.message_token = self.decode_tokens(self.tokenizer.encode("<|message|>"))[0]
        self.channel_token = self.decode_tokens(self.tokenizer.encode("<|channel|>"))[0]
        self.analysis_tokens = self.decode_tokens(
            self.tokenizer.encode("analysis")
        )  # The following tokens (analysis, commentary, final) are not reserved, and therefore they are not guaranteed to be single tokens.
        self.final_tokens = self.decode_tokens(self.tokenizer.encode("final"))
        self.commentary_tokens = self.decode_tokens(self.tokenizer.encode("commentary"))

    def extract_channel_content(
        self, token_bytes: list[bytes], i: int
    ) -> dict[str, Union[list[bytes], bool]] | None:
        """Extract content between the ``<|message|>`` token at position ``i`` and the next ``<|end|>`` token.

        Args:
            token_bytes (list[bytes]): The full token sequence.
            i (int): Index of the ``<|message|>`` token.

        Returns:
            (dict | None): A dict with keys ``"content"`` (list of byte tokens) and
                ``"is_prefix"`` (bool), or ``None`` if ``i`` is out of bounds.
        """

        if i >= len(token_bytes):
            return None  # pragma: no cover
        i += 1
        if self.end_token in token_bytes[i:]:
            end_position = token_bytes.index(self.end_token, i)
            content = token_bytes[i:end_position]
            is_prefix = False
        else:
            content = token_bytes[i:]
            is_prefix = True

        return {"content": content, "is_prefix": is_prefix}

    def extract_harmony_channels_from_tokens(
        self, token_bytes: list[bytes]
    ) -> dict[str, dict[str, Union[list[bytes], bool]] | None]:
        """Extract analysis, final, and commentary content from token bytes.

        Args:
            token_bytes (list[bytes]): List of byte tokens.

        Returns:
            (dict): A dictionary mapping channel names to their extracted content,
                or ``None`` if the channel is not present.

        Raises:
            AssertionError: If the token bytes do not form a valid harmony chat.
        """

        assert self.validate_harmony_format(token_bytes), (
            f"The context is not a valid harmony chat: {token_bytes}"
        )
        results = {"analysis": None, "final": None, "commentary": None}

        for i, token in enumerate(token_bytes[:-2]):
            # The harmony format assumes that the <|channel|> token is immediately followed by the channel type, thus we can stop before the last two tokens.
            # Look for <|channel|> token followed by analysis/final/commentary.
            if token == self.channel_token:
                j = i + 1
                # Check whether the analysis, final or commentary tokens follow the channel opening.
                if (
                    len(token_bytes) >= j + len(self.analysis_tokens)
                    and token_bytes[j : j + len(self.analysis_tokens)]
                    == self.analysis_tokens
                ):
                    results["analysis"] = self.extract_channel_content(
                        token_bytes, j + len(self.analysis_tokens)
                    )
                elif (
                    len(token_bytes) >= j + len(self.final_tokens)
                    and token_bytes[j : j + len(self.final_tokens)] == self.final_tokens
                ):
                    results["final"] = self.extract_channel_content(
                        token_bytes, j + len(self.final_tokens)
                    )
                elif (
                    len(token_bytes) >= j + len(self.commentary_tokens)
                    and token_bytes[j : j + len(self.commentary_tokens)]
                    == self.commentary_tokens
                ):
                    results["commentary"] = self.extract_channel_content(
                        token_bytes, j + len(self.commentary_tokens)
                    )

        return results

    def extract_harmony_channels_from_string(
        self, string: str, add_special_tokens: bool = False
    ) -> dict[str, dict[str, Union[list[bytes], bool]] | None]:
        """Extract analysis, final, and commentary content from a string.

        Uses the tokenizer to map from string to token IDs and from token IDs to token bytes,
        then calls :meth:`extract_harmony_channels_from_tokens`.

        Args:
            string (str): The harmony chat format string to extract channels from.
            add_special_tokens (bool): Whether to add special tokens during encoding.

        Returns:
            (dict): A dictionary mapping channel names to their extracted content
                (same format as :meth:`extract_harmony_channels_from_tokens`).
        """
        token_ids = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        token_bytes = self.decode_tokens(token_ids)
        return self.extract_harmony_channels_from_tokens(token_bytes)

    def encode_tokens(self, tokens: list[bytes]) -> list[int]:
        """Encode a list of byte tokens to token IDs in the language model's vocabulary.

        Args:
            tokens (list[bytes]): List of byte tokens to encode.

        Returns:
            (list[int]): A list of token IDs corresponding to the input tokens.

        Raises:
            ValueError: If any token is not in the vocabulary.
        """
        assert all(isinstance(x, bytes) for x in tokens), "Tokens must be bytes"
        try:
            return [self.token_maps.encode[x] for x in tokens]
        except KeyError as e:  # pragma: no cover
            raise ValueError(
                f"Token {e.args[0]} not in vocabulary"
            ) from e  # pragma: no cover

    def decode_tokens(self, ids: list[int]) -> list[bytes]:
        """Decode a list of token IDs to byte tokens.

        Args:
            ids (list[int]): A list of token IDs in the language model's vocabulary.

        Returns:
            (list[bytes]): A list of byte tokens corresponding to the input token IDs.
        """
        assert all(isinstance(x, int) for x in ids), "Token IDs must be integers"
        return [self.token_maps.decode[x] for x in ids]

    def validate_harmony_format(self, context: Union[str, list[bytes]]) -> bool:
        """Validate that the context is a valid harmony chat.

        Validates the "assistant" field of the chat format, which is generated
        by the language model.

        Args:
            context (str | list[bytes]): A string or a list of byte tokens.

        Returns:
            (bool): ``True`` if the context is a valid harmony chat, ``False`` otherwise.
        """
        if (
            isinstance(context, list) and len(context) > 0 and context[-1] == EOS
        ):  # Remove the EOS token if present.
            context = context[:-1]  # pragma: no cover

        if isinstance(context, list) and all(isinstance(x, bytes) for x in context):
            context_str = b"".join(context).decode("utf-8", errors="replace")
        elif isinstance(context, str):  # pragma: no cover
            context_str = context  # pragma: no cover
        else:  # pragma: no cover
            raise ValueError(
                f"Context must be a string or a list of bytes tokens, got {type(context)}"
            )  # pragma: no cover

        pattern = r"""
            ^
            (?:
                (?:<\|start\|>assistant)? # The assistant field is optional, the first one is part of the prompt and not the generated tokens
                (?:\s+to=functions\.\w+)? # Optional Function call field
                <\|channel\|> # We start with the channel specifications
                (analysis|commentary|final) # Choose between the three possible channels
                (?:\s+json)?
                <\|message\|> # The message content begins
                (?:(?!<\|start\|>|<\|message\|>|<\|channel\|>|<\|call\|>|<\|return\|>).)*  # The actual message content can contain everything except the special tokens.
                (?:<\|end\|>|<\|call\|>|<\|return\|>) # The channel is closed by the <|end|>, <|call|>, or <|return|> tokens.
            )*
            $
        """

        match = regex.match(
            pattern, context_str, regex.VERBOSE | regex.DOTALL, partial=True
        )
        if not match:  # If the string does not match, we return False.
            return False  # pragma: no cover

        channel_types = match.captures(1)
        counts = Counter(
            channel_types
        )  # Validate that each channel is used at most once in a turn.
        if any(count > 1 for count in counts.values()):
            return False  # pragma: no cover
        return True


VALID_CHANNELS = {"analysis", "final", "commentary"}


class HarmonyPotential(Potential):
    """A potential that applies a base constraint to specific channels of the Harmony chat format.

    The Harmony chat format structures LLM output into named channels (analysis, final, commentary).
    This potential extracts the content of specified channels and evaluates them under a base
    potential, leaving unconstrained channels free.

    Attributes:
        base_potential (Potential): The potential applied to constrained channel contents.
        harmony_chat (HarmonyChat): Parser for the Harmony chat format.
        constrained_channels (list[str]): Channels to which the base potential is applied.
    """

    def __init__(
        self,
        base_potential: Potential,
        llm_tokenizer: Any,
        constrained_channels: list[str],
    ) -> None:
        """Initialize the HarmonyPotential.

        Args:
            base_potential (Potential): A base potential applied to the constrained channels.
            llm_tokenizer: A tokenizer that supports the harmony chat format.
            constrained_channels (list[str]): A non-empty list of channels to constrain.
                Each element must be one of ``"analysis"``, ``"final"``, or ``"commentary"``.

        Raises:
            ValueError: If ``constrained_channels`` is empty or contains invalid channel names.
            AssertionError: If the base potential's vocabulary is not a subset of the
                harmony potential's vocabulary.
        """
        if not constrained_channels:
            raise ValueError("constrained_channels must be a non-empty list.")
        invalid = set(constrained_channels) - VALID_CHANNELS
        if invalid:
            raise ValueError(
                f"Invalid channel names: {invalid}. Must be one of {VALID_CHANNELS}."
            )

        self.base_potential = base_potential
        self.harmony_chat = HarmonyChat(llm_tokenizer)
        self.constrained_channels = constrained_channels

        super().__init__(self.harmony_chat.potential_vocab)

        assert set(base_potential.vocab) <= set(self.vocab), (
            "The base potential's vocabulary must be a subset of the harmony potential's vocabulary."
        )

    async def complete(self, context: list[bytes]) -> float:
        """Compute the log weight of the constrained channels as complete sequences.

        Args:
            context (list[bytes]): A list of byte tokens.

        Returns:
            (float): The sum (in log space) of the base potential's complete weight for each
                constrained channel. Returns 0 if no constrained channel is present.
        """
        channels = self.harmony_chat.extract_harmony_channels_from_tokens(context)

        coroutines = [
            self.base_potential.complete(channels[key]["content"])
            for key in channels
            if channels[key] is not None and key in self.constrained_channels
        ]
        if not coroutines:
            return 0.0
        results = await asyncio.gather(*coroutines)
        return sum(results)

    async def prefix(self, context: list[bytes]) -> float:
        """Compute the log weight of the constrained channels as a prefix.

        Each constrained channel is evaluated with the base potential: completed
        channels use ``complete``, while the currently open channel uses ``prefix``.

        Args:
            context (list[bytes]): A list of byte tokens.

        Returns:
            (float): The sum (in log space) of the base potential's weight for each
                constrained channel. Returns 0 if no constrained channel is present.
        """
        channels = self.harmony_chat.extract_harmony_channels_from_tokens(context)
        coroutines = []
        for key in channels:
            if channels[key] is not None and key in self.constrained_channels:
                if channels[key]["is_prefix"]:
                    coroutines.append(
                        self.base_potential.prefix(channels[key]["content"])
                    )
                else:
                    # Completed channels also contribute to the prefix weight.
                    coroutines.append(
                        self.base_potential.complete(channels[key]["content"])
                    )
        if not coroutines:
            return 0.0
        results = await asyncio.gather(*coroutines)
        return sum(results)

    async def logw_next(self, context: list[bytes]) -> LazyWeights:
        """Compute next-token log weights for each possible next token, including EOS.

        Args:
            context (list[bytes]): A list of byte tokens.

        Returns:
            (LazyWeights): Weights of each token in the vocabulary and EOS.

        Note:
            In the harmony chat format, the analysis and commentary channels are
            closed by the ``<|end|>`` token, while the final channel is closed by
            ``<|return|>`` (which also closes the chat and halts generation).

            The base potential uses the built-in EOS symbol to represent "the
            constrained string ends here." We need to remap this to the token
            the LLM actually emits to close the channel:

            - **analysis/commentary**: Move the base potential's EOS weight to the
              ``<|end|>`` token and set EOS to -inf, since generation must not halt
              mid-turn.
            - **final**: No remapping needed, because ``PromptedLLM`` already moves
              ``<|return|>`` probability to EOS, so the base potential and the LLM
              are already aligned.
        """

        channels = self.harmony_chat.extract_harmony_channels_from_tokens(context)

        next_token_weights = self.make_lazy_weights(np.zeros(len(self.vocab_eos)))
        incomplete_channels = {
            key
            for key in channels
            if channels[key] is not None and channels[key]["is_prefix"]
        }
        assert len(incomplete_channels) <= 1, (
            "At most one channel can have the 'is_prefix' flag set to true."
        )

        if len(incomplete_channels) == 0:
            return next_token_weights  # pragma: no cover

        key = incomplete_channels.pop()
        if key is not None and key in self.constrained_channels:
            if await self.base_potential.prefix(channels[key]["content"]) == float(
                "-inf"
            ):
                raise ValueError(  # pragma: no cover
                    f"Context {channels[key]['content']!r} has weight zero under `prefix`."
                )

            next_token_weights.weights += (
                await self.base_potential.logw_next(channels[key]["content"])
            ).weights

            if key == "analysis" or key == "commentary":
                # The base potential's EOS weight represents "string is complete."
                # Remap it to <|end|> (which the LLM emits to close these channels)
                # and set EOS to -inf to prevent the LLM from halting mid-turn.
                eos_weight = next_token_weights.weights[-1]
                idx = next_token_weights.encode[self.harmony_chat.end_token]
                next_token_weights.weights[idx] = eos_weight
                next_token_weights.weights[-1] = float("-inf")

            # For the final channel, no remapping is needed: PromptedLLM already
            # maps <|return|> to EOS, so the base potential's EOS weight is
            # already aligned with the LLM's halting token.

        return next_token_weights
