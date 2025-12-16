from genlm.control.potential import Potential
from genlm.backend import decode_vocab
from genlm.control.potential.built_in.llm import TokenMappings
import numpy as np
import warnings


class HarmonyChat:
    """
    This class encodes the "assistant" field of the Harmony chat Format,
    and provides the methods to extract the harmony channels from the context.

    Since it operates on the byte representation of tokens it also provides
    methods to extract the byte representation from the token ids.
    """


    def __init__(self, tokenizer):
        """
        Initialize HarmonyChat with a tokenizer.

        Args:
            tokenizer: A tokenizer that supports the harmony chat format.
                The tokenizer must be able to encode the harmony chat tokens
                as single tokens.

        """
        # Check that the tokenizer object has the minimum required methods:
        assert (
            hasattr(tokenizer, "encode")
            and hasattr(tokenizer, "decode")
            and hasattr(tokenizer, "apply_chat_template")
        ), (
            "The tokenizer object does not have the minimum required methods or attributes."
        )
        assert all(
            len(tokenizer.encode(key)) == 1
            for key in [
                "<|channel|>",
                "<|message|>",
                "<|end|>",
            ]
        ), (
            "The tokenizer does not appear to support the harmony chat format, or does not support the specific format of the current implementation(gpt-oss, August 2025)."
        )

        self.tokenizer = tokenizer
        _byte_vocab, _ = decode_vocab(
            tokenizer
        )  # This is the byte representation of the tokenizer's token. Note that it follows the same schema of PromptedLLM.
        _eos_tokens = [
            _byte_vocab[
                tokenizer.eos_token_id
            ]  # for gpt-oss, this is the <|return|> token.
        ]

        self.token_maps = TokenMappings.create(
            decode=_byte_vocab, eos_tokens=_eos_tokens
        )
        self.potential_vocab = self.token_maps.potential_vocab

        # We encode the special tokens that are needed in harmony as instance variables.
        self.end_token = self.decode_tokens(self.tokenizer.encode("<|end|>"))[0]
        self.message_token = self.decode_tokens(self.tokenizer.encode("<|message|>"))[0]
        self.channel_token = self.decode_tokens(self.tokenizer.encode("<|channel|>"))[0]
        self.analysis_tokens = self.decode_tokens(self.tokenizer.encode("analysis")) # The following tokens (analysis, commentary, final) are not reserved, and therefore they are not guaranteed to be single tokens.
        self.final_tokens = self.decode_tokens(self.tokenizer.encode("final"))
        self.commentary_tokens = self.decode_tokens(self.tokenizer.encode("commentary"))

    def extract_channel_content(self, token_bytes, i):
        """Extract content between start_idx and end_token."""
       
        if i >= len(token_bytes):
            return None
        if token_bytes[i] != self.message_token:
            raise ValueError(f"Message token expected after the channel type declaration, but got {token_bytes[i]}")

        i += 1
        if self.end_token in token_bytes[i:]:
            end_position = token_bytes.index(self.end_token, i)
            content = token_bytes[i:end_position]
            is_prefix = False
        else:
            content = token_bytes[i:]
            is_prefix = True

        return {"content": content, "is_prefix": is_prefix}

    def extract_harmony_channels_from_tokens(self, token_bytes):
        """
        Extract analysis, final, and commentary content from token IDs.

        Args:
            token_bytes: List of token IDs

        Returns:
            Dictionary with extracted channel contents
        """

        results = {"analysis": None, "final": None, "commentary": None}

        for i, token in enumerate(token_bytes[:-2]):
            # The harmony format assumes that the <|channel|> token is immediately followed by the channel type, thus we can stop before the last two tokens.
            # Look for <|channel|> token followed by analysis/final/commentary.
            if token == self.channel_token:
                j = i + 1
                # Check wheter the analysis, final or commentary tokens follow the channel opening.
                if len(token_bytes) >= j + len(self.analysis_tokens) \
                    and token_bytes[j : j + len(self.analysis_tokens)] == self.analysis_tokens:
                    results["analysis"] = self.extract_channel_content(token_bytes, j + len(self.analysis_tokens)) # TODO: simplify the extract_channel_content field as we can directly check that the next token is the <|end|> token.
                elif len(token_bytes) >= j + len(self.final_tokens) and\
                    token_bytes[j : j + len(self.final_tokens)] == self.final_tokens:
                    results["final"] = self.extract_channel_content(token_bytes, j + len(self.final_tokens))
                elif len(token_bytes) >= j + len(self.commentary_tokens) and \
                    token_bytes[j : j + len(self.commentary_tokens)] == self.commentary_tokens:
                    results["commentary"] = self.extract_channel_content(token_bytes, j + len(self.commentary_tokens))
                elif len(token_bytes) < j + len(self.analysis_tokens) \
                    and len(token_bytes) < j + len(self.final_tokens) \
                    and len(token_bytes) < j + len(self.commentary_tokens): # This means that there is not enough tokens left to extract the channel content.
                    continue 
                else: # In this case, it means that the channel is not valid.
                    raise ValueError(
                        f"Unexpected channel: {token_bytes[j]}"
                    )  # pragma: no cover

        return results

    def extract_harmony_channels_from_string(self, string, add_special_tokens=False):
        """
        Extract analysis, final, and commentary content from a string.
        Uses the tokenizer to map from string to token ids and from token ids to token bytes.
        Then calls the "extract_harmony_channels_from_tokens" method.

        Args:
        string: The harmony chat format string to extract channels from
        add_special_tokens: Whether to add special tokens during encoding (default: False)

        Returns:
            Dictionary with extracted channel contents (same format as extract_harmony_channels_from_tokens)
        """
        token_ids = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        token_bytes = self.decode_tokens(token_ids)
        return self.extract_harmony_channels_from_tokens(token_bytes)

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
        assert all(isinstance(x, bytes) for x in tokens), "Tokens must be bytes"
        try:
            return [self.token_maps.encode[x] for x in tokens]
        except KeyError as e:  # pragma: no cover
            raise ValueError(
                f"Token {e.args[0]} not in vocabulary"
            ) from e  # pragma: no cover

    def decode_tokens(self, ids):
        """
        Decode a list of token IDs in the language model's vocabulary to a list of byte tokens.

        Args:
            ids (list[int]): A list of token IDs in the language model's vocabulary.

        Returns:
            (list[bytes]): A list of byte tokens corresponding to the input token IDs.
        """
        assert all(isinstance(x, int) for x in ids), "Token IDs must be integers"
        return [self.token_maps.decode[x] for x in ids]


class HarmonyPotential(Potential):
    def __init__(self, base_potential, llm_tokenizer, constrained_channels=None):
        """
        Inputs:
            Base Potential: a base potential which is applied to the context channels.
            llm_tokenizer: a tokenizer of a language model that supports the harmony chat format. NB: we need to verify whether the format is still evolving or not.
            Constrained Channels: A list of channels to which the base potential is applied.
        Importantly, for compatibility with the genlm library, we assume that the tokens are represented as bytes.
        """  # Need to adapt the coerce method so that it does not prune the vocabulary -->(This would cause an error when sampling from channels that we do not want to constrain)

        self.base_potential = base_potential
        self.harmony_chat = HarmonyChat(llm_tokenizer)
        self.constrained_channels = constrained_channels or [] # default to empty list if no channels are provided.

        super().__init__(self.harmony_chat.potential_vocab)

        if not set(base_potential.vocab) <= set(self.vocab):
            warnings.warn(  # pragma: no cover
                "The base potential's vocabulary must be a subset of the harmony potential's vocabulary."
            )

    def set_constrained_channels(self, constrained_channels):
        """
        A list of channels to be constrained.
        """
        assert isinstance(constrained_channels, list) and all(
            x in {"analysis", "final", "commentary"} for x in constrained_channels
        ), "Constrained channels must be one of analysis, final, or commentary."
        self.constrained_channels = constrained_channels

    async def complete(self, context):
        """
        Input: a list of bytes tokens.
        The Log probability of the constrained channels of the context.
        To each context we apply the complete potential, as if the string was completed.
        """
        channels = self.harmony_chat.extract_harmony_channels_from_tokens(
            context
        )  # Extract the channels from the context.

        log_weight = (
            0  # Note that if no channel to be constrained is detected we return 0
        )
        for key in channels:
            if channels[key] is not None and key in self.constrained_channels:
                log_weight += await self.base_potential.complete(
                    channels[key]["content"]
                )  # We accumulate in log space the product of all the channels to be constrained.
                # Note that we don't care whether the channel is marked as complete.
                # In fact if the content of the channels to be constrained is not a complete valid string
                # the complete potential should be -inf.

        return log_weight

    async def prefix(self, context):
        """
        Input: A list of byte tokens in bytes format.
        Note that each channel to be constrained is extracted from the context.
        and depending on whether the string is complete or not, the complete or prefix potential is applied.
        Output: The sum of the log probabilities of the constrained channels, according to the base potential.
        If one of the channels is not marked as complete, it is evaluated as a prefix according to the base potential.
        """
        channels = self.harmony_chat.extract_harmony_channels_from_tokens(
            context
        )  # Extract the channels from the context.
        log_weight = 0
        for key in channels:
            if channels[key] is not None and key in self.constrained_channels:
                if channels[key]["is_prefix"]:
                    log_weight += await self.base_potential.prefix(
                        channels[key]["content"]
                    )
                else:  # Note that to compute the prefix weight, we alos accumulate the weight of the channels that have been completed so far.
                    log_weight += await self.base_potential.complete(
                        channels[key]["content"]
                    )
        return log_weight

    async def logw_next(self, context):
        """
        Input: A list of byte tokens in bytes format.
        Output: The sum of log probabilities for the next token logprobs for each possible next-token, including the EOS symbol.
        """

        channels = self.harmony_chat.extract_harmony_channels_from_tokens(
            context
        )  # Extract the channels from the context.

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
            return next_token_weights  # If there are no incomplete channels,\ pragma: no cover
            # we can return the weights as is. Every possible next token is valid for the harmony format.

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
            ).weights  # This should directly return the normalized next-token log weights.
            EOS_weight = next_token_weights.weights[-1]  # Keep track of the EOS weight.

            if (
                key == "analysis" or key == "commentary"
            ):  # In this case, the EOS weight needs to be moved to the <|end|> token.
                idx = next_token_weights.encode[self.harmony_chat.end_token]
                next_token_weights.weights[idx] = (
                    EOS_weight  # Move the EOS weight to the <|end|>, which is the token that will be sampled by the llm to close the channel.
                )
                next_token_weights.weights[-1] = float(
                    "-inf"
                )  # Set the EOS weight to -inf.
            elif (
                key == "final"
            ):  # If the channel is final, the prompted LLM automatically moves the weight of <|return|> to EOS. so we don't have to do anything else.
                pass

        return next_token_weights  # we return the normalized next-token weights.
