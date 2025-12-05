# from genlm.backend import EOS
from genlm.control.potential import Potential
from genlm.backend import decode_vocab
from genlm.control.potential.built_in.llm import TokenMappings
import numpy as np
import warnings

class HarmonyChat: # This dictionary encodes the channel of the Harmony chat format. Note that this is independent from the tokenizer.
    harmony_chat_keys = [
        "analysis",
        "final",
        "commentary",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
    ]  # Note: we may also want to include here the <|return|> token, which is used to tag the end of the final channel message.
    # Additionally this is required in order to correctly mark the complete status of the final channel in the output.
    # Unfortunately, this is currently not possible as the sampler artificially "swaps" the logits of the END with those of the
    # LM's eos token (in this case, <|return|>).

    def __init__(self, tokenizer):
        """
        This class encodes the Harmony chat Format, and provides the methods to extract the harmony channels from the context.
        Since it operates on the byte representation of tokens it also provides methods to extract the byte representation from the token ids.
        """

        self.tokenizer = tokenizer  # The tokenizer should support the harmony chat format. otherwise we should throw an error.

        _byte_vocab, _ = decode_vocab(
            tokenizer
        )  # This is the byte representation of the tokenizer's token. Note that it follows the construction in the Backend.
        _eos_tokens = [
            _byte_vocab[tokenizer.eos_token_id]
        ]  # This matches the construction described in Prompted LLM.

        self.token_maps = TokenMappings.create(
            decode=_byte_vocab, eos_tokens=_eos_tokens
        )
        self.potential_vocab = self.token_maps.potential_vocab
        self.token_dict = {}
        for key in HarmonyChat.harmony_chat_keys:  # This completes the token dict with the tokens that may change according to the tokenizer.
            self.token_dict[key] = self.decode_tokens(self.tokenizer.encode(key))

    def extract_channel_content(self, token_bytes, start_idx):
        """Extract content between start_idx and end_token."""
        content = []
        i = start_idx
        end_token = self.token_dict["<|end|>"][0]
        message_token = self.token_dict["<|message|>"][0]

        is_prefix = False

        while (
            token_bytes[i] != message_token
        ):  # Iterate until we find the <|message|> token # Right now we assume that token sequence is in the right harmony sequence. We may need to add an assertion to ensure this.
            i += 1
            if (
                i >= len(token_bytes)
            ):  # If we reach the string end without having entered the channel content, it means, that there is no content and we can return.
                return None # pragma: no cover
        i += 1
        while True:
            if len(token_bytes[i:]) == 0:
                is_prefix = True
                break
            elif (
                token_bytes[i] == end_token
            ):  # Or EOS token? This is an important point: do we always assume that the final chat contains the <|end|> token as part of the final result?
                break
            content.append(token_bytes[i])
            i += 1

        return {"content": content, "is_prefix": is_prefix}

    def extract_harmony_channels_from_tokens(self, token_bytes):
        """
        Extract analysis, final, and commentary content from token IDs.

        Args:
            token_ids: List of token IDs
            token_dict: Dictionary with token mappings

        Returns:
            Dictionary with extracted channel contents
        """
        analysis_tokens = self.token_dict["analysis"]
        final_tokens = self.token_dict["final"]
        commentary_tokens = self.token_dict["commentary"]
        channel_token = self.token_dict["<|channel|>"][0]  # Always a single token.

        results = {"analysis": None, "final": None, "commentary": None}

        # Find all channel positions
        i = 0
        while (
            i < len(token_bytes) - 2
        ):  # The harmony format assumes that channel is immediately followed by the channel type, thus we can stop before the last two tokens.
            # Look for <|channel|> token followed by analysis/final/commentary.
            if token_bytes[i] == channel_token:
                # Check what channel follows.
                if token_bytes[i + 1] == analysis_tokens[0]:
                    results["analysis"] = self.extract_channel_content(token_bytes, i)
                elif token_bytes[i + 1] == final_tokens[0]:
                    results["final"] = self.extract_channel_content(token_bytes, i)
                elif token_bytes[i + 1] == commentary_tokens[0]:
                    results["commentary"] = self.extract_channel_content(token_bytes, i)
                else:
                    raise ValueError(f"Unexpected channel: {token_bytes[i + 1]}") #pragma: no cover

            i += 1

        return results

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
        assert self.token_maps is not None, (
            "Token maps must be initialized to call encode_tokens"
        )
        try:
            return [self.token_maps.encode[x] for x in tokens]
        except KeyError as e: # pragma: no cover
            raise ValueError(f"Token {e.args[0]} not in vocabulary") from e #pragma: no cover

    def decode_tokens(self, ids):
        """
        Decode a list of token IDs in the language model's vocabulary to a list of byte tokens.

        Args:
            ids (list[int]): A list of token IDs in the language model's vocabulary.

        Returns:
            (list[bytes]): A list of byte tokens corresponding to the input token IDs.
        """
        assert all(isinstance(x, int) for x in ids), "Token IDs must be integers"
        assert self.token_maps is not None, (
            "Token maps must be initialized to call decode_tokens"
        )
        return [self.token_maps.decode[x] for x in ids]


class HarmonyPotential(Potential):
    def __init__(self, base_potential, llm_tokenizer, constrained_channels=[]):
        """
        Inputs:
            Base Potential: a base potential which is applied to the context channels.
            llm_tokenizer: a tokenizer of a language model that supports the harmony chat format. NB: we need to verify whether the format is still evolving or not.
            Constrained Channels: A list of channels to which the base potential is applied.
        Importantly, for compatibility with the genlm library, we assume that the tokens are represented as bytes.
        """  # Need to adapt the coerce method so that it does not prune the vocabulary -->(This would cause an error when sampling from channels that we do not want to constrain)

        self.base_potential = base_potential
        self.harmony_chat = HarmonyChat(llm_tokenizer)
        self.constrained_channels = constrained_channels

        super().__init__(
            self.harmony_chat.potential_vocab
        )  # is this the right vocab, or should it rather be the llm's?

        if not set(base_potential.vocab) <= set(self.vocab):
            warnings.warn( # pragma: no cover
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
        )  # Extract the channels from the context. #Note we may need to patch this so that the EOS of the Potential is paired with the <end> token.

        log_weight = (
            0  # Note that if no channel to be constrained is detected we return 0
        )
        for key in channels:
            if channels[key] is not None and key in self.constrained_channels:
                log_weight += await self.base_potential.complete(
                    channels[key]["content"]
                )  # Is this the right way to treat the complete potentials? I need to check.

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
                if channels[key]["is_prefix"]:  # Note: this
                    # This may need to be adapted, so that the EOS token of the potential matches the <end> token.
                    log_weight += await self.base_potential.prefix(
                        channels[key]["content"]
                    )
                else:  # If the channel has the "is_prefix" flag set to false, it means that it has the <|end|> token
                    # in the end, which implies that we can call the complete potential (<|END|> ~ EOS )
                    log_weight += await self.base_potential.complete(
                        channels[key]["content"]
                    )

        return log_weight

    async def logw_next(self, context):
        """
        Input: A list of byte tokens in bytes format.
        Output: The sum of log probabilities for the next token logprobs for each possible next-token, including the EOS symbol.
        """

        end_token = self.harmony_chat.token_dict["<|end|>"][0]
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
                raise ValueError( # pragma: no cover
                    f"Context {channels[key]['content']!r} has weight zero under `prefix`."
                )
            next_token_weights.weights += (
                await self.base_potential.logw_next(channels[key]["content"])
            ).weights  # This should directly return the normalized next-token log weights.
            EOS_weight = next_token_weights.weights[-1]  # Keep track of the EOS weight.

            if (
                key == "analysis" or key == "commentary"
            ):  # In this case, the EOS weight needs to be moved to the <|end|> token.
                idx = next_token_weights.encode[end_token]
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
