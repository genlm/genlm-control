# from genlm.backend import EOS
from genlm.control.potential import Potential
from genlm.backend import decode_vocab
from genlm.control.potential.built_in.llm import TokenMappings
from genlm.control.constant import EOS



class HarmonyChat:
    

    harmony_chat_keys = ["analysis", "final", "commentary","<|channel|>","<|message|>","<|end|>"]

    def __init__(self, tokenizer):

        """ 
        This class encodes the Harmony chat Format, and provdes the methods to extract the harmony channels from th econtext.
        Since it operates of the bytes representation of tokens it also provides methods to extract the byte represenattion from the token ids. 
        """

        self.tokenizer = tokenizer # The tokenizer should support the harmony chat format. otherwise we should throw an error 

        self.byte_vocab, _ = decode_vocab(tokenizer) # This is the byte representation of the tokenizer's token. Note that it follows the construction in the Backend.
        self.eos_tokens = [self.byte_vocab[tokenizer.eos_token_id]] #This matches the construction described in Prompted LLM.
        self.token_maps = TokenMappings.create(
            decode=self.byte_vocab,
            eos_tokens=self.eos_tokens
        )
        self.token_dict = {}
        for key in HarmonyChat.harmony_chat_keys: # This completes the token dict with the tokens that may change according to the tokenizer.
            self.token_dict[key] = self.decode_tokens(self.tokenizer.encode(key))
        

    def extract_channel_content(self, token_bytes, start_idx):
        """Extract content between start_idx and end_token."""
        content = []
        i = start_idx
        end_token = self.token_dict['<|end|>'][0]
        message_token = self.token_dict['<|message|>'][0]

        is_prefix = False

        while token_bytes[i] != message_token:
            i+=1
            if i >= len(token_bytes): #If we have not yet entered the channel content yet, we return none.
                return None
        i+=1
        while True:
            if len(token_bytes[i:]) == 0:
                is_prefix = True
                break
            elif token_bytes[i] == end_token: # Or EOS token?
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
        analysis_tokens = self.token_dict['analysis']
        final_tokens = self.token_dict['final'] 
        commentary_tokens = self.token_dict['commentary']
        channel_token = self.token_dict['<|channel|>'][0]  # Always a single token.
        
        results = {
            'analysis': None,
            'final': None, 
            'commentary': None
        }
        
        # Find all channel positions
        i = 0
        while i < len(token_bytes)-1:
            # Look for channel token followed by analysis/final/commentary
            if token_bytes[i] == channel_token:
                # Check what channel type follows
                if token_bytes[i+1] == analysis_tokens[0]:
                    results['analysis'] = self.extract_channel_content(token_bytes, i)
                elif token_bytes[i+1] == final_tokens[0]:
                    results['final'] = self.extract_channel_content(token_bytes, i)
                elif token_bytes[i+1] == commentary_tokens[0]:
                    results['commentary'] = self.extract_channel_content(token_bytes, i)
                else:
                    raise ValueError(f"Unexpected channel: {token_bytes[i+1]}")
                        
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
        assert self.token_maps is not None, "Token maps must be initialized to call encode_tokens"
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
        assert all(isinstance(x, int) for x in ids), "Token IDs must be integers"
        assert self.token_maps is not None, "Token maps must be initialized to call decode_tokens"
        return [self.token_maps.decode[x] for x in ids]


class HarmonyPotential(Potential):

    def __init__(self, base_potential, llm_tokenizer, constrained_channels=["analysis", "final", "commentary"]):
        """ 
        Inputs: 
            Base Potential: a base potential which is applie dto the context channels.
            llm_tokenizer: a tokenizer of a language model that supports the harmony chat format. NB: we need to verify wether the format is still evolving or not.
            Constrained Channels: A list of channels to which the base potential is applied.
        Importantly, for compatibility with the genlm library, we assume that the tokens are represented as bytes.
        """ # Need to adapt the coerce method so that it does not prune the vocabulary -->(This would cause an error when sampling from channels that we do not want to constrain)

        self.base_potential = base_potential
        self.harmony_chat = HarmonyChat(llm_tokenizer)
        self.constrained_channels = constrained_channels
        
        super().__init__(self.harmony_chat.byte_vocab) # is this the right vocab, or should it rather be the llm's?
            

    async def complete(self, context):
        """
        Input: a list of bytes tokens.
        The Log probability of the constrained channels of the context. 
        To each context we apply the complete potential, as if the string was completed.
        """
        channels = self.harmony_chat.extract_harmony_channels_from_tokens(context) # Extract the channels from the context. #Note we may need to patch this so that the EOS of the Potential is paired with the <end> token.

        log_weight = 0
        for key in channels:
            if channels[key] is not None and key in self.constrained_channels:
                log_weight += await self.base_potential.complete(channels[key]['content']) # Is this the right way to treat the complete potentials? I need to check.
                
        print(f"complete chat -- context: {context}, log_weight: {log_weight}")
        return log_weight

    async def prefix(self, context):
        """
        Input: A list of byte tokens in bytes format. 
        Note that each channel is channel to be constrained is extracted from th econtext
        and depending on weather the string is complere or not, the complete or prefix potential is applied. 
        Output: The sum of the log probabilities of the constrained channels, according to the base potential.
        If one of the channels is not marked as complete, it is evaluated as a prefix according to the base potential.
        """
        channels = self.harmony_chat.extract_harmony_channels_from_tokens(context) # Extract the channels from the context.

        log_weight = 0
        for key in channels:
            if channels[key] is not None and key in self.constrained_channels:
                if channels[key]['is_prefix']: #
                    # This needs to be adapted, so that the EOS token of the potential matches the <end> token.
                    log_weight += await self.base_potential.prefix(channels[key]['content'])
                else: # Note that multiple channels cannot have the "is_prefix" status at the same time.
                    # I don't think that right now we need to take into account the special EOS case, right now, as this will be taken into account
                    # by the Harmony chat format directly. However, we may nee to handle this differently in the next-tokens function.
                    log_weight += await self.base_potential.complete(channels[key]['content'])

        print(f"prefix Potential -- log_weight: {log_weight}")
        return log_weight
