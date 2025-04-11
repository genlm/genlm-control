import asyncio
import numpy as np
import torch
from genlm.control.potential.base import Potential
from genlm.control.constant import EOS


class Canonical(Potential):
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
        from genlm.control.tokenization.bpe2 import FastCanonicalityFilterBPE
        self.canonicality_filter = FastCanonicalityFilterBPE.from_huggingface(tokenizer)
        self.canonicality_filter.set_overrides(model_name)
        self.tokenizer = tokenizer
        self.eos = EOS
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
    