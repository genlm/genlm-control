"""
COLLIE evaluation using token-level PromptedLLM + AWRS.

This script evaluates a model's ability to generate constrained text according
to COLLIE benchmark constraints.
"""

import argparse
import asyncio
import dill
import sys
from pathlib import Path
from typing import List, Optional

# Add genlm-eval to path
sys.path.insert(0, str(Path(__file__).parent.parent / "genlm-eval"))
sys.path.insert(0, str(Path(__file__).parent.parent / "Collie"))

from genlm.control import PromptedLLM, AWRS, Potential
from genlm.eval.core import run_evaluation
from genlm.eval.core.model import ModelOutput, ModelResponse
from genlm.eval.core.dataset import Instance, Dataset


# Global variable to hold the LLM
TOKEN_LLM = None


class CollieConstraintPotential(Potential):
    """Potential wrapper for COLLIE constraints that works with token sequences."""
    
    def __init__(self, constraint, targets, tokenizer):
        """
        Args:
            constraint: COLLIE Constraint object
            targets: Target values for constraint checking
            tokenizer: Tokenizer to decode token sequences to text
        """
        # Initialize with a dummy vocabulary - will be coerced to actual LLM vocab
        super().__init__(vocabulary=[b''])
        self.constraint = constraint
        self.targets = targets
        self.tokenizer = tokenizer
    
    async def complete(self, context):
        """Check if complete sequence satisfies the constraint.
        
        Args:
            context: Sequence of tokens
            
        Returns:
            Log weight: 0.0 if satisfied, -inf otherwise
        """
        try:
            # Decode token sequence to text
            if isinstance(context[0], bytes):
                text = b''.join(context).decode('utf-8', errors='ignore')
            else:
                # Assume token IDs
                text = self.tokenizer.decode(context, skip_special_tokens=True)
            # Check constraint
            is_satisfied = self.constraint.check(text, self.targets)
            return 0.0 if is_satisfied else float('-inf')
        except Exception:
            return float('-inf')
    
    async def prefix(self, context):
        """Check if prefix could potentially satisfy constraint.
        
        For now, we allow all prefixes (optimistic approach).
        Args:
            context: Sequence of tokens
            
        Returns:
            Log weight: 0.0 (allow all prefixes)
        """
        return 0.0
    
    async def logw_next(self, context):
        """Compute next-token log weights.
        
        Since we can't predict constraint satisfaction per token,
        we allow all tokens equally.
        
        Args:
            context: Current token sequence
            
        Returns:
            LazyWeights for next tokens
        """
        from genlm.control.util import LazyWeights
        # Allow all tokens equally - constraint checking happens at complete()
        return LazyWeights([0.0] * len(self.vocab_eos))
    
    def coerce(self, llm, f):
        """Coerce to work with token-level LLM."""
        # Create new instance with LLM's vocabulary
        coerced = CollieConstraintPotential(self.constraint, self.targets, self.tokenizer)
        coerced.vocab = llm.vocab
        coerced.vocab_eos = llm.vocab_eos
        coerced.lookup = llm.lookup
        coerced.token_type = llm.token_type
        coerced.eos = llm.eos
        coerced.tokenizer = llm.model.tokenizer  # Use LLM's tokenizer
        return coerced


class CollieInstance(Instance):
    """Schema for COLLIE instance."""
    
    prompt: str
    constraint: object  # COLLIE Constraint object
    targets: object  # Target values
    example: str  # Original example text
    constraint_key: str  # Key like 'wiki_c14'
    instance_id: int
    
    def __repr__(self):
        return f"CollieInstance(id={self.instance_id}, key={self.constraint_key})"


class CollieDataset(Dataset[CollieInstance]):
    """Dataset wrapper for COLLIE data."""
    
    def __init__(self, all_data: dict):
        """
        Args:
            all_data: Dictionary from COLLIE all_data.dill
        """
        self.instances = []
        instance_id = 0
        for key, examples in all_data.items():
            for example in examples:
                inst = CollieInstance(
                    prompt=example['prompt'],
                    constraint=example['constraint'],
                    targets=example['targets'],
                    example=example['example'],
                    constraint_key=key,
                    instance_id=instance_id,
                )
                self.instances.append(inst)
                instance_id += 1
    
    def __len__(self):
        return len(self.instances)
    
    def __iter__(self):
        return iter(self.instances)
    
    @property
    def schema(self):
        return CollieInstance


class CollieEvaluator:
    """Evaluator for COLLIE constraints."""
    
    def evaluate_sample(self, instance: CollieInstance, response: str) -> dict:
        """Evaluate if response satisfies the constraint.
        
        Args:
            instance: COLLIE instance
            response: Generated text response
            
        Returns:
            Evaluation result dictionary
        """
        is_valid = instance.constraint.check(response, instance.targets)
        return {
            'score': 1.0 if is_valid else 0.0,
            'desc': 'valid' if is_valid else 'invalid'
        }
    
    def evaluate(self, instance: CollieInstance, responses: List[ModelResponse]) -> dict:
        """Evaluate weighted responses.
        
        Args:
            instance: COLLIE instance
            responses: List of ModelResponse objects with weights
            
        Returns:
            Evaluation result dictionary
        """
        weighted_score = sum(
            resp.weight * self.evaluate_sample(instance, resp.response)['score']
            for resp in responses
        )
        return {
            'score': weighted_score,
            'weighted_score': weighted_score,
            'valid': weighted_score > 0.5  # Threshold for validity
        }


def build_token_llm(model_name: str = "meta-llama/Llama-3.2-1B-Instruct") -> PromptedLLM:
    """Build a token-level PromptedLLM with specified EOS tokens."""
    llm = PromptedLLM.from_name(
        model_name,
        backend="hf",
        eos_tokens=[b"\n", b"\n\n", b"<|end_of_text|>", b"<|eot_id|>"],
        temperature=1.0,
    )
    
    # Set pad_token to avoid batching issues
    if llm.model.tokenizer.pad_token is None:
        if llm.model.tokenizer.eos_token:
            llm.model.tokenizer.pad_token = llm.model.tokenizer.eos_token
        else:
            llm.model.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    llm.model.tokenizer.padding_side = 'left'
    
    print(f"Loaded model: {model_name}")
    print(f"EOS tokens: {llm.eos_tokens}")
    return llm


async def collie_model_adaptor(instance: CollieInstance, output_dir: str, replicate: int) -> ModelOutput:
    """Model adaptor for COLLIE task."""
    global TOKEN_LLM
    
    # Tokenize prompt
    prompt_ids = TOKEN_LLM.model.tokenizer.encode(instance.prompt, return_tensors='pt')
    TOKEN_LLM.prompt_ids = prompt_ids[0].tolist()
    
    # Create constraint potential
    condition = CollieConstraintPotential(instance.constraint, instance.targets)
    condition = condition.coerce(TOKEN_LLM, f=b"".join)
    
    sampler = AWRS(TOKEN_LLM, condition)
    sequences = await sampler.smc(
        n_particles=args.n_particles,
        ess_threshold=args.ess_threshold,
        max_tokens=args.max_tokens,
        verbosity=0,
    )
    
    responses: List[ModelResponse] = [
        ModelResponse(response=seq, weight=float(prob))
        for seq, prob in sequences.decoded_posterior.items()
    ]
    return ModelOutput(responses=responses, runtime_seconds=None)


async def main():
    parser = argparse.ArgumentParser(
        description="COLLIE eval with token-level model (PromptedLLM + AWRS)"
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)  # exclusive
    parser.add_argument("--skip_idxs", type=str, default="", help="Comma/space-separated indices to skip")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--ess_threshold", type=float, default=0.5, help="ESS threshold for resampling")
    parser.add_argument("--n_particles", type=int, default=5, help="Number of particles for SMC")
    parser.add_argument("--max_tokens", type=int, default=200, help="Maximum tokens to generate")
    parser.add_argument("--dataset_path", type=str, default="/teamspace/studios/this_studio/Collie/data/all_data.dill")
    parser.add_argument("--constraint_filter", type=str, default=None, help="Filter by constraint key (e.g., 'wiki_c14')")
    parser.add_argument("--use_chat_format", action="store_true", help="Use chat template formatting")
    args = parser.parse_args()
    
    global TOKEN_LLM
    TOKEN_LLM = build_token_llm(args.model)
    
    # Load COLLIE dataset
    print(f"\nLoading COLLIE dataset from {args.dataset_path}")
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        all_data = dill.load(f)
    
    # Filter by constraint key if specified
    if args.constraint_filter:
        filtered_data = {k: v for k, v in all_data.items() if args.constraint_filter in k}
        if not filtered_data:
            print(f"Warning: No data found for filter '{args.constraint_filter}'")
            return
        all_data = filtered_data
        print(f"Filtered to constraint keys containing '{args.constraint_filter}': {list(all_data.keys())}")
    
    # Create dataset
    dataset = CollieDataset(all_data)
    print(f"Total instances in dataset: {len(dataset)}")
    
    # Parse skip indices
    skip_idxs = set()
    if args.skip_idxs:
        skip_str = args.skip_idxs.replace(",", " ")
        skip_idxs = {int(x) for x in skip_str.split() if x.strip()}
        print(f"Skipping indices: {sorted(skip_idxs)}")
    
    # Slice dataset
    all_instances = list(dataset)
    end_idx = args.end_idx if args.end_idx is not None else len(all_instances)
    selected_instances = [
        inst for i, inst in enumerate(all_instances[args.start_idx:end_idx], start=args.start_idx)
        if i not in skip_idxs
    ]
    
    print(f"\nEvaluating on instances {args.start_idx} to {end_idx}")
    print(f"Total instances after filtering: {len(selected_instances)}")
    print(f"n_particles: {args.n_particles}, ess_threshold: {args.ess_threshold}, max_tokens: {args.max_tokens}")
    
    evaluator = CollieEvaluator()
    
    # Modify the adaptor to use command-line args
    async def adaptor_with_args(instance, output_dir: str, replicate: int) -> ModelOutput:
        global TOKEN_LLM
        prompt_ids = TOKEN_LLM.model.tokenizer.encode(instance.prompt, return_tensors='pt')
        TOKEN_LLM.prompt_ids = prompt_ids[0].tolist()
        
        condition = CollieConstraintPotential(
            instance.constraint, 
            instance.targets,
            TOKEN_LLM.model.tokenizer
        )
        condition = condition.coerce(TOKEN_LLM, f=b"".join)
        
        sampler = AWRS(TOKEN_LLM, condition)
        sequences = await sampler.smc(
            n_particles=args.n_particles,
            ess_threshold=args.ess_threshold,
            max_tokens=400,
            verbosity=0,
        )
        
        responses: List[ModelResponse] = [
            ModelResponse(response=seq, weight=float(prob))
            for seq, prob in sequences.decoded_posterior.items()
        ]
        return ModelOutput(responses=responses, runtime_seconds=None)
    
    # Custom evaluation loop since COLLIE evaluator doesn't match genlm-eval interface exactly
    print("\n" + "=" * 80)
    print("Starting evaluation...")
    print("=" * 80)
    
    total_score = 0.0
    n_valid = 0
    
    for i, instance in enumerate(selected_instances):
        if args.max_instances and i >= args.max_instances:
            break
        
        print(f"\n[{i+1}/{len(selected_instances)}] Instance {instance.instance_id} ({instance.constraint_key})")
        print(f"Prompt: {instance.prompt[:100]}...")
        
        try:
            output = await adaptor_with_args(instance, None, 0)
            eval_result = evaluator.evaluate(instance, output.responses)
            
            total_score += eval_result['weighted_score']
            if eval_result['valid']:
                n_valid += 1
            
            print(f"  Weighted score: {eval_result['weighted_score']:.4f}")
            print(f"  Valid: {eval_result['valid']}")
            print(f"  Generated: {output.responses[0].response[:100] if output.responses else 'None'}...")
            
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    n_evaluated = min(len(selected_instances), args.max_instances or len(selected_instances))
    avg_score = total_score / n_evaluated if n_evaluated > 0 else 0.0
    
    print("\n" + "=" * 80)
    print(f"Final Results (model={args.model}, ess_threshold={args.ess_threshold})")
    print("=" * 80)
    print(f"Average Weighted Score: {avg_score:.4f}")
    print(f"Valid instances: {n_valid}/{n_evaluated}")
    print(f"Number of instances evaluated: {n_evaluated}")


if __name__ == "__main__":
    asyncio.run(main())

