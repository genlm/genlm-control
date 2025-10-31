"""
Molecular Synthesis evaluation using character-level ByteLLM + AWRS.

This script evaluates a model's ability to generate valid drug-like molecules
in SMILES notation, constrained by the PartialSMILES grammar.
"""

import argparse
import asyncio
import gzip
import random
import sys
from pathlib import Path
from typing import List

# Add genlm-eval to path
sys.path.insert(0, str(Path(__file__).parent.parent / "genlm-eval"))

from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams
from genlm.control import ByteLLM, AWRS
from genlm.eval import ModelOutput, ModelResponse, run_evaluation
from genlm.eval.domains.molecular_synthesis import (
    MolecularSynthesisDataset,
    MolecularSynthesisEvaluator,
    default_prompt_formatter,
    PartialSMILES,
)


# Global variable to hold the LLM
BYTE_LLM = None


def build_bytelm(model_name: str = "gpt2", eos_tokens=None, heal=True):
    """Build a character-level ByteLLM with specified EOS tokens."""
    if eos_tokens is None:
        eos_tokens = {b"\n", b"\n\n"}
    
    llm = load_model_by_name(model_name, backend="hf")
    
    beam_params = BeamParams(
        K=16,
        prune_threshold=0.0,
        eos_tokens=eos_tokens,
        heal=heal,
    )
    byte_llm = ByteLLM(llm, beam_params)
    print(f"Loaded model: {model_name}")
    print(f"EOS tokens: {beam_params.eos_tokens}")
    return byte_llm


def load_smiles_file(smiles_path: str):
    """Load SMILES strings from a file, handling both plain text and gzipped files."""
    path = Path(smiles_path)
    if path.suffix == '.gz' or path.suffixes[-1] == '.gz':
        with gzip.open(smiles_path, 'rt') as f:
            return f.readlines()
    else:
        with open(smiles_path) as f:
            return f.readlines()


async def molecular_model_adaptor(instance, output_dir: str, replicate: int) -> ModelOutput:
    """Model adaptor for molecular synthesis task."""
    global BYTE_LLM

    # Format prompt and set on ByteLLM
    prompt_ids = default_prompt_formatter(
        BYTE_LLM.llm.tokenizer,
        instance,
        use_chat_format=False,  # Molecular synthesis doesn't support chat format yet
    )
    prompt_str = BYTE_LLM.llm.tokenizer.decode(prompt_ids)
    BYTE_LLM.set_prompt_from_str(prompt_str)

    # SMILES grammar constraint coerced to byte LM
    condition = PartialSMILES().coerce(BYTE_LLM, f=b"".join)

    sampler = AWRS(BYTE_LLM, condition)
    sequences = await sampler.smc(
        n_particles=5,
        ess_threshold=0.5,
        max_tokens=100,
        verbosity=0,
    )

    responses: List[ModelResponse] = [
        ModelResponse(response=seq, weight=float(prob))
        for seq, prob in sequences.decoded_posterior.items()
    ]
    
    # Cleanup beam/trie background tasks between instances
    await BYTE_LLM.cleanup()
    
    return ModelOutput(responses=responses, runtime_seconds=None)


async def main():
    parser = argparse.ArgumentParser(
        description="Molecular Synthesis eval with character-level model (ByteLLM + AWRS)"
    )
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)  # exclusive
    parser.add_argument("--skip_idxs", type=str, default="", help="Comma/space-separated indices to skip")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--n_molecules", type=int, default=20, help="Number of molecules per instance")
    parser.add_argument("--n_instances", type=int, default=5, help="Total instances in dataset")
    parser.add_argument("--dataset_path", type=str, default="/teamspace/studios/this_studio/genlm-control/GDB17.50000000.smi.gz")
    parser.add_argument("--heal", action="store_true", default=True, help="Enable healing for ByteLLM")
    parser.add_argument("--no-heal", dest="heal", action="store_false", help="Disable healing for ByteLLM")
    args = parser.parse_args()

    # Resolve dataset path
    dataset_path = Path(args.dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    global BYTE_LLM
    BYTE_LLM = build_bytelm(args.model, heal=args.heal)

    # Load dataset
    print(f"\nLoading molecular synthesis dataset from {dataset_path}")
    print(f"Sampling {args.n_instances} instances with {args.n_molecules} molecules each")
    
    # Load molecules from file (handles both plain and gzipped files)
    molecules = load_smiles_file(str(dataset_path))
    
    # Create dataset with sampling (same logic as from_smiles but handles gzipped files)
    random.seed(1234)
    prompt_molecules = []
    for _ in range(args.n_instances):
        molecule_ids = random.sample(range(len(molecules)), min(args.n_molecules, len(molecules)))
        prompt_molecules.append([molecules[i] for i in molecule_ids])
    
    dataset = MolecularSynthesisDataset(prompt_molecules)

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
    
    # Create a filtered dataset
    class FilteredDataset:
        def __init__(self, instances):
            self.instances = instances
            self.schema = MolecularSynthesisDataset.schema
        
        def __iter__(self):
            return iter(self.instances)
        
        def __len__(self):
            return len(self.instances)
    
    filtered_dataset = FilteredDataset(selected_instances)
    
    print(f"\nEvaluating on instances {args.start_idx} to {end_idx}")
    print(f"Total instances after filtering: {len(filtered_dataset)}")

    evaluator = MolecularSynthesisEvaluator()

    # Run evaluation
    results = await run_evaluation(
        dataset=filtered_dataset,
        model=molecular_model_adaptor,
        evaluator=evaluator,
        n_replicates=1,
        max_instances=args.max_instances if args.max_instances else float("inf"),
        verbosity=1,
    )

    print("\n" + "=" * 80)
    print(f"Final Results (model={args.model})")
    print("=" * 80)
    print(f"Average Weighted Accuracy (QED score): {results['average_weighted_accuracy']:.4f}")
    print(f"Number of instances evaluated: {results['n_instances']}")

    # Clean up ByteLLM beam state resources
    await BYTE_LLM.cleanup()


if __name__ == "__main__":
    asyncio.run(main())

