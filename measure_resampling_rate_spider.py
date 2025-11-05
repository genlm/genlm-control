"""
Script to measure resampling rate for character-level vs token-level models on Spider (text2sql).

This script runs SMC on Spider tasks and tracks how often resampling occurs.
Resampling rate = number of resampling events / total steps.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add genlm-eval to path
sys.path.insert(0, str(Path(__file__).parent.parent / "genlm-eval"))

from genlm.control import PromptedLLM, ByteLLM, AWRS
from genlm.bytes import BeamParams
from genlm.backend import load_model_by_name
from genlm.eval.domains.spider.spider import (
    SpiderDataset,
    default_prompt_formatter,
)
from genlm.eval.domains.spider.table_column_potential import SpiderTableColumnVerifier


# Default paths
SPIDER_DATA_DIR = "/teamspace/studios/this_studio/spider_data"
SPIDER_GRAMMARS = "/teamspace/studios/this_studio/genlm-eval/assets/spider/grammars.json"


async def measure_token_resampling(
    model_name: str,
    n_particles: int = 5,
    ess_threshold: float = 0.5,
    max_tokens: int = 100,
    n_instances: int = 10,
):
    """Measure resampling rate for token-level model on Spider."""
    print(f"\n{'='*60}")
    print(f"Token-Level Model: {model_name}")
    print(f"{'='*60}")
    
    # Build token LLM
    llm = PromptedLLM.from_name(
        model_name,
        backend="hf",
        eos_tokens=[b"\n", b"\n\n", b"<|end_of_text|>", b"<|eot_id|>"],
        temperature=1.0,
    )
    
    # Load dataset
    dataset = SpiderDataset.from_spider_dir(
        SPIDER_DATA_DIR,
        grammar_json_path=SPIDER_GRAMMARS,
        few_shot_example_ids=[],
    )
    
    all_stats = []
    
    # Run on first few instances
    for i, instance in enumerate(list(dataset)[:n_instances]):
        print(f"\nInstance {i+1}/{n_instances}")
        
        # Set up prompt
        prompt_ids = default_prompt_formatter(
            llm.model.tokenizer,
            instance,
            use_chat_format=True,
        )
        llm.prompt_ids = prompt_ids
        
        # Set up constraint
        condition = SpiderTableColumnVerifier(
            grammar=instance.lark_grammar,
            tables=instance.tables,
        ).coerce(llm, f=b"".join)
        
        # Use JSON output to track resampling
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            sampler = AWRS(llm, condition)
            sequences = await sampler.smc(
                n_particles=n_particles,
                ess_threshold=ess_threshold,
                max_tokens=max_tokens,
                verbosity=0,
                json_path=json_path,
            )
            
            # Parse JSON to count resampling events
            try:
                with open(json_path) as f:
                    smc_data = json.load(f)
                
                # Count resampling events
                resampling_events = 0
                total_steps = 0
                
                if isinstance(smc_data, list):
                    for step in smc_data:
                        # Count all steps (init, smc_step, resample)
                        if step.get("mode") in ["init", "smc_step", "resample"]:
                            total_steps += 1
                        # Count resampling events specifically
                        if step.get("mode") == "resample":
                            resampling_events += 1
                
                stats = {
                    "instance": i,
                    "resampling_events": resampling_events,
                    "total_steps": total_steps,
                    "resampling_rate": resampling_events / total_steps if total_steps > 0 else 0.0,
                }
                all_stats.append(stats)
                print(f"  Resampling events: {resampling_events}, Steps: {total_steps}, Rate: {stats['resampling_rate']:.3f}")
                
            except Exception as e:
                print(f"  Warning: Could not parse JSON: {e}")
                
        finally:
            Path(json_path).unlink(missing_ok=True)
    
    # Aggregate statistics
    if all_stats:
        avg_rate = sum(s["resampling_rate"] for s in all_stats) / len(all_stats)
        avg_events = sum(s["resampling_events"] for s in all_stats) / len(all_stats)
        avg_steps = sum(s["total_steps"] for s in all_stats) / len(all_stats)
        
        print(f"\n{'='*60}")
        print(f"Token-Level Summary:")
        print(f"  Average resampling rate: {avg_rate:.3f}")
        print(f"  Average resampling events per instance: {avg_events:.2f}")
        print(f"  Average steps per instance: {avg_steps:.2f}")
        print(f"{'='*60}")
        
        return {
            "model_type": "token",
            "model_name": model_name,
            "avg_resampling_rate": avg_rate,
            "avg_resampling_events": avg_events,
            "avg_steps": avg_steps,
            "per_instance": all_stats,
        }
    
    return None


async def measure_char_resampling(
    model_name: str,
    n_particles: int = 5,
    ess_threshold: float = 0.5,
    max_tokens: int = 100,
    K: int = 10,
    n_instances: int = 10,
):
    """Measure resampling rate for character-level model on Spider."""
    print(f"\n{'='*60}")
    print(f"Character-Level Model: {model_name}")
    print(f"{'='*60}")
    
    # Build byte LLM
    llm = load_model_by_name(model_name, backend="hf")
    if llm.tokenizer.pad_token is None:
        if llm.tokenizer.eos_token:
            llm.tokenizer.pad_token = llm.tokenizer.eos_token
        else:
            llm.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    llm.tokenizer.padding_side = 'left'
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam_params = BeamParams(
        K=K,
        prune_threshold=0.0,
        eos_tokens=[b"\n", b"\n\n", b"<|end_of_text|>", b"<|eot_id|>"],
        heal=True,
    )
    byte_llm = ByteLLM(llm, beam_params)
    
    # Load dataset
    dataset = SpiderDataset.from_spider_dir(
        SPIDER_DATA_DIR,
        grammar_json_path=SPIDER_GRAMMARS,
        few_shot_example_ids=[],
    )
    
    all_stats = []
    
    # Run on first few instances
    for i, instance in enumerate(list(dataset)[:n_instances]):
        print(f"\nInstance {i+1}/{n_instances}")
        
        # Set up prompt
        prompt_ids = default_prompt_formatter(
            byte_llm.llm.tokenizer,
            instance,
            use_chat_format=True,
        )
        prompt_text = byte_llm.llm.tokenizer.decode(prompt_ids)
        byte_llm.set_prompt_from_str(prompt_text)
        
        # Set up constraint
        condition = SpiderTableColumnVerifier(
            grammar=instance.lark_grammar,
            tables=instance.tables,
        ).coerce(byte_llm, f=b"".join)
        
        # Use JSON output to track resampling
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json_path = f.name
        
        try:
            sampler = AWRS(byte_llm, condition)
            sequences = await sampler.smc(
                n_particles=n_particles,
                ess_threshold=ess_threshold,
                max_tokens=max_tokens,
                verbosity=0,
                json_path=json_path,
            )
            
            # Parse JSON to count resampling events
            try:
                with open(json_path) as f:
                    smc_data = json.load(f)
                
                # Count resampling events
                resampling_events = 0
                total_steps = 0
                
                if isinstance(smc_data, list):
                    for step in smc_data:
                        # Count all steps (init, smc_step, resample)
                        if step.get("mode") in ["init", "smc_step", "resample"]:
                            total_steps += 1
                        # Count resampling events specifically
                        if step.get("mode") == "resample":
                            resampling_events += 1
                
                stats = {
                    "instance": i,
                    "resampling_events": resampling_events,
                    "total_steps": total_steps,
                    "resampling_rate": resampling_events / total_steps if total_steps > 0 else 0.0,
                }
                all_stats.append(stats)
                print(f"  Resampling events: {resampling_events}, Steps: {total_steps}, Rate: {stats['resampling_rate']:.3f}")
                
            except Exception as e:
                print(f"  Warning: Could not parse JSON: {e}")
            
            # Cleanup byte LLM
            await byte_llm.cleanup()
                
        finally:
            Path(json_path).unlink(missing_ok=True)
    
    # Aggregate statistics
    if all_stats:
        avg_rate = sum(s["resampling_rate"] for s in all_stats) / len(all_stats)
        avg_events = sum(s["resampling_events"] for s in all_stats) / len(all_stats)
        avg_steps = sum(s["total_steps"] for s in all_stats) / len(all_stats)
        
        print(f"\n{'='*60}")
        print(f"Character-Level Summary:")
        print(f"  Average resampling rate: {avg_rate:.3f}")
        print(f"  Average resampling events per instance: {avg_events:.2f}")
        print(f"  Average steps per instance: {avg_steps:.2f}")
        print(f"{'='*60}")
        
        return {
            "model_type": "character",
            "model_name": model_name,
            "avg_resampling_rate": avg_rate,
            "avg_resampling_events": avg_events,
            "avg_steps": avg_steps,
            "per_instance": all_stats,
        }
    
    return None


async def main():
    parser = argparse.ArgumentParser(description="Measure resampling rate for char vs token models on Spider")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--n_particles", type=int, default=5)
    parser.add_argument("--ess_threshold", type=float, default=0.5)
    parser.add_argument("--max_tokens", type=int, default=100)
    parser.add_argument("--n_instances", type=int, default=10)
    parser.add_argument("--K", type=int, default=10, help="Beam size for ByteLLM")
    parser.add_argument("--token_only", action="store_true", help="Only measure token-level")
    parser.add_argument("--char_only", action="store_true", help="Only measure character-level")
    args = parser.parse_args()
    
    results = {}
    
    if not args.char_only:
        token_result = await measure_token_resampling(
            model_name=args.model,
            n_particles=args.n_particles,
            ess_threshold=args.ess_threshold,
            max_tokens=args.max_tokens,
            n_instances=args.n_instances,
        )
        if token_result:
            results["token"] = token_result
    
    if not args.token_only:
        char_result = await measure_char_resampling(
            model_name=args.model,
            n_particles=args.n_particles,
            ess_threshold=args.ess_threshold,
            max_tokens=args.max_tokens,
            K=args.K,
            n_instances=args.n_instances,
        )
        if char_result:
            results["character"] = char_result
    
    # Final comparison
    if len(results) == 2:
        print(f"\n{'='*60}")
        print("COMPARISON:")
        print(f"{'='*60}")
        print(f"Token-Level Resampling Rate:   {results['token']['avg_resampling_rate']:.3f}")
        print(f"Character-Level Resampling Rate: {results['character']['avg_resampling_rate']:.3f}")
        print(f"Difference: {abs(results['token']['avg_resampling_rate'] - results['character']['avg_resampling_rate']):.3f}")
        print(f"\nToken-Level Avg Steps:   {results['token']['avg_steps']:.2f}")
        print(f"Character-Level Avg Steps: {results['character']['avg_steps']:.2f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())


