import argparse
import asyncio
from typing import List

from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams
from genlm.control import ByteLLM, AWRS
from genlm.eval.core import run_evaluation
from genlm.eval.core.model import ModelOutput, ModelResponse
from genlm.eval.domains.spider.spider import (
    SpiderDataset,
    SpiderEvaluator,
    default_prompt_formatter,
)
from genlm.eval.domains.spider.table_column_potential import SpiderTableColumnVerifier


# Default paths; change SPIDER_DATA_DIR to full dataset if desired
SPIDER_DATA_DIR = "/teamspace/studios/this_studio/spider_data"
    # "/teamspace/studios/this_studio/genlm-eval/assets/spider/spider_sample"
SPIDER_GRAMMARS = "/teamspace/studios/this_studio/genlm-eval/assets/spider/grammars.json"




def build_byte_llm(model_name: str = "meta-llama/Llama-3.2-1B-Instruct") -> ByteLLM:
    # Use ByteLLM with byte-level EOS; AWRS will enforce constraints
    llm = load_model_by_name(model_name, backend="hf")
    
    # Set pad_token to avoid batching issues
    if llm.tokenizer.pad_token is None:
        if llm.tokenizer.eos_token:
            llm.tokenizer.pad_token = llm.tokenizer.eos_token
        else:
            llm.tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    llm.tokenizer.padding_side = 'left'
    
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam_params = BeamParams(
        K=10,
        prune_threshold=0.0,
        # eos_tokens={b"\n"},
        eos_tokens=[b"\n", b"\n\n", b"<|end_of_text|>", b"<|eot_id|>"],
        heal=True,
    )
    print("eos tokens", beam_params.eos_tokens)
    return ByteLLM(llm, beam_params)


async def spider_model_adaptor(instance, output_dir: str, replicate: int) -> ModelOutput:
    global BYTE_LLM

    # Format prompt and set on ByteLLM
    prompt_ids = default_prompt_formatter(
        BYTE_LLM.llm.tokenizer,
        instance,
        use_chat_format=True,
    )
    prompt_text = BYTE_LLM.llm.tokenizer.decode(prompt_ids)
    BYTE_LLM.set_prompt_from_str(prompt_text)

    # Grammar + schema constraint coerced to byte LM
    condition = SpiderTableColumnVerifier(
        grammar=instance.lark_grammar,
        tables=instance.tables,
    ).coerce(BYTE_LLM, f=b"".join)

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
    
    await BYTE_LLM.cleanup()
    
    return ModelOutput(responses=responses, runtime_seconds=None)


async def main():
    parser = argparse.ArgumentParser(description="Spider eval with byte-level model (ByteLLM + AWRS)")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)  # exclusive
    parser.add_argument("--skip_idxs", type=str, default="", help="Comma/space-separated indices to skip")
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument("--use_sample", action="store_true", help="Use spider_sample set")
    args = parser.parse_args()

    # Build byte LM
    global BYTE_LLM
    BYTE_LLM = build_byte_llm(args.model)
    
    # Dataset
    dataset = SpiderDataset.from_spider_dir(
        SPIDER_DATA_DIR if args.use_sample else "/teamspace/studios/this_studio/spider_data",
        grammar_json_path=SPIDER_GRAMMARS,
        few_shot_example_ids=[0,1],
    )

    # Parse skip indices
    skip_set = set()
    if args.skip_idxs:
        for tok in args.skip_idxs.replace(",", " ").split():
            try:
                skip_set.add(int(tok))
            except ValueError:
                pass

    # Slice/filter by original indices
    selected = []
    for i, inst in enumerate(dataset):
        if i in skip_set:
            continue
        if i < args.start_idx:
            continue
        if args.end_idx is not None and i >= args.end_idx:
            break
        selected.append(inst)
    dataset = selected

    evaluator = SpiderEvaluator(
        SPIDER_DATA_DIR if args.use_sample else "/teamspace/studios/this_studio/spider_data"
    )
    results = await run_evaluation(
        dataset=dataset,
        model=spider_model_adaptor,
        evaluator=evaluator,
        output_dir=None,
        n_replicates=1,
        overwrite_results=False,
        overwrite_outputs=False,
        # max_instances=args.max_instances,
        verbosity=1,
    )

    print("\nSummary:")
    print(f"n_instances: {results['n_instances']}")
    print(f"average_weighted_accuracy: {results['average_weighted_accuracy']}")
    
    await BYTE_LLM.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
