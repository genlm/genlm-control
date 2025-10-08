import asyncio
from typing import List

from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams
from genlm.control import ByteLLM, AWRS, direct_token_sampler
from genlm.eval.core import run_evaluation
from genlm.eval.core.model import ModelOutput, ModelResponse
from genlm.eval.domains.spider.spider import (
    SpiderDataset,
    SpiderEvaluator,
    default_prompt_formatter,
)
from genlm.eval.domains.spider.table_column_potential import SpiderTableColumnVerifier
from pprint import pprint as pp

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "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"
# "/teamspace/studios/this_studio/spider_data"
SPIDER_DATA_DIR = "/teamspace/studios/this_studio/genlm-eval/assets/spider/spider_sample"
# SPIDER_DATA_DIR = "/teamspace/studios/this_studio/spider_data"
SPIDER_GRAMMARS = "/teamspace/studios/this_studio/genlm-eval/assets/spider/grammars.json"


def build_bytelm():
    llm = load_model_by_name("meta-llama/Llama-3.2-1B-Instruct", backend="hf")
    # llm = load_model_by_name("gpt2", backend="hf")
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam_params = BeamParams(
        K=1,
        prune_threshold=0.0,
        eos_tokens={b"\n", b"\n\n"},
        heal=True,
    )
    return ByteLLM(llm, beam_params)


async def spider_model_adaptor(instance, output_dir: str, replicate: int) -> ModelOutput:
    global BYTE_LLM

    prompt_ids = default_prompt_formatter(
        BYTE_LLM.llm.tokenizer,
        instance,
        use_chat_format=False,
    )
    prompt_text = BYTE_LLM.llm.tokenizer.decode(prompt_ids)
    BYTE_LLM.set_prompt_from_str(prompt_text)

    # Print few-shot examples being used
    print("\n--- Few-shot examples ---")
    if instance.few_shot_examples:
        for i, (inp, out) in enumerate(instance.few_shot_examples):
            preview = inp.replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."
            print(f"[{i}] Input: {preview}")
            print(f"    Output: {out}")
    else:
        print("(none)")

    # Print prompt and gold SQL for inspection
    print("\n--- Prompt ---\n" + prompt_text)
    print("\n--- Gold SQL ---\n" + instance.gold)
    condition = SpiderTableColumnVerifier(grammar=instance.lark_grammar, tables=instance.tables).coerce(BYTE_LLM, f=b"".join)
    sampler = AWRS(BYTE_LLM, condition)

    sequences = await sampler.smc(
        n_particles=2,
        ess_threshold=0.9,
        max_tokens=70,
        verbosity=1,
    )

    print("Decoded posterior:", sequences.decoded_posterior)

    responses: List[ModelResponse] = [
        ModelResponse(response=seq, weight=float(prob))
        for seq, prob in sequences.decoded_posterior.items()
    ]

    await BYTE_LLM.cleanup()

    return ModelOutput(responses=responses, runtime_seconds=None)


async def main():
    global BYTE_LLM
    BYTE_LLM = build_bytelm()

    dataset = SpiderDataset.from_spider_dir(
        SPIDER_DATA_DIR,
        grammar_json_path=SPIDER_GRAMMARS,
        few_shot_example_ids=[0,1],
    )

    evaluator = SpiderEvaluator(SPIDER_DATA_DIR)
    results = await run_evaluation(
        dataset=dataset,
        model=spider_model_adaptor,
        evaluator=evaluator,
        output_dir=None,
        n_replicates=1,
        overwrite_results=False,
        overwrite_outputs=False,
        max_instances=1,
        verbosity=1,
    )

    print("\nSummary:")
    print(f"n_instances: {results['n_instances']}")
    print(f"average_weighted_accuracy: {results['average_weighted_accuracy']}")

    await BYTE_LLM.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
