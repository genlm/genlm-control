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


# Paths to the sample Spider assets shipped in genlm-eval
SPIDER_DATA_DIR = \
    "/Users/yemara/Desktop/genlm/genlm-eval/assets/spider/spider_sample"
SPIDER_GRAMMARS = \
    "/Users/yemara/Desktop/genlm/genlm-eval/assets/spider/grammars.json"


def build_bytelm():
    llm = load_model_by_name("gpt2", backend="hf")
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

    # Prepare prompt for this instance (chat template → prompt_ids → str)
    # GPT-2 has no chat template; use a plain prompt format
    prompt_ids = default_prompt_formatter(
        BYTE_LLM.llm.tokenizer,
        instance,
        use_chat_format=False,
    )
    BYTE_LLM.set_prompt_from_str(BYTE_LLM.llm.tokenizer.decode(prompt_ids))

    # Optional constraint: table/column verifier if grammar available
    if instance.lark_grammar:
        condition = SpiderTableColumnVerifier(
            grammar=instance.lark_grammar, tables=instance.tables
        ).coerce(BYTE_LLM, f=b"".join)
        sampler = AWRS(BYTE_LLM, condition)
    else:
        # Fallback: no grammar available -> unconstrained LM sampling
        sampler = direct_token_sampler(BYTE_LLM)

    sequences = await sampler.smc(
        n_particles=1,
        ess_threshold=0.5,
        max_tokens=400,
        verbosity=1,
    )

    print("Decoded posterior:", sequences.decoded_posterior)

    responses: List[ModelResponse] = [
        ModelResponse(response=seq, weight=float(prob))
        for seq, prob in sequences.decoded_posterior.items()
    ]

    # Cleanup between instances to silence background task warnings
    await BYTE_LLM.cleanup()

    return ModelOutput(responses=responses, runtime_seconds=None)


async def main():
    global BYTE_LLM
    BYTE_LLM = build_bytelm()

    dataset = SpiderDataset.from_spider_dir(
        SPIDER_DATA_DIR, grammar_json_path=SPIDER_GRAMMARS, few_shot_example_ids=[0, 1]
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
        max_instances=2,
        verbosity=1,
    )

    print("\nSummary:")
    print(f"n_instances: {results['n_instances']}")
    print(f"average_weighted_accuracy: {results['average_weighted_accuracy']}")

    await BYTE_LLM.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
