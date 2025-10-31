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
# HF_TOKEN should be set via environment variable or huggingface-cli login
# "/teamspace/studios/this_studio/spider_data"
SPIDER_DATA_DIR = "/teamspace/studios/this_studio/genlm-eval/assets/spider/spider_sample"
SPIDER_GRAMMARS = "/teamspace/studios/this_studio/genlm-eval/assets/spider/grammars.json"


def build_bytelm(model_name: str = "meta-llama/Llama-3.2-1B-Instruct"):
    llm = load_model_by_name(model_name, backend="hf")
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam_params = BeamParams(
        K=1,
        prune_threshold=0.0,
        eos_tokens={b"\n", b"\n\n"},
        heal=True,
    )
    return ByteLLM(llm, beam_params)


async def smc_generate_sql(instance, n_particles: int = 5, max_tokens: int = 200, use_chat_format: bool = False):
    """Generate SQL for one Spider instance using ByteLLM + grammar constraint via SMC.

    Returns list of (response, weight) sorted by weight desc.
    """
    bytelm = build_bytelm()

    prompt_ids = default_prompt_formatter(
        bytelm.llm.tokenizer,
        instance,
        use_chat_format=use_chat_format,
    )
    bytelm.set_prompt_from_str(bytelm.llm.tokenizer.decode(prompt_ids))
    # import ipdb; ipdb.set_trace()
    condition = SpiderTableColumnVerifier(grammar=instance.lark_grammar, tables=instance.tables).coerce(bytelm, f=b"".join)

    sampler = AWRS(bytelm, condition)
    sequences = await sampler.smc(
        n_particles=n_particles,
        ess_threshold=0.5,
        max_tokens=max_tokens,
        verbosity=1,
    )
    import ipdb; ipdb.set_trace()
    print(sequences.decoded_posterior)
    # results = [(seq, float(w)) for seq, w in sequences.decoded_posterior.items()]
    # results.sort(key=lambda x: -x[1])

    await bytelm.cleanup()
    
    # return results


def main():
    # Load dataset and pick the first dev example
    dataset = SpiderDataset.from_spider_dir(
        SPIDER_DATA_DIR, grammar_json_path=SPIDER_GRAMMARS, few_shot_example_ids=[0,1]
        )
    inst = next(iter(dataset))

    # Example predicted SQL (replace with your model's output)
    # For a guaranteed correct example, you can start with the gold query:
    # pred_sql = inst.gold
    # pred_sql = "SELECT count(*) FROM singer;"

    # evaluator = SpiderEvaluator(SPIDER_DATA_DIR)
    # result = evaluator.evaluate_sample(inst, pred_sql)

    # print("Instance:", inst)
    # print("Pred:", pred_sql)
    # print("Gold:", inst.gold)
    # print("Score:", result.score, "Reason:", result.desc, "Meta:", result.metadata)

    print("\nSMC generation (grammar-constrained) candidates:")
    import asyncio as _asyncio
    _asyncio.run(smc_generate_sql(inst, n_particles=5, max_tokens=50, use_chat_format=True))


if __name__ == "__main__":
    main()
