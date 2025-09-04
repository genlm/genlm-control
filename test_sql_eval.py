from genlm.eval.domains.spider import SpiderDataset, SpiderEvaluator
spider_data_dir = "/Users/yemara/Desktop/genlm/genlm-eval/assets/spider/spider_sample"  # Replace with your path to the spider dataset
spider_grammars = "/Users/yemara/Desktop/genlm/genlm-eval/assets/spider/grammars.json"  # Replace with your path to the spider grammars

dataset = SpiderDataset.from_spider_dir(
    spider_data_dir, grammar_json_path=spider_grammars, few_shot_example_ids=[0, 1]
)

evaluator = SpiderEvaluator(spider_data_dir)

