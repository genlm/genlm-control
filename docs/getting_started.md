# Getting Started with GenLM Control

This example demonstrates how to use the `genlm-control` library, starting with basic usage and building up to more complex scenarios. It's a good starting point for understanding how to build increasingly complex genlm-control programs, even though the actual example is somewhat contrived.

## Basic LLM Sampling

First, let's look at basic language model sampling using a [`PromptedLLM`][genlm.control.PromptedLLM]:

```python
from genlm.control import PromptedLLM, direct_token_sampler

# Load gpt2 (or any other Hugging Face model)
mtl_llm = PromptedLLM.from_name("gpt2", temperature=0.5, eos_tokens=[b'.'])

# Set the fixed prompt prefix for the language model
# All language model predictions will be conditioned on this prompt
mtl_llm.set_prompt_from_str("Montreal is")

# Load a sampler that proposes tokens by sampling directly from the LM's distribution
token_sampler = direct_token_sampler(mtl_llm)

# Run SMC with 5 particles, a maximum of 25 tokens.
# With no additional constraints, weights are uniform, so we set `ess_threshold` to 0.0
sequences = await token_sampler.smc(n_particles=5, max_tokens=25, ess_threshold=0.0)

# Show the posterior over token sequences
sequences.posterior

# Show the posterior over complete UTF-8 decodable sequences
sequences.decoded_posterior
```

Note: Sequences are lists of `bytes` objects because each token in the language model's vocabulary is represented as a bytes object.

## Prompt Intersection

Next, we'll look at combining prompts from multiple language models using a [`Product`][genlm.control.potential.Product] potential:

```python
# Spawn a new language model (shallow copy, sharing the same underlying model)
bos_llm = mtl_llm.spawn()
bos_llm.set_prompt_from_str("Boston is")

# Take the product of the two language models
# This defines a `Product` potential which is the element-wise product of the two LMs
product = mtl_llm * bos_llm

# Create a sampler that proposes tokens by sampling directly from the product
token_sampler = direct_token_sampler(product)

sequences = await token_sampler.smc(n_particles=5, max_tokens=25, ess_threshold=0.5)

sequences.posterior

sequences.decoded_posterior
```

## Adding Regex Constraints

We can add regex constraints to our `product` using a [`BoolFSA`][genlm.control.potential.built_in.wfsa.BoolFSA] and the [`AWRS`][genlm.control.sampler.token.AWRS] token sampler:

```python
from genlm.control import BoolFSA, AWRS

# Create a regex constraint that matches sequences containing the word "the"
# followed by either "best" or "worst" and then anything else
best_fsa = BoolFSA.from_regex(r"\sthe\s(best|worst).*")

# BoolFSA's are defined over individual bytes by default
# Their `prefix` and `complete` methods are called on byte sequences
print("best_fsa.prefix(b'the bes') =", await best_fsa.prefix(b"the bes"))
print(
    "best_fsa.complete(b'the best city') =",
    await best_fsa.complete(b"the best city"),
)

# Coerce the FSA to work with the LLM's vocabulary
coerced_fsa = best_fsa.coerce(product, f=b"".join)

# Use the AWRS token sampler; it will only call the fsa on a subset of the product vocabulary
token_sampler = AWRS(product, coerced_fsa)

sequences = await token_sampler.smc(n_particles=5, max_tokens=25, ess_threshold=0.5)

sequences.posterior

sequences.decoded_posterior
```

## Custom Sentiment Analysis Potential

Now we'll create a custom potential by subclassing [`Potential`][genlm.control.potential.base.Potential] and use it as a **critic** to further guide generation:

```python
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from genlm.control import Potential

# Create our own custom potential for sentiment analysis.
# Custom potentials must subclass `Potential` and implement the `prefix` and `complete` methods.
# They can also override other methods, like `batch_prefix`, and `batch_complete` for improved performance.
# Each Potential needs to specify its vocabulary of tokens; this potential has a vocabulary of individual bytes.
class SentimentAnalysis(Potential):
    def __init__(self, model, tokenizer, sentiment="POSITIVE"):
        self.model = model
        self.tokenizer = tokenizer

        self.sentiment_idx = model.config.label2id.get(sentiment, None)
        if self.sentiment_idx is None:
            raise ValueError(f"Sentiment {sentiment} not found in model labels")

        super().__init__(vocabulary=list(range(256)))  # Defined over bytes

    def _forward(self, contexts):
        strings = [bytes(context).decode("utf-8", errors="ignore") for context in contexts]
        inputs = self.tokenizer(strings, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        return logits.log_softmax(dim=-1)[:, self.sentiment_idx].cpu().numpy()

    async def prefix(self, context):
        return self._forward([context])[0].item()

    async def complete(self, context):
        return self._forward([context])[0].item()

    async def batch_complete(self, contexts):
        return self._forward(contexts)

    async def batch_prefix(self, contexts):
        return self._forward(contexts)

# Initialize sentiment analysis potential
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
sentiment_analysis = SentimentAnalysis(
    model=DistilBertForSequenceClassification.from_pretrained(model_name),
    tokenizer=DistilBertTokenizer.from_pretrained(model_name),
    sentiment="POSITIVE",
)

# Test the potential
print("\nSentiment analysis test:")
print(
    "sentiment_analysis.prefix(b'so good') =",
    await sentiment_analysis.prefix(b"so good"),
)
print(
    "sentiment_analysis.prefix(b'so bad') =",
    await sentiment_analysis.prefix(b"so bad"),
)

# Verify the potential satisfies required properties
await sentiment_analysis.assert_logw_next_consistency(b"the best", top=5)
await sentiment_analysis.assert_autoreg_fact(b"the best")

# Set up efficient sampling with the sentiment analysis potential
token_sampler = AWRS(product, coerced_fsa)
critic = sentiment_analysis.coerce(token_sampler.target, f=b"".join)

# Run SMC using the sentiment analysis potential as a critic
sequences = await token_sampler.smc(
    n_particles=5,
    max_tokens=25,
    ess_threshold=0.5,
    critic=critic, # Pass the critic to the SMC sampler; this will reweight samples at each step based on their positivity
)

# Show the posterior over complete UTF-8 decodable sequences
sequences.decoded_posterior
```

## Optimizing with Autobatching

Finally, we can optimize performance using autobatching. During generation, all requests to the sentiment analysis potential are made to the instance methods (`prefix`, `complete`). We can take advantage of the fact that we have parallelized batch versions of these methods using the [`to_autobatched`][genlm.control.potential.operators.PotentialOps.to_autobatched] method.

```python
from arsenal.timer import timeit

# Create an autobatched version of the critic
# This creates a new potential that automatically batches concurrent
# requests to the instance methods (`prefix`, `complete`, `logw_next`)
# and processes them using the batch methods (`batch_complete`, `batch_prefix`, `batch_logw_next`).
autobatched_critic = critic.to_autobatched()

# Run SMC with timing for comparison
with timeit("Timing sentiment-guided sampling with autobatching"):
    sequences = await token_sampler.smc(
        n_particles=10,
        max_tokens=25,
        ess_threshold=0.5,
        critic=autobatched_critic, # Pass the autobatched critic to the SMC sampler
    )

sequences.decoded_posterior

# The autobatched version should be significantly faster than this version
with timeit("Timing sentiment-guided sampling without autobatching"):
    sequences = await token_sampler.smc(
        n_particles=10,
        max_tokens=25,
        ess_threshold=0.5,
        critic=critic,
    )

sequences.decoded_posterior
```
