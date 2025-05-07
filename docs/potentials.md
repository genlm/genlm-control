# Potentials

[Potentials][genlm.control.potential] are the core object in `genlm-control`. A potential encodes constraints or preferences by assigning non-negative weights to sequences of tokens.

Potentials guide text generation by:

* Acting as components of [**samplers**](samplers.md), which serve to propose new tokens at each step of the generation process.
* Serving as **critics**, which serve to reweight sequences based on whether they satisfy the constraint encoded by the potential at each step of the generation process.

## Key concepts

### Vocabulary

Each potential has a **vocabulary** which defines the set of tokens it operates on. Most built-in potentials operate on vocabularies whose tokens are `bytes` or `int` objects (the latter often representing individual bytes).

### Weight assignment

Potentials assign weights to sequences of tokens from their vocabulary. These weights are always non-negative real numbers, though they are computed in log space for numerical stability.

A potential defines two core weighting functions:

1. `complete` - Assigns weights to sequences that are considered "finished" or "complete". For example, a potential enforcing grammatical correctness would assign positive weights to grammatically valid sentences and zero weights (negative infinity in log space) to invalid ones.

2. `prefix` - Assigns weights to partial sequences that could potentially be extended into valid complete sequences. For example, a potential enforcing grammatical correctness would assign positive weights to prefixes of grammatically valid sequences.

    Given a complete method, there are many possible prefix methods that could be used, providing as much or as little information as desired. The key requirement is that if a prefix has zero weight, then all of its extensions and completions must also have zero weight - in other words, prefix cannot rule out sequences that could later become valid.

The relationship between complete and prefix weights is formalized in the [Formalization](#formalization) section.

### Next-token weights

Potentials also implement a `logw_next` method, which computes weights for each possible next token in the potential's vocabulary (and a reserved end-of-sequence token) given a context sequence. These weights are crucial for controlled text generation as they can be used to guide the selection of the next token at each step.

The `logw_next` method is implemented by default in terms of the `complete` and `prefix` methods. Potentials will often override this method to provide a more efficient implementation. However, `logw_next` must satisfy a contract with `complete`/`prefix`, specified in [Formalization](#formalization).

### Batch methods

For improved performance with large batches of inputs, potentials support batch operations:

* `batch_complete(contexts)`
* `batch_prefix(contexts)`
* `batch_logw_next(contexts)`
* `batch_score(contexts)`

By default, these methods simply call the corresponding non-batch method for all inputs, but potentials can override them to provide more efficient implementations. They can be used in conjunction with [auto batching](performance.md#auto-batching) for improved performance during generation.

## Built-in potentials

`genlm-control` comes with a number of built-in potentials that can be used in controlled text generation.

### Language models

[`PromptedLLM`][genlm.control.potential.built_in.llm.PromptedLLM] represents a language model conditioned on a fixed prompt prefix.

```python
# Load GPT-2 with temperature 0.5
llm = PromptedLLM.from_name("gpt2", temperature=0.5)

# Set a prompt prefix that all generations will be conditioned on
llm.set_prompt_from_str("Montreal is")
```

`PromptedLLM`s have a vocabulary of `bytes` tokens, obtained from the language model's tokenizer.

### Finite-state automata

`genlm-control` provides two [FSA implementations][genlm.control.potential.built_in.wfsa]:

1. `WFSA` (Weighted Finite-State Automata) - For weighted constraints:
```python
# Create a WFSA from a regex pattern
# Transitions are automatically normalized to form probability distributions
wfsa = WFSA.from_regex(r"\sthe\s(best|worst).*😎")
```

2. `BoolFSA` (Boolean Finite-State Automata) - For hard constraints:
```python
# Create a boolean FSA from a regex pattern
# Transitions are binary (0 or -inf in log space)
fsa = BoolFSA.from_regex(r"\sthe\s(best|worst).*😎")
```

Both FSAs:

* Support regex patterns with standard syntax
* Operate on byte-level sequences by default
* Can be combined with other potentials via products

### Context-free grammars

Similar to FSAs, `genlm-control` provides two [CFG implementations][genlm.control.potential.built_in.wcfg]:

1. `WCFG` (Weighted Context-Free Grammar).
```python
cfg = WCFG.from_string("""
    1.0: S -> NP VP
    0.5: NP -> the N
    0.5: NP -> a N
    1.0: VP -> V NP
    0.5: N -> cat
    0.5: N -> dog
    0.5: V -> saw
    0.5: V -> chased
""")
```

2. `BoolCFG` (Boolean Context-Free Grammar).
```python
# Create a boolean CFG from a Lark grammar string
cfg = BoolCFG.from_lark("""
    start: np vp
    np: ("the" | "a") WS n
    vp: WS v WS np
    n: "cat" | "dog"
    v: "saw" | "chased"
    %import common.WS
""")
```

`BoolCFG`s support grammar specification via [Lark syntax](https://lark-parser.readthedocs.io/en/latest/grammar.html).

Both CFGs:

* Use Earley parsing for efficient recognition
* Can be combined with other potentials
* Operate on byte-level sequences by default

> **Note:** It is recommended to specify grammars via lark syntax. The `from_string` method is provided for convenience, but it is not as flexible and robust.

## Custom potentials

You can create custom potentials to implement specialized constraints or preferences that aren't covered by the built-in options.

### Creating a custom potential

To define a custom potential:

1. Create a subclass of `Potential`
2. Implement the `complete` and `prefix` methods
3. Optionally override `logw_next` and the batch methods for performance optimization

When implementing custom potentials, the key is understanding the relationship between `complete` and `prefix`. Consider the following example of a potential that only allows sequences of a given length:

```python
class LengthPotential(Potential):
    """ A potential that only allows sequences of a given length. """
    def __init__(self, vocabulary, length):
        # Initialize the superclass with the potential's vocabulary.
        super().__init__(vocabulary)
        self.length = length

    async def complete(self, context):
        # Note: 0.0 = log(1.0) and float('-inf') = log(0.0)
        return 0.0 if len(context) == self.length else float('-inf')

    async def prefix(self, context):
        # Note: 0.0 = log(1.0) and float('-inf') = log(0.0)
        return 0.0 if len(context) <= self.length else float('-inf')

length_potential = LengthPotential(vocabulary=[b'the', b'a', b'cat', b'dog', b'saw', b'chased'], length=5)
```

This example illustrates the key difference between `complete` and `prefix`: the `complete` method only allows sequences of exactly the target length, while the `prefix` method allows any sequence that could potentially reach the target length (i.e., any sequence not exceeding the target length).

### Common pitfalls

When implementing custom potentials, be aware of these common issues:

1. **Inconsistent complete/prefix relationship** - If your `prefix` method assigns zero weight to a sequence, all extensions must also have zero weight.

2. **Inefficient implementations** - For complex potentials, consider overriding `logw_next` with a more efficient implementation than the default.

3. **Not handling async properly** - All potential methods are asynchronous. Make sure to use `await` when calling them and define your methods with `async def`.

### Testing your custom potential

Potentials automatically inherit from the [`PotentialTests`][genlm.control.potential.testing] mixin, which provides a number of tests for validating the correctness of the potential's implementation.

```python
# These will raise an exception if the potential implementation does not satisfy the properties
await potential.assert_logw_next_consistency(context)
await potential.assert_autoreg_fact(context)
await potential.assert_batch_consistency(contexts)
```

## Complex usage

### Products of potentials

The [`Product`][genlm.control.potential.product] class allows you to combine two potentials. A `Product` is itself is a potential, meaning that it implements all potential methods and that it is possible to chain products to combine more than two potentials.

```python
# Example: Prompt intersection
mtl_llm = PromptedLLM.from_name("gpt2")
mtl_llm.set_prompt_from_str("Montreal is")

bos_llm = mtl_llm.spawn()
bos_llm.set_prompt_from_str("Boston is")

# Create product using multiplication operator
product = mtl_llm * bos_llm
```

The product potential operates on the intersection of the two potentials' vocabularies. For a product potential:

- The vocabulary $\A$ is the intersection of the two potentials' vocabularies: $\A = \A_1 \cap \A_2$.
- The prefix potential $\prefix$ is the product (sum in log space) of the individual prefix potentials: $\log \prefix(\xx) = \log \prefix_1(\xx) + \log \prefix_2(\xx)$.
- The complete potential $\complete$ is the product (sum in log space) of the individual complete potentials: $\log \complete(\xx) = \log \complete_1(\xx) + \log \complete_2(\xx)$.
- The next-token potential $\pot(\cdot \mid \xx)$ is the product (sum in log space) of the individual next-token potentials: $\log \pot(x \mid \xx) = \log \pot_1(x \mid \xx) + \log \pot_2(x \mid \xx)$ for $x \in (\A_1 \cap \A_2) \cup \{\eos\}$

> **Warning:** Be careful when taking products of potentials with minimal vocabulary overlap, as the resulting potential will only operate on tokens present in both vocabularies. A warning will be raised if the vocabulary overlap is less than 10% of either potential's vocabulary.


### Coerced potentials

The [`Coerced`][genlm.control.potential.coerce] class allows you to adapt a potential to work with a different vocabulary using a coercion function. The coercion function must map between sequences in the new vocabulary and sequences in the potential's original vocabulary. This is particularly useful when combining potentials that operate on different types of tokens.

```python
# Example: Coercing a byte-level FSA to work with a language model's tokens
fsa = BoolFSA.from_regex(r"\sthe\s(best|worst).*")  # Works on bytes
llm = PromptedLLM.from_name("gpt2")  # Works on byte sequences

# Coerce the FSA to work with the LLM's tokens by joining tokens into bytes
coerced_fsa = fsa.coerce(llm, f=b''.join)

# Now we can combine them using the product operator!
product = llm * coerced_fsa
```

Common use cases for coercion include:

- Adapting byte-level constraints (like FSAs) to work with token-level language models (which have vocabularies of byte *sequences*)
- Implementing constraints that operate on processed versions of the tokens (e.g., lowercase text)
- Converting between different tokenization schemes

> **Performance Note:** The coercion operation can impact performance, especially when mapping from a coarser token type to a finer token type (e.g., byte sequences to individual bytes). To sample tokens from a coerced product, consider using specialized samplers (e.g., `eager_token_sampler`, `topk_token_sampler`).

### Performance optimizations

`genlm-control` provides a number of performance optimizations for potentials, described in the [performance](performance.md) section.


## Formalization

This section provides a formal definition of potentials and the relationships between their complete, prefix, and next-token potentials.

**Notation** Let $\A$ be a vocabulary of tokens and $\eos$ a specialized end-of-sequence token. Let $\A^*$ denote the set of all sequences of tokens which can be built from $\A$ (including the empty sequence $\epsilon$) and $\A^*{\eos} = \{\xx\eos : \xx \in \A^*\}$ the set of $\eos$-terminated sequences. We refer to $\A^*$ as the set of *prefix* sequences and $\A^*{\eos}$ the set of *complete* sequences.

A potential $\pot$ is a function $\pot: \A^* \cup\A^*{\eos} \rightarrow \mathbb{R}_{\geq 0}$ which assigns a non-negative real number to prefix and complete sequences from its vocabulary $\A$:

$$
\pot(\xx) = \begin{cases}
    \prefix(\xx) & \text{if } \xx \in \A^* \\
    \complete(\yy) & \text{if } \xx = \yy\eos, \yy \in \A^*
\end{cases}
$$

where

* $\prefix : \A^* \rightarrow \mathbb{R}_{\geq 0}$ is the **prefix potential**
* $\complete : \A^* \rightarrow \mathbb{R}_{\geq 0}$ is the **complete potential**

The complete and prefix potentials are related by the following equality:

$$
\prefix(\xx) = 0 \implies \complete(\xx\yy) = 0 \, \forall \xx,\yy \text{ such that } \xx\yy \in \A^*
$$

Intuitively, this means that the prefix potential cannot rule out a sequence which can later on turn out to be valid according to the complete potential.

Finally, we define the **next-token weights function** $\pot(x \mid \xx) : \A \cup \{\eos\} \rightarrow \mathbb{R}_{\geq 0}$, which assigns a non-negative real number to each token $x \in \A \cup \{\eos\}$ given a sequence $\xx \in \A^*$:

$$
\pot(x \mid \xx) = \frac{\pot(\xx x)}{\prefix(\xx)} = \begin{cases}
    \frac{\prefix(\xx x)}{\prefix(\xx)} & \text{if } x \in \A \\
    \frac{\complete(\xx)}{\prefix(\xx)} & \text{if } x = \eos
\end{cases}
$$

$\pot(\cdot \mid \xx)$ is related to the complete and prefix potentials according to the following autoregressive factorization:

$$
\frac{\complete(\xx)}{\prefix(\epsilon)} = \pot(\eos \mid \xx) \prod_{x \in \xx} \pot(x \mid \xx)
$$

### Correspondance with the `Potential` class

Each of the quantities above directly corresponds to a method or attribute of the `Potential` class:

| Method/Attribute | Mathematical Quantity | Description |
|-----------------|----------------------|-------------|
| `vocab` | $\A$ | The vocabulary of the potential. |
| `eos` | $\eos$ | The end-of-sequence token. |
| `vocab_eos` | $\A \cup \{\eos\}$ | The vocabulary of the potential including the end-of-sequence token. |
| `complete(self, context)` | $\log \complete(\xx)$ | The complete potential for a given sequence. |
| `prefix(self, context)` | $\log \prefix(\xx)$ | The prefix potential for a given sequence. |
| `logw_next(self, context)` | $\log \pot(\cdot \mid \xx)$ | The next-token potential for a given prefix sequence. |
| `score(self, context)` | $\log \pot(\xx)$ | The potential, dispatching to `complete` for eos-terminated sequences and `prefix` otherwise. |
