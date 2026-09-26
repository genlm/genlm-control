# Performance Optimizations

The main levers for the speed of a `genlm-control` program are:

- **Backend choice**: Which inference engine serves the language model
- **Auto-batching**: Concurrent requests to a potential's instance methods execute as one batch call (on by default)
- **Concurrent SMC runs**: Independent inference problems share the same batches
- **Multiprocessing**: Runs multiple instances of a `Potential` in parallel across CPU cores


## Choosing a backend

`PromptedLLM.from_name` takes `backend="vllm"`, `"hf"`, or `"mlx"`. The default is `vllm` when CUDA is available and `hf` otherwise.

- **vllm** (GPU): The fastest. The engine keeps one request per live context, so extending a context by one token appends to its request instead of re-prefilling it, and concurrent requests run as one forward.
- **mlx** (Apple silicon): Keeps the KV cache of the previous batch and extends it when the next batch continues it.
- **hf** (transformers): Runs anywhere; the slowest.

```python
llm = PromptedLLM.from_name("meta-llama/Llama-3.2-1B", backend="vllm")
```

## Auto-batching

Concurrent calls to a potential's instance methods (`complete`, `prefix`, `logw_next`, `score`) can execute as one call to the corresponding batch method (`batch_complete`, `batch_prefix`, `batch_logw_next`, `batch_score`). During SMC every particle makes these calls at every step, so batching turns N calls into one: for a language model, one forward over N contexts; for any other potential, whatever its batch methods save over N single calls.

This is on by default. `DirectTokenSampler` and `AWRS` wrap the potentials they are given, `EagerSetSampler` and `TopKSetSampler` wrap their iterable potential, and `SMC` wraps its critic, each in an [`AutoBatchedPotential`][genlm.control.potential.autobatch]. The wrapper collects the calls made during one pass of the event loop and dispatches them together; nothing runs in the background. Each of these takes `autobatch=False` to opt out.

A potential used outside a sampler or `SMC` can be wrapped by hand with `to_autobatched()`:

```python
autobatched_potential = potential.to_autobatched()
results = await asyncio.gather(
    *(autobatched_potential.complete(seq) for seq in sequences)  # one batch_complete call
)
```

Wrapping is memoized on the potential, so wrapping an already-wrapped potential returns the same wrapper.

Auto-batching only helps when the batch methods are faster than the single calls they replace. The defaults on `Potential` call the single method once per input, so a custom potential should implement the batch methods; the sentiment critic in [Getting Started](getting_started.md#autobatching) is an example.

## Concurrent SMC runs

One `SMC` call is one inference problem. Problems run concurrently share their batches: their particles' calls meet in the same auto-batched dispatch and the same backend forward, so running several under `asyncio.gather` costs close to running one. This holds when the runs share a language model instance; spawned copies with different prompts do.

```python
sequences_a, sequences_b = await asyncio.gather(
    sampler_a.smc(n_particles=10, ess_threshold=0.5, max_tokens=30),
    sampler_b.smc(n_particles=10, ess_threshold=0.5, max_tokens=30),
)
```

## Multiprocessing

CPU parallelization can significantly improve performance for compute-intensive `Potential` classes. This is particularly useful when methods like `complete`, `prefix`, or `logw_next` involve heavy computation.

### Usage

To enable multiprocessing, use the `to_multiprocess()` method:

```python
# Create a multiprocess wrapper with desired number of workers
mp_potential = potential.to_multiprocess(num_workers=2)
# Use it like a regular potential - requests are distributed across workers
results = await asyncio.gather(
    *(mp_potential.complete(seq) for seq in sequences) # These will be distributed across workers
)
```

This creates a new potential that is a wrapper ([`MultiProcPotential`][genlm.control.potential.multi_proc]) around the original potential. The wrapper asynchronously distributes requests across multiple processes (in a non-blocking manner). This allows you to scale your computations across multiple cores without changing your code structure.

### Requirements

For multiprocessing to work, the potential must implement a picklable `spawn()` method that creates a new instance of the potential. Only some built-in `Potential` classes support this by default. Custom potentials need to implement their own `spawn()` method.

### Performance Benefits

Multiprocessing improves performance for both batched methods (`batch_complete`, `batch_prefix`, `batch_logw_next`) and unbatched methods (`complete`, `prefix`, `logw_next`).

In the batched case, requests within a batch are processed in parallel across workers. For individual method calls, requests are distributed to available worker processes and are executed asynchronously.

## When to use each optimization

> **Note:** Language model requests are also batched at the backend: concurrent `next_token_logprobs` requests to one model run as one forward whether or not they arrive through an `AutoBatchedPotential`.

- Use auto-batching when the potential's batch operations are more efficient than sequential operations
- Use multiprocessing when the potential's operations are compute-intensive and can benefit from parallel processing
- Consider the overhead of each optimization when deciding which to use. Multiprocessing in particular incurs a significant overhead when the potential's operations are not compute-intensive.
