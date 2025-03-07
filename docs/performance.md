# Performance Optimizations

The `genlm-control` library offers two key performance optimizations for instances of the `Potential` class:

- **Autobatching**: Automatically batches concurrent requests to the potential's instances methods
- **Multiprocessing**: Runs multiple instances of a `Potential` in parallel across CPU cores


## Auto-batching

Auto-batching improves performance when a `Potential` class's batch methods (`batch_complete`,  `batch_prefix`, `batch_logw_next`, `batch_score`) are more efficient than sequentially running individual instance methods.

### Usage

To enable auto-batching, use the `to_autobatched()` method:

```python
autobatched_potential = potential.to_autobatched()
# Use it exactly like a regular potential - batching happens automatically
results = await asyncio.gather(
    *(autobatched.complete(seq) for seq in sequences) # These will batched and processed by batch_complete
)
```

This creates a new potential that is a wrapper ([`AutoBatchedPotential`](../reference/genlm_control/potential/autobatch)) around the original potential. The wrapper automatically collects concurrent requests in the background and processes them together using the potential's batch methods. This happens transparently without requiring changes to your code structure.

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

This creates a new potential that is a wrapper ([`MultiProcPotential`](../reference/genlm_control/potential/multi_proc)) around the original potential. The wrapper asynchronously distributes requests across multiple processes (in a non-blocking manner). This allows you to scale your computations across multiple cores without changing your code structure.

### Requirements

For multiprocessing to work, the potential must implement a picklable `spawn()` method that creates a new instance of the potential. Only some built-in `Potential` classes support this by default. Custom potentials need to implement their own `spawn()` method.

### Performance Benefits

Multiprocessing improves performance for both batched methods (`batch_complete`, `batch_prefix`, `batch_logw_next`) and unbatched methods (`complete`, `prefix`, `logw_next`).

In the batched case, requests within a batch are processed in parallel across workers. For individual method calls, requests are distributed to available worker processes and are executed asynchronously.

## When to use each optimization

> **Note:** Built-in `Potential` classes that can benefit from auto-batching support (e.g., `PromptedLLM`) will have auto-batching enabled by default.

- Use auto-batching when the potential's batch operations are more efficient than sequential operations
- Use multiprocessing when the potential's operations are compute-intensive and can benefit from parallel processing
- Consider the overhead of each optimization when deciding which to use. Multiprocessing in particular incurs a significant overhead when the potential's operations are not compute-intensive.
