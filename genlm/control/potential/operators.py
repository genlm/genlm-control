class PotentialOps:
    """Mixin providing operations for potential functions:

    1. Product (`*`): Take the product of two potentials.\n
    2. Coercion (`coerce`): Coerce the potential to operate on another potential's vocabulary.\n
    3. Auto-batching (`to_autobatched`): Create a version that automatically batches concurrent requests to the instance methods.\n
    4. Parallelization (`to_multiprocess`): Create a version that parallelizes operations over multiple processes.\n
    """

    def __mul__(self, other):
        """Take the product of two potentials.

        See [`Product`][genlm.control.potential.product.Product] for more details.

        Args:
            other (Potential): Another potential instance to take the product with.

        Returns:
            (Product): A Product instance representing the unnormalized product of the two potentials.

        Note:
            Potentials must operate on the same token type and the intersection of their vocabularies must be non-empty.
        """
        from genlm.control.potential.product import Product

        return Product(self, other)

    def coerce(self, other, f, prune=True, homomorphic=None):
        """Coerce the current potential to operate on the vocabulary of another potential.

        See [`Coerced`][genlm.control.potential.coerce.Coerced] for more details.

        Args:
            other (Potential): The potential instance whose vocabulary will be used.
            f (callable): A function mapping sequences of tokens from self's vocab to sequences of tokens from other's vocab.
            prune (bool): Whether to prune the coerced potential's vocabulary to only include tokens that can be mapped to the original potential's vocabulary.
                If `False`, the coerced potential's vocabulary will include all tokens from the target vocabulary.
            homomorphic (bool | None): Whether `f` distributes over concatenation
                (enables the trie fast path). `None` probes it; pass `True`/`False`
                to declare it explicitly. See `Coerced`.

        Returns:
            (Coerced): A Potential that operates on the vocabulary of `other`.
        """
        from genlm.control.potential.coerce import Coerced

        return Coerced(self, other.vocab, f=f, prune=prune, homomorphic=homomorphic)

    def to_autobatched(self):
        """Create a new potential instance that automatically batches concurrent requests to the instance methods.

        See [`AutoBatchedPotential`][genlm.control.potential.autobatch.AutoBatchedPotential] for more details.

        Returns:
            (AutoBatchedPotential): A new potential instance that wraps the current potential and automatically batches concurrent requests to the instance methods.
        """
        from genlm.control.potential.autobatch import AutoBatchedPotential

        return AutoBatchedPotential(self)

    def to_multiprocess(self, num_workers=2, spawn_args=None):
        """Create a new potential instance that parallelizes operations using multiprocessing.

        See [`MultiProcPotential`][genlm.control.potential.multi_proc.MultiProcPotential] for more details.

        Args:
            num_workers (int): The number of workers to use in the multiprocessing pool.
            spawn_args (tuple): The positional arguments to pass to the potential's `spawn` method.

        Returns:
            (MultiProcPotential): A new potential instance that wraps the current potential and uses multiprocessing to parallelize operations.

        Note:
            For this method to be used, the potential must implement a picklable `spawn` method.
        """
        from genlm.control.potential.multi_proc import MultiProcPotential

        factory_args = spawn_args or ()
        return MultiProcPotential(
            potential_factory=self.spawn,
            factory_args=factory_args,
            num_workers=num_workers,
        )
