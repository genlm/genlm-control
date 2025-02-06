class PotentialOps:
    """Mixin providing operations for potential functions:

    1. Composition (*): Take the product of two potentials.\n
    2. Lifting (lift): Lift the potential to operate on another potential's vocabulary.\n
    3. Auto-batching (to_auto_batched): Create a version that automatically batches concurrent requests to the instance methods.\n
    4. Parallelization (to_multiprocess): Create a version that parallelizes batch operations over multiple processes.\n
    """

    def __mul__(self, other):
        """Take the product of two potentials.

        The intersection of the vocabularies of both potentials must be non-empty.

        Args:
            other (Potential): Another potential instance to take the product with.

        Returns:
            (Product): A Product instance representing the unnormalized product of the two potentials.
        """
        from genlm_control.product import Product

        return Product(self, other)

    def lift(self, other, f, g):
        """Lift the current potential to operate on the vocabulary of another potential.

        Args:
            other (Potential): The potential instance whose vocabulary will be used.

        Returns:
            (Lifted): A Potential that operates on the vocabulary of `other`.
        """
        from genlm_control.potential.lifted import Lifted

        return Lifted(self, other.decode, f=f, g=g)

    def to_autobatched(self):
        """Create a new potential instance that automatically batches concurrent requests to the instance methods.

        Returns:
            (AutoBatchedPotential): A new potential instance that wraps the current potential and
            automatically batches concurrent requests to the instance methods.
        """
        from genlm_control.potential.autobatch import AutoBatchedPotential

        return AutoBatchedPotential(self)

    def to_multiprocess(self, num_workers=2, spawn_args=None):
        """Create a new potential instance that parallelizes batch operations
        using multiprocessing.

        Args:
            num_workers (int): The number of workers to use in the multiprocessing pool.
            spawn_args (tuple): The positional arguments to pass to the potential's `spawn` method.

        Returns:
            (MPPotential) A new potential instance that wraps the current potential and uses multiprocessing to parallelize
            batch operations.

        Note:
            For this method to be used, the potential must implement a picklable `spawn` method.
        """
        from genlm_control.potential.mp import MPPotential

        factory_args = spawn_args or ()
        return MPPotential(
            potential_factory=self.spawn,
            factory_args=factory_args,
            num_workers=num_workers,
        )
