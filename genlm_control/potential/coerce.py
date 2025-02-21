from genlm_control.potential import Potential


class Coerced(Potential):
    """
    Coerce a potential to operate on another vocabulary.

    This class allows a potential to be adapted to work with a different set of tokens,
    defined by a target vocabulary and coersion function.

    This class inherits all methods from [`Potential`][genlm_control.potential.base.Potential].
    Each method delegates to the corresponding method of the underlying potential, but first
    maps any input token sequences from the target vocabulary to the original potential's vocabulary
    using the coercion function.

    Attributes:
        potential (Potential): The original potential instance that is being coerced.
        f (callable): A function that maps sequences of tokens from the target vocabulary to the original potential's vocabulary.
    """

    def __init__(self, potential, target_vocab, f):
        """
        Initialize a Coerced potential.

        Args:
            potential (Potential): The original potential instance that is being coerced.
            target_vocab (list): The target vocabulary that the potential will operate on.
            f (callable): A function that maps sequences of tokens from the target vocabulary to the original potential's vocabulary.

        Raises:
            ValueError: If no valid tokens are found in the target vocabulary that can be mapped to the original potential's vocabulary.
        """
        self.potential = potential
        self.f = f

        valid_tokens = []
        for target_token in target_vocab:
            base_token = f([target_token])
            if set(base_token) <= set(potential.decode):
                valid_tokens.append(target_token)

        if not valid_tokens:
            raise ValueError("No valid tokens found in target vocabulary")

        super().__init__(valid_tokens)

    def _batch_f(self, contexts):
        return [self.f(context) for context in contexts]

    async def complete(self, context):
        return await self.potential.complete(context=self.f(context))

    async def prefix(self, context):
        return await self.potential.prefix(context=self.f(context))

    async def logw_next(self, context):
        Ws = await self.potential.batch_logw_next_seq(
            context=self.f(context), extensions=self.decode_eos
        )
        return self.make_lazy_weights(Ws)

    async def logw_next_seq(self, context, extension):
        return await self.potential.logw_next_seq(
            context=self.f(context), extension=self.f(extension)
        )

    async def batch_complete(self, contexts):
        return await self.potential.batch_complete(contexts=self._batch_f(contexts))

    async def batch_prefix(self, contexts):
        return await self.potential.batch_prefix(contexts=self._batch_f(contexts))

    async def batch_logw_next_seq(self, context, extensions):
        return await self.potential.batch_logw_next_seq(
            context=self.f(context), extensions=self._batch_f(extensions)
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({self.potential!r})"
