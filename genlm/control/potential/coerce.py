from genlm.control.potential import Potential
from itertools import chain
import asyncio


class Coerced(Potential):
    """
    Coerce a potential to operate on another vocabulary.

    This class allows a potential to be adapted to work with a different set of tokens,
    defined by a target vocabulary and coersion function.

    This class inherits all methods from [`Potential`][genlm.control.potential.base.Potential].
    Each method delegates to the corresponding method of the underlying potential, but first
    maps any input token sequences from the target vocabulary to the original potential's vocabulary
    using the coercion function.

    Formally, if $f$ is the coercion function, then for any sequence $x_1, \\ldots, x_n$ of tokens from the target vocabulary,
    $$
    \\textsf{Coerced.prefix}(x_1, \\ldots, x_n) = \\textsf{Coerced.potential.prefix}(f(x_1, \\ldots, x_n))
    $$

    $$
    \\textsf{Coerced.complete}(x_1, \\ldots, x_n) = \\textsf{Coerced.potential.complete}(f(x_1, \\ldots, x_n))
    $$

    Attributes:
        potential (Potential): The original potential instance that is being coerced.
        f (callable): A function that maps sequences of tokens from the target vocabulary to sequences of tokens from
            the original potential's vocabulary.

    Note:
        The coerced potential's vocabulary will by default be pruned to only include tokens that can be mapped to the original potential's vocabulary
        via the coercion function (i.e. `set(f([x])) <= set(potential.vocab)`). If no such tokens are found, a `ValueError` is raised.
        This behavior can be overridden by setting `prune=False`, in which case the coerced potential's vocabulary will include all tokens from the target vocabulary.
    """

    def __init__(self, potential, target_vocab, f, prune=True):
        """
        Initialize a Coerced potential.

        Args:
            potential (Potential): The original potential instance that is being coerced.
            target_vocab (list): The target vocabulary that the potential will operate on.
                Each element of `target_vocab` must be hashable.
            f (callable): A function that maps iterables of tokens from the target vocabulary
                to the original potential's vocabulary.
            prune (bool): Whether to prune the coerced potential's vocabulary to only include tokens that can be mapped to the original potential's vocabulary.
                If `False`, the coerced potential's vocabulary will include all tokens from the target vocabulary.

        Raises:
            ValueError: If no valid tokens are found in the target vocabulary that can be mapped to the original potential's vocabulary.
        """
        self.potential = potential
        self.f = f

        if prune:
            # Extract items from potential.vocab for comparison
            # For Token objects, extract byte values from byte_string to match set() behavior
            if potential.vocab and hasattr(potential.vocab[0], "byte_string"):
                potential_bytes = set()
                for tok in potential.vocab:
                    potential_bytes.update(
                        tok.byte_string
                    )  # Add byte values (integers)
            else:
                potential_bytes = set(potential.vocab)

            # Check if target_vocab contains Token objects
            has_token_objects = target_vocab and hasattr(target_vocab[0], "byte_string")

            tokens = []
            for target_token in target_vocab:
                # If target_token is a Token object, extract byte_string for the coercion function
                if has_token_objects:
                    target_token_for_f = [target_token.byte_string]
                else:
                    target_token_for_f = [target_token]

                base_token = f(target_token_for_f)
                if set(base_token) <= potential_bytes:
                    tokens.append(target_token)
        else:
            tokens = target_vocab

        if not tokens:
            raise ValueError("No valid tokens found in target vocabulary")

        # Store whether vocab contains Token objects for efficient extraction
        self._has_token_vocab = tokens and hasattr(tokens[0], "byte_string")

        super().__init__(tokens)

    def _extract_bytes(self, context):
        """Extract byte_strings from Token objects in context for the coercion function."""
        if not self._has_token_vocab:
            return context

        return [
            tok.byte_string if hasattr(tok, "byte_string") else tok for tok in context
        ]

    def _batch_f(self, contexts):
        return [self.f(self._extract_bytes(context)) for context in contexts]

    async def complete(self, context):
        return await self.potential.complete(
            context=self.f(self._extract_bytes(context))
        )

    async def prefix(self, context):
        return await self.potential.prefix(context=self.f(self._extract_bytes(context)))

    async def logw_next(self, context):
        Ws = self.alloc_logws()
        ctx = self.f(self._extract_bytes(context))
        ctx_w = await self.potential.prefix(ctx)
        Ws[-1] = await self.potential.complete(ctx) - ctx_w
        exts = [
            self.f(self._extract_bytes(list(chain(context, [x])))) for x in self.vocab
        ]  # slow!!
        Ws[:-1] = await self.potential.batch_prefix(exts) - ctx_w
        return self.make_lazy_weights(Ws)

    async def batch_complete(self, contexts):
        return await self.potential.batch_complete(contexts=self._batch_f(contexts))

    async def batch_prefix(self, contexts):
        return await self.potential.batch_prefix(contexts=self._batch_f(contexts))

    async def batch_logw_next(self, contexts):
        return await asyncio.gather(*[self.logw_next(context) for context in contexts])

    def __repr__(self):
        return f"{self.__class__.__name__}({self.potential!r})"
