import asyncio
import numpy as np

from genlm.control.util import to_numpy


class PotentialTests:
    """A mixin class providing testing utilities for validating Potential implementations.

    This class provides methods to verify the mathematical consistency and correctness
    of Potential implementations through various assertions:

    - logw_next consistency: Verifies that token-level log weights are consistent with
      prefix and complete scores.
    - Autoregressive factorization: Validates that complete scores factor correctly as
      a sum of log token weights (with an additional correction term corresponding to the
      prefix weight of the empty sequence).
    - Batch consistency: Ensures batch operations produce identical results to
      their non-batch counterparts.

    All Potential instances inherit from this class and thus automatically gain access to these
    testing utilities.
    """

    colors = {
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "red": "\033[91m",
        "reset": "\033[0m",
    }

    async def assert_logw_next_consistency(
        self, context, rtol=1e-3, atol=1e-5, top=None, verbosity=0, method_args=()
    ):
        """
        Assert that `logw_next` is consistent with `prefix` and `complete`.

        For a `context` of tokens $x_1, \\ldots, x_{n-1}$, this checks (in log space) whether:

        $$
        \\textsf{logw\\_next}(x_n | x_1, \\ldots, x_{n-1}) = \\textsf{score}(x_1, \\ldots, x_n) - \\textsf{prefix}(x_1, \\ldots, x_{n-1})
        $$
        for $x_n \\in \\textsf{vocab_eos}$, i.e., the potential's vocabulary and end-of-sequence token.

        Args:
            context (list): Context to test.
            rtol (float): Relative tolerance for floating point comparison.
            atol (float): Absolute tolerance for floating point comparison.
            top (int):If specified, only test the top-k tokens by log weight. If None, test all tokens.
            verbosity (int): Verbosity level.
            method_args (tuple): Positional arguments to pass to `logw_next`, `prefix`, and `batch_score`.
                Defaults to empty tuple.

        Raises:
            AssertionError: If `logw_next` is not consistent with `prefix` and `complete`.
        """
        top_logw_next = (await self.logw_next(context, *method_args)).materialize(
            top=top
        )
        tokens = list(top_logw_next.keys())
        extended = [[*context, x] for x in tokens]

        context_w = await self.prefix(context, *method_args)
        extended_ws = np.array(
            await asyncio.gather(*[self.score(e, *method_args) for e in extended])
        )

        wants = np.array([top_logw_next[x] for x in tokens])
        haves = extended_ws - context_w

        errors, valids = [], []
        for i, (want, have) in enumerate(zip(wants, haves)):
            abs_diff, rel_diff = self._compute_diff(want, have)
            info = (want, have, abs_diff, rel_diff, tokens[i])
            (valids if abs_diff <= atol and rel_diff <= rtol else errors).append(info)

        if valids and verbosity > 0:
            print(
                f"{self.colors['green']}logw_next consistency with context={context!r} satisfied for tokens:{self.colors['reset']}\n"
            )
            for valid in valids:
                want, have, abs_diff, rel_diff, token = valid
                print(
                    self._format_diff(want, have, abs_diff, rel_diff, atol, rtol, token)
                )

        if errors:
            error_msg = f"{self.colors['red']}logw_next consistency with context={context!r} not satisfied for tokens:{self.colors['reset']}\n\n"
            for error in errors:
                want, have, abs_diff, rel_diff, token = error
                error_msg += self._format_diff(
                    want, have, abs_diff, rel_diff, atol, rtol, token
                )
            raise AssertionError(error_msg)

    async def assert_autoreg_fact(
        self, context, rtol=1e-3, atol=1e-5, verbosity=0, method_args=()
    ):
        """
        Assert that `complete` factors as an autoregressive sum of `logw_next`s.

        For a `context` of tokens $x_1, \\ldots, x_n$, this checks (in log space) whether:

        $$
        \\textsf{complete}(x_1, \\ldots, x_n) - \\textsf{prefix}(\\epsilon) = \\textsf{logw\\_next}(\\textsf{eos} \\mid x_1, \\ldots, x_{n}) + \\sum_{i=1}^{n} \\textsf{logw_next}(x_i \\mid x_1, \\ldots, x_{i-1})
        $$
        where $\\epsilon$ is the empty sequence.

        Args:
            context (list): Context to test.
            rtol (float): Relative tolerance for floating point comparison.
            atol (float): Absolute tolerance for floating point comparison.
            verbosity (int): Verbosity level.
            method_args (tuple): Positional arguments to pass to `complete`, `prefix`, and `logw_next`.
                Defaults to empty tuple.

        Raises:
            AssertionError: If the autoregressive factorization is not satisfied.
        """
        want = (await self.complete(context, *method_args)) - (
            await self.prefix([], *method_args)
        )

        logw_next_results = await asyncio.gather(
            *[self.logw_next(context[:i], *method_args) for i in range(len(context))],
            self.logw_next(context, *method_args),
        )

        have = (
            sum(logw_next_results[i][context[i]] for i in range(len(context)))
            + logw_next_results[-1][self.eos]
        )

        abs_diff, rel_diff = self._compute_diff(want, have)
        if abs_diff > atol or rel_diff > rtol:
            error_msg = (
                f"{self.colors['red']}Factorization not satisfied for context {context!r}:{self.colors['reset']}\n"
                + self._format_diff(want, have, abs_diff, rel_diff, atol, rtol)
            )
            raise AssertionError(error_msg)

        if verbosity > 0:
            print(
                f"{self.colors['green']}Factorization property satisfied for context {context}:{self.colors['reset']}\n"
            )
            print(self._format_diff(want, have, abs_diff, rel_diff, atol, rtol))

    async def assert_batch_consistency(
        self,
        contexts,
        rtol=1e-3,
        atol=1e-5,
        verbosity=0,
        batch_method_args=(),
        method_args=(),
    ):
        """
        Assert that batch results are equal to non-batch results.

        Args:
            contexts (list[list]): Contexts to test.
            rtol (float): Relative tolerance for floating point comparison.
            atol (float): Absolute tolerance for floating point comparison.
            verbosity (int): Verbosity level.
            batch_method_args (tuple): Positional arguments to pass to batch methods.
                Defaults to empty tuple.
            method_args (tuple): Positional arguments to pass to underlying potential methods.
                Defaults to empty tuple.

        Raises:
            AssertionError: If the batch results are not equal to the non-batch results.
        """
        batch_logw_nexts = await self.batch_logw_next(contexts, *batch_method_args)
        batch_scores = await self.batch_score(contexts, *batch_method_args)

        for i, context in enumerate(contexts):
            logw_next = await self.logw_next(context, *method_args)
            # batch_logw_next now returns ONE batched LazyWeights ([N, V+1]); row i is .weights[i]
            batch_row = to_numpy(batch_logw_nexts.weights[i])
            try:
                np.testing.assert_allclose(
                    batch_row, to_numpy(logw_next.weights), rtol=rtol, atol=atol
                )
                if verbosity > 0:
                    print(
                        f"{self.colors['green']}Batch logw_next consistency satisfied for context {context}:{self.colors['reset']}"
                    )
                    print(
                        f"{self.colors['green']}Non-batched: {logw_next.weights}\n"
                        + f"{self.colors['green']}Batched:     {batch_row}{self.colors['reset']}\n"
                    )
            except AssertionError:
                raise AssertionError(
                    f"{self.colors['red']}Batch logw_next mismatch for context {context}:{self.colors['reset']}\n"
                    + f"{self.colors['green']}Non-batched: {logw_next.weights}\n"
                    + f"{self.colors['red']}Batched:     {batch_row}{self.colors['reset']}"
                )

            score = await self.score(context, *method_args)
            abs_diff, rel_diff = self._compute_diff(score, batch_scores[i])
            if abs_diff > atol or rel_diff > rtol:
                raise AssertionError(
                    f"{self.colors['red']}Batch score mismatch for context {context}:{self.colors['reset']}\n"
                    + f"{self.colors['green']}Non-batched: {score}\n"
                    + f"{self.colors['red']}Batched:     {batch_scores[i]}{self.colors['reset']}"
                )
            elif verbosity > 0:
                print(
                    f"{self.colors['green']}Batch score consistency satisfied for context {context}:{self.colors['reset']}"
                )
                print(
                    f"{self.colors['green']}Non-batched: {score}\n"
                    + f"{self.colors['green']}Batched:     {batch_scores[i]}{self.colors['reset']}\n"
                )

    def _compute_diff(self, want, have):
        is_inf = want == float("-inf") and have == float("-inf")
        abs_diff = 0 if is_inf else abs(want - have)
        if want == 0:
            rel_diff = 0 if have == 0 else float("inf")
        else:
            rel_diff = 0 if is_inf else abs((want - have) / want)
        return abs_diff, rel_diff

    def _format_diff(self, want, have, abs_diff, rel_diff, atol, rtol, token=None):
        abs_diff_str = (
            f"{self.colors['cyan']}Abs Diff: {abs_diff:.6f} <= {atol=}\033[0m"
        )
        rel_diff_str = (
            f"{self.colors['magenta']}Rel Diff: {rel_diff:.6f} <= {rtol=}\033[0m"
        )

        want_str = f"{self.colors['green']}Expected: {want:.6f}{self.colors['reset']}"
        have_clr = (
            self.colors["yellow"]
            if abs_diff <= atol and rel_diff <= rtol
            else self.colors["red"]
        )
        have_str = f"{have_clr}Actual:   {have:.6f}{self.colors['reset']}"

        if abs_diff <= atol:
            abs_diff_str = f"{self.colors['green']}Abs Diff: {abs_diff:.6f} <= {atol=}{self.colors['reset']}"
        else:
            abs_diff_str = f"{self.colors['red']}Abs Diff: {abs_diff:.6f} > {atol=}{self.colors['reset']}"

        if rel_diff <= rtol:
            rel_diff_str = f"{self.colors['green']}Rel Diff: {rel_diff:.6f} <= {rtol=}{self.colors['reset']}"
        else:
            rel_diff_str = f"{self.colors['red']}Rel Diff: {rel_diff:.6f} > {rtol=}{self.colors['reset']}"

        token_str = (
            f"{self.colors['blue']}Token:    {token}{self.colors['reset']}\n"
            if token
            else ""
        )
        return f"{token_str}{want_str}\n{have_str}\n{abs_diff_str}\n{rel_diff_str}\n\n"
