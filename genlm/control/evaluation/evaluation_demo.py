import asyncio
import numpy as np
from metrics import kl_divergence_direct, kl_divergence_potentials


class MockLanguageModel:
    """Mock language model for demonstration."""

    def __init__(self, vocab_probs):
        self.vocab_probs = vocab_probs
        self.vocab = list(vocab_probs.keys())
        self.probs = np.array(list(vocab_probs.values()))
        self.probs = self.probs / self.probs.sum()  # Normalize

    def log_prob(self, samples):
        """Return log probabilities for samples."""
        if isinstance(samples, str):
            samples = [samples]

        log_probs = []
        for sample in samples:
            if sample in self.vocab_probs:
                prob = self.vocab_probs[sample] / sum(self.vocab_probs.values())
                log_probs.append(np.log(prob))
            else:
                log_probs.append(np.log(1e-10))  # Small prob for unseen

        return np.array(log_probs)

    def sample(self, n_samples):
        """Sample from the model."""
        return np.random.choice(self.vocab, size=n_samples, p=self.probs).tolist()


def demo_mock_models():
    """Demo KL divergence estimation with mock models."""
    print("=== Mock Language Models Demo ===")

    # Create two language models
    model_p = MockLanguageModel({"hello": 0.6, "world": 0.3, "goodbye": 0.1})

    model_q = MockLanguageModel({"hello": 0.4, "world": 0.4, "goodbye": 0.2})

    # Generate samples from model P
    samples = model_p.sample(1000)
    print(f"Generated {len(samples)} samples from model P")

    # Compute KL divergence using direct log probability evaluation
    kl_div = kl_divergence_direct(model_p, model_q, samples)
    print(f"KL(P||Q) = {kl_div:.4f}")

    # Show the clean formula
    print("\nThis implements: KL(P||Q) = E_P[log P(x) - log Q(x)]")
    print("Where:")
    print("  logp = model_p.log_prob(samples)")
    print("  logq = model_q.log_prob(samples)")
    print("  kl = (logp - logq).mean()")
    print()


async def demo_prompted_llm():
    """Demo KL divergence estimation with real PromptedLLM models."""
    print("=== Real PromptedLLM Demo ===")

    try:
        from genlm.control import PromptedLLM, direct_token_sampler

        # Create two PromptedLLM models with different prompts/temperatures
        print("Loading GPT-2 models...")

        # Model P: Creative writing prompt with higher temperature
        model_p = PromptedLLM.from_name(
            "gpt2", backend="hf", temperature=1.2, eos_tokens=[b"."]
        )
        model_p.set_prompt_from_str("Once upon a time, in a magical forest,")

        # Model Q: Factual prompt with lower temperature
        model_q = model_p.spawn()
        model_q.set_prompt_from_str("The capital of France is")
        model_q.temperature = 0.8

        # Actually sample completions from model P using SMC
        print("Sampling completions from creative model...")
        sampler_p = direct_token_sampler(model_p)
        sequences_p = await sampler_p.smc(
            n_particles=10, max_tokens=15, ess_threshold=0.5
        )

        samples = list(sequences_p.decoded_posterior.keys())
        print(f"Generated {len(samples)} unique completed sequences from model P")

        print("\nSample completions:")
        for i, sample in enumerate(samples[:3]):
            print(f"  {i + 1}. '{sample}'")

        if not samples:
            print(
                "⚠️  No completed sequences found. Try increasing max_tokens or reducing ess_threshold."
            )
            return

        # Compute KL divergence using the potentials interface
        kl_div = await kl_divergence_potentials(model_p, model_q, samples)
        print(f"\nKL(Creative||Factual) = {kl_div:.4f}")

        print("\nIndividual log probabilities:")
        for sample in samples[:2]:  # Just show first 2
            tokens = model_p.tokenize(sample)
            logp_p = await model_p.complete(tokens)
            logp_q = await model_q.complete(tokens)
            print(f"  '{sample}':")
            print(f"    Creative model: {logp_p:.3f}")
            print(f"    Factual model:  {logp_q:.3f}")
            print(f"    Log ratio:      {logp_p - logp_q:.3f}")

        print("\n🎯 Key insight: We sampled actual completions from model P,")
        print("   then evaluated both P(x) and Q(x) on those samples!")
        print("   This gives us KL(P||Q) = E_P[log P(x) - log Q(x)]")

        # Cleanup
        await sampler_p.cleanup()

    except ImportError as e:
        print(f"Skipping PromptedLLM demo - missing dependencies: {e}")
    except Exception as e:
        print(f"PromptedLLM demo failed: {e}")
        print("This is expected if running without proper model setup")

    print()


async def main():
    """Run all demonstrations."""
    print("KL Divergence Estimation Demo")
    print("=" * 40)
    print()
    await demo_prompted_llm()


if __name__ == "__main__":
    asyncio.run(main())
