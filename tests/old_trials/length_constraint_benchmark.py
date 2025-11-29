import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np

from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams
from genlm.control import AWRS, ByteLLM, PromptedLLM
from genlm.control.constant import EndOfSequence
from genlm.control.potential.built_in.length import ExactLength

DEFAULT_PROMPTS = [
    "Write a single sentence about ocean exploration.",
    "Explain photosynthesis for kids.",
    "Describe a futuristic city skyline.",
    "Summarize the benefits of renewable energy in one paragraph.",
    "Describe a memorable meal in vivid sensory detail.",
    "Provide instructions for planting a backyard herb garden.",
    "Draft a short product blurb for a smart home thermostat.",
    "Explain the rules of ultimate frisbee to a beginner.",
    "Compose a concise biography of Ada Lovelace.",
    "Outline a morning routine for improved productivity.",
    "Write a brief review of an imaginary sci-fi novel.",
    "Describe the scene at a bustling farmers market.",
    "Give travel advice for visiting Kyoto in spring.",
    "Explain black holes using an everyday analogy.",
    "Create a short dialogue between two rival inventors.",
]

PROMPTS_FILE = Path(__file__).with_name("length_prompts.txt")


def load_prompts() -> List[str]:
    if PROMPTS_FILE.exists():
        return [
            line.strip()
            for line in PROMPTS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return DEFAULT_PROMPTS


PROMPTS = load_prompts()

TARGET_LENGTH = 10
N_PARTICLES = 5
ESS_THRESHOLD = 0.5
MAX_TOKENS = 200


def decode_bytes(context: Iterable[object]) -> bytes:
    chunks: List[bytes] = []
    for token in context:
        if isinstance(token, EndOfSequence):
            break
        if isinstance(token, bytes):
            chunks.append(token)
        elif isinstance(token, int):
            chunks.append(bytes([token]))
        else:
            raise TypeError(f"Unexpected token type: {type(token)}")
    return b"".join(chunks)


def decode_text(context: Iterable[object]) -> str:
    return decode_bytes(context).decode("utf-8", errors="ignore")


@dataclass
class PromptMetrics:
    prompt: str
    mean_logprob: float
    weighted_logprob: float
    satisfaction_rate: float
    mean_length: float
    total_sequences: int
    satisfied_sequences: int


def summarise_sequences(sequences, target_length: int) -> PromptMetrics:
    lengths = []
    satisfied = []
    finite_log_weights = []

    for context, logw in zip(sequences.contexts, sequences.log_weights):
        text = decode_text(context)
        lengths.append(len(text))
        satisfied.append(len(text) == target_length)
        if np.isfinite(logw):
            finite_log_weights.append(logw)

    normalized = sequences.normalized_weights
    log_weights = sequences.log_weights
    finite_mask = np.isfinite(log_weights)

    if np.any(finite_mask):
        weighted_logprob = float(
            np.sum(normalized[finite_mask] * log_weights[finite_mask])
        )
        mean_logprob = float(np.mean(log_weights[finite_mask]))
    else:
        weighted_logprob = float("-inf")
        mean_logprob = float("-inf")

    total_sequences = len(sequences.contexts)
    satisfied_count = int(np.sum(satisfied))

    return PromptMetrics(
        prompt="",
        mean_logprob=mean_logprob,
        weighted_logprob=weighted_logprob,
        satisfaction_rate=satisfied_count / total_sequences if total_sequences else 0.0,
        mean_length=float(np.mean(lengths)) if lengths else 0.0,
        total_sequences=total_sequences,
        satisfied_sequences=satisfied_count,
    )


def aggregate(metrics: List[PromptMetrics]) -> PromptMetrics:
    total_sequences = sum(m.total_sequences for m in metrics)
    satisfied_sequences = sum(m.satisfied_sequences for m in metrics)
    finite_mean_logprobs = [m.mean_logprob for m in metrics if np.isfinite(m.mean_logprob)]
    finite_weighted_logprobs = [
        m.weighted_logprob for m in metrics if np.isfinite(m.weighted_logprob)
    ]

    overall_mean = (
        float(np.mean(finite_mean_logprobs)) if finite_mean_logprobs else float("-inf")
    )
    overall_weighted = (
        float(np.mean(finite_weighted_logprobs))
        if finite_weighted_logprobs
        else float("-inf")
    )

    return PromptMetrics(
        prompt="",
        mean_logprob=overall_mean,
        weighted_logprob=overall_weighted,
        satisfaction_rate=(
            satisfied_sequences / total_sequences if total_sequences else 0.0
        ),
        mean_length=float(
            np.mean([m.mean_length for m in metrics]) if metrics else 0.0
        ),
        total_sequences=total_sequences,
        satisfied_sequences=satisfied_sequences,
    )


async def evaluate_bytellm() -> Tuple[List[PromptMetrics], PromptMetrics]:
    llm = load_model_by_name("gpt2", backend="hf")
    eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam_params = BeamParams(K=5, prune_threshold=0.0, eos_tokens={eos_token})
    bytellm = ByteLLM(llm, beam_params)

    length_potential = ExactLength(
        bytellm.vocab,
        TARGET_LENGTH,
        measure=lambda _: 1,
    )

    metrics: List[PromptMetrics] = []

    try:
        for idx, prompt in enumerate(PROMPTS):
            bytellm.set_prompt_from_str(prompt)
            sampler = AWRS(bytellm, length_potential, seed=42 + idx)
            sequences = await sampler.smc(
                n_particles=N_PARTICLES,
                ess_threshold=ESS_THRESHOLD,
                max_tokens=MAX_TOKENS,
                verbosity=0,
            )
            result = summarise_sequences(sequences, TARGET_LENGTH)
            result.prompt = prompt
            metrics.append(result)
            await sampler.cleanup()
    finally:
        await bytellm.cleanup()

    return metrics, aggregate(metrics)


async def evaluate_promptedllm() -> Tuple[List[PromptMetrics], PromptMetrics]:
    prompted = PromptedLLM.from_name("gpt2", backend="hf")

    def measure(token: bytes) -> int:
        decoded = token.decode("utf-8", errors="ignore")
        return len(decoded) if decoded else len(token)

    length_potential = ExactLength(prompted.vocab, TARGET_LENGTH, measure=measure)

    metrics: List[PromptMetrics] = []

    try:
        for idx, prompt in enumerate(PROMPTS):
            prompted.set_prompt_from_str(prompt)
            sampler = AWRS(prompted, length_potential, seed=84 + idx)
            sequences = await sampler.smc(
                n_particles=N_PARTICLES,
                ess_threshold=ESS_THRESHOLD,
                max_tokens=MAX_TOKENS,
                verbosity=0,
            )
            result = summarise_sequences(sequences, TARGET_LENGTH)
            result.prompt = prompt
            metrics.append(result)
            await sampler.cleanup()
    finally:
        prompted.model.clear_cache()

    return metrics, aggregate(metrics)


def display_results(name: str, per_prompt: List[PromptMetrics], summary: PromptMetrics):
    print(f"\n=== {name} ===")
    print(
        f"Satisfaction: {summary.satisfaction_rate * 100:.1f}% "
        f"({summary.satisfied_sequences}/{summary.total_sequences})"
    )
    print(f"Mean logprob (finite): {summary.mean_logprob:.3f}")
    print(f"Weighted logprob (finite): {summary.weighted_logprob:.3f}")
    print(f"Mean decoded length: {summary.mean_length:.1f}")

    for metrics in per_prompt:
        print(
            f"- Prompt: {metrics.prompt}\n"
            f"  logprob mean={metrics.mean_logprob:.3f}, "
            f"weighted={metrics.weighted_logprob:.3f}, "
            f"satisfaction={metrics.satisfaction_rate * 100:.1f}%, "
            f"avg length={metrics.mean_length:.1f}"
        )


async def main():
    bytellm_results, bytellm_summary = await evaluate_bytellm()
    prompted_results, prompted_summary = await evaluate_promptedllm()

    display_results("ByteLLM", bytellm_results, bytellm_summary)
    display_results("PromptedLLM", prompted_results, prompted_summary)


if __name__ == "__main__":
    asyncio.run(main())

