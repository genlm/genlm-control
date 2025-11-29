import asyncio
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from genlm.bytes import BeamParams
from genlm.control import AWRS, BoolFSA, ByteLLM, PromptedLLM
from genlm.control.constant import EndOfSequence
from genlm.backend import load_model_by_name

FRUITS = {"peach", "pear", "apple", "mango"}
VEGETABLES = {"peas", "peanut", "pepper"}
ALL_WORDS = sorted(FRUITS | VEGETABLES, key=len, reverse=True)

PROMPT = (
    "Produce a bracketed list with five items, each chosen from this set: "
    + ", ".join(ALL_WORDS)
    + ". Only output the bracketed list."
)

N_PARTICLES = 5
ESS_THRESHOLD = 0.5
MAX_TOKENS = 120


def _decode_bytes(tokens: Iterable[object]) -> bytes:
    chunks: List[bytes] = []
    for token in tokens:
        if isinstance(token, EndOfSequence):
            break
        if isinstance(token, bytes):
            chunks.append(token)
        elif isinstance(token, int):
            chunks.append(bytes([token]))
        else:
            raise TypeError(f"Unexpected token type: {type(token)}")
    return b"".join(chunks)


def _decode_text(tokens: Iterable[object]) -> str:
    return _decode_bytes(tokens).decode("utf-8", errors="ignore")


def _parse_items(text: str) -> Tuple[bool, List[str]]:
    stripped = text.strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return False, []
    body = stripped[1:-1]
    items = [item.strip().lower() for item in body.split(",")]
    return True, items


def _word_pattern(words: Iterable[str]) -> str:
    escaped = [word.replace(" ", "\\ ") for word in words]
    return "(" + "|".join(escaped) + ")"


def _build_constraint():
    pattern = _word_pattern(ALL_WORDS)
    regex = r"\[ *" + pattern + r"(?:, *" + pattern + r"){4} *\]"
    return BoolFSA.from_regex(regex)


@dataclass
class ResultMetrics:
    prompt: str
    satisfied_mass: float
    vegetable_mass: float
    top_sequences: List[Tuple[float, str]]


async def run_sampler(base, constraint):
    sampler = AWRS(base, constraint)
    sequences = await sampler.smc(
        n_particles=N_PARTICLES,
        ess_threshold=ESS_THRESHOLD,
        max_tokens=MAX_TOKENS,
        verbosity=1,
    )
    await sampler.cleanup()
    return sequences


def analyze_sequences(sequences) -> ResultMetrics:
    norm = sequences.normalized_weights
    satisfied_mass = 0.0
    vegetable_mass = 0.0
    decoded = []

    for (context, logw), weight in zip(sequences, norm):
        text = _decode_text(context)
        ok, items = _parse_items(text)
        if not ok:
            continue
        is_all_fruits = all(item in FRUITS for item in items)
        has_vegetable = any(item in VEGETABLES for item in items)
        if is_all_fruits:
            satisfied_mass += weight
        if has_vegetable:
            vegetable_mass += weight
        decoded.append((weight, text, logw))

    decoded.sort(key=lambda tup: tup[0], reverse=True)
    top = [(float(weight), text) for weight, text, _ in decoded[:10]]

    return ResultMetrics(
        prompt=PROMPT,
        satisfied_mass=float(satisfied_mass),
        vegetable_mass=float(vegetable_mass),
        top_sequences=top,
    )


async def evaluate_bytellm():
    llm = load_model_by_name("gpt2", backend="hf")
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam_params = BeamParams(K=6, prune_threshold=0.0, eos_tokens={model_eos_token})
    byte_llm = ByteLLM.from_name("gpt2", beam_params=beam_params, backend="hf")
    byte_llm.set_prompt_from_str(PROMPT)
    try:
        constraint = _build_constraint().coerce(byte_llm, f=b"".join)
        sequences = await run_sampler(byte_llm, constraint)
        return analyze_sequences(sequences)
    finally:
        await byte_llm.cleanup()


async def evaluate_promptedllm():
    prompted = PromptedLLM.from_name("gpt2", backend="hf")
    prompted.set_prompt_from_str(PROMPT)
    constraint = _build_constraint().coerce(prompted, f=b"".join)
    sequences = await run_sampler(prompted, constraint)
    prompted.model.clear_cache()
    return analyze_sequences(sequences)


def display_results(label: str, metrics: ResultMetrics):
    print(f"\n=== {label} ===")
    print(f"Mass on all-fruit outputs: {metrics.satisfied_mass * 100:.2f}%")
    print(f"Mass containing vegetables: {metrics.vegetable_mass * 100:.2f}%")
    print("Top sequences (weight, text):")
    for weight, text in metrics.top_sequences:
        print(f"  {weight:.3f}  {text}")


async def main():
    bytellm_metrics = await evaluate_bytellm()
    prompted_metrics = await evaluate_promptedllm()
    display_results("ByteLLM", bytellm_metrics)
    display_results("PromptedLLM", prompted_metrics)


if __name__ == "__main__":
    asyncio.run(main())

