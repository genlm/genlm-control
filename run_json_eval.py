import argparse
import json
import os
from typing import List, Tuple

from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams
from genlm.control import ByteLLM, AWRS, BoolFSA, direct_token_sampler
from genlm.control.potential.built_in.llm import PromptedLLM


SYSTEM_PROMPT = (
    "You are a coding assistant. Generate a single-line JSON object that matches the required schema. "
    "Do not include any markdown, commentary, or code fences. Output only the JSON."
)


def build_bytelm(model_name: str = "meta-llama/Llama-3.2-1B-Instruct", K: int = 1) -> ByteLLM:
    llm = load_model_by_name(model_name, backend="hf")
    beam_params = BeamParams(
        K=K,
        prune_threshold=0.0,
        eos_tokens={b"\n", b"\n\n"},
        heal=True,
    )
    return ByteLLM(llm, beam_params)


def json_constraint_regex(allow_countries: List[str]) -> str:
    # Very small, permissive regex for a one-line JSON object with specific keys.
    # Schema: {"name": "<letters/space>", "age": <1-3 digits>, "country": "(US|UK|CA)"}
    # Note: No whitespace-newlines allowed; allow spaces after commas/colons.
    countries = "|".join(map(lambda s: s.replace("|", "\|"), allow_countries))
    pattern = (
        r"\{\s*\"name\"\s*:\s*\"[A-Za-z ]+\"\s*,\s*"
        r"\"age\"\s*:\s*[0-9]{1,3}\s*,\s*"
        r"\"country\"\s*:\s*\"(" + countries + r")\"\s*\}"
    )
    return pattern


def is_valid_json_instance(text: str, allow_countries: List[str]) -> Tuple[bool, str]:
    try:
        obj = json.loads(text)
    except Exception as e:
        return False, f"json_parse_error: {e}"
    if not isinstance(obj, dict):
        return False, "not_object"
    for k in ("name", "age", "country"):
        if k not in obj:
            return False, f"missing_{k}"
    if not isinstance(obj["name"], str) or len(obj["name"]) == 0:
        return False, "bad_name"
    if not isinstance(obj["age"], int) or obj["age"] < 0 or obj["age"] > 150:
        return False, "bad_age"
    if not isinstance(obj["country"], str) or obj["country"] not in allow_countries:
        return False, "bad_country"
    # Ensure one-line output (no newlines)
    if "\n" in text:
        return False, "has_newline"
    return True, "ok"


async def main():
    parser = argparse.ArgumentParser(description="Byte-level JSON constrained generation eval")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--max_tokens", type=int, default=120)
    parser.add_argument("--K", type=int, default=1)
    parser.add_argument("--n_particles", type=int, default=3)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--csv_out", type=str, default=None)
    parser.add_argument("--countries", type=str, default="US,UK,CA")
    parser.add_argument("--token_model", action="store_true", help="Use token-level model (PromptedLLM) instead of byte-level ByteLLM")
    args = parser.parse_args()

    countries = [c.strip() for c in args.countries.split(",") if c.strip()]
    prompt = (
        SYSTEM_PROMPT
        + "\nSchema: {\"name\": string, \"age\": integer, \"country\": one of ["
        + ", ".join(countries)
        + "]}\n"
        + "Return a single-line JSON object exactly matching the schema."
    )

    bytelm = None
    llm_token = None
    condition = None
    # Build a boolean regex constraint over bytes (shared)
    pattern = json_constraint_regex(countries)
    if args.token_model:
        # Token-level path (constrained via regex BoolFSA coerced to token model)
        llm_token = PromptedLLM.from_name(args.model, backend="hf", eos_tokens=[b"\n", b"\n\n"], temperature=1.0)
        tok = llm_token.model.tokenizer
        llm_token.prompt_ids = tok.encode(prompt)
        fsa = BoolFSA.from_regex(pattern)
        condition = fsa.coerce(llm_token, f=b"".join)
    else:
        # Byte-level path with regex constraint
        bytelm = build_bytelm(args.model, K=args.K)
        bytelm.set_prompt_from_str(prompt)
        fsa = BoolFSA.from_regex(pattern)
        condition = fsa.coerce(bytelm, f=b"".join)

    import csv

    all_rows = []
    for run_idx in range(args.runs):
        sampler = AWRS(llm_token if args.token_model else bytelm, condition)
        valid = 0
        results: List[Tuple[str, float, str]] = []

        for i in range(args.num_samples):
            sequences = await sampler.smc(
                n_particles=args.n_particles,
                ess_threshold=0.5,
                max_tokens=args.max_tokens,
                verbosity=0,
            )
            if not sequences.decoded_posterior:
                results.append(("", 0.0, "empty_posterior"))
                all_rows.append([run_idx, i, "", 0.0, "empty_posterior"])
                continue
            best_text, best_w = max(sequences.decoded_posterior.items(), key=lambda kv: kv[1])
            if isinstance(best_text, (bytes, bytearray)):
                try:
                    best_text = best_text.decode("utf-8", errors="ignore")
                except Exception:
                    best_text = str(best_text)

            ok, reason = is_valid_json_instance(best_text.strip(), countries)
            if ok:
                valid += 1
            results.append((best_text, float(best_w), reason))
            all_rows.append([run_idx, i, best_text, float(best_w), reason])

        print(f"\nRun {run_idx}: n={args.num_samples}, valid={valid}, rate={valid/args.num_samples:.3f}")
        for t, w, r in results[:5]:
            print(f"  w={w:.3f} => {t}  [{r}]")

    if args.csv_out:
        with open(args.csv_out, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["run", "index", "text", "weight", "reason"])
            writer.writerows(all_rows)
        print(f"\nSaved details to {args.csv_out}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import asyncio

    asyncio.run(main())


