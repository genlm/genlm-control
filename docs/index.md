![Logo](logo.png)

[![Docs](https://github.com/genlm/genlm-control/actions/workflows/docs.yml/badge.svg)](https://genlm.github.io/genlm-control/)
[![Tests](https://github.com/genlm/genlm-control/actions/workflows/pytest.yml/badge.svg)](https://genlm.github.io/genlm-control/)
[![codecov](https://codecov.io/github/genlm/genlm-control/graph/badge.svg?token=665ffkDXvZ)](https://codecov.io/github/genlm/genlm-control)

GenLM Control is a library for controlled generation from language models using programmable constraints. It leverages sequential Monte Carlo (SMC) methods to efficiently generate text that satisfies constraints or preferences encoded by arbitrary potential functions.

## Quick Start

This library can be installed using pip:

```bash
pip install genlm-control
```

See [DEVELOPING.md](https://github.com/genlm/genlm-control/tree/main/DEVELOPING.md) for details on how to install the project for development.

## Examples

### Controlling an LLM with a regular expression

This example demonstrates how to constrain an LLM using a regular expression.

```python
from genlm.control import PromptedLLM, BoolFSA, AWRS

# Create a language model potential.
llm = PromptedLLM.from_name("gpt2")
llm.set_prompt_from_str("Here is my honest opinion:")

# Create a finite-state automaton potential using a regular expression.
fsa = BoolFSA.from_regex(r" SMC is (🔥🔥|😍😍|🤌🤌) with LMs")

# Coerce the FSA so that it operates on the token type of the language model.
coerced_fsa = fsa.coerce(llm, f=b"".join)

# Create a token sampler that combines the language model and FSA.
token_sampler = AWRS(llm, coerced_fsa)

# Generate text using SMC.
# Generation is asynchronous; use `await` if calling in an async context (like in an async
# function or in a Jupyter notebook) and `asyncio.run(token_sampler.smc(...))` otherwise.
sequences = await token_sampler.smc(
    n_particles=10, # Number of candidate sequences to maintain
    ess_threshold=0.5, # Threshold for resampling
    max_tokens=30, # Maximum sequence length
    verbosity=1 # Print particles at each step
)

sequences.decoded_posterior
# Example output:
# {
#   ' SMC is 🔥🔥 with LMs': 1.0,
# }
```

### Controlling an LLM with a JSON schema

This example demonstrates how to control an LLM to generate JSON objects that match a given schema.

```python
import json
from genlm.control import PromptedLLM, JsonSchema, AWRS

person_schema = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "enum": ["Alice", "Bob", "Charlie"],
            "description": "The name of the person"
        },
        "age": {
            "type": "integer",
            "minimum": 20,
            "maximum": 80,
            "description": "The age of the person"
        },
    },
}

book_schema = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "minLength": 1,
            "description": "The title of the book"
        },
        "pages": {
            "type": "integer",
            "minimum": 1,
            "maximum": 2000,
            "description": "The number of pages in the book"
        },
        "genre": {
            "type": "string",
            "enum": ["fiction", "non-fiction", "mystery"],
            "description": "The genre of the book"
        }
    },
}

# Create a language model potential.
# Since this task is harder, we use a larger model.
# (You will need to login via the Hugging Face CLI and have access to the model.)
llm = PromptedLLM.from_name(
    "meta-llama/Llama-3.2-1B-Instruct",
    eos_byte_strings=[b"<|eom_id|>", b"<|eot_id|>"],
    temperature=0.8
)

# Set the prompt for the language model.
# Since we are using an instruction-tuned model, we use the chat template.
# The prompt contains an example of a schema and a generated object,
# followed by the schema we want to match.
llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": "You need to generate a JSON object that matches the schema below. Only generate the JSON object on a single line with no other text."},
        {"role": "user", "content": json.dumps(person_schema)},
        {"role": "assistant", "content": '{"name": "Alice", "age": 30}'},
        {"role": "user", "content": json.dumps(book_schema)},
    ],
    tokenize=True,
    add_generation_prompt=True
)

# Create a schema potential.
schema_potential = JsonSchema(book_schema)

# Coerce the schema potential so that it operates on the token type of the language model.
coerced_schema = schema_potential.coerce(llm, f=b"".join)

# Create a token sampler that combines the language model and the schema potential.
token_sampler = AWRS(llm, coerced_schema)

# Generate text using SMC.
# Generation is asynchronous; use `await` if calling in an async context (like in an async
# function or in a Jupyter notebook) and `asyncio.run(token_sampler.smc(...))` otherwise.
sequences = await token_sampler.smc(
    n_particles=2, # Number of candidate sequences to maintain
    ess_threshold=0.5, # Threshold for resampling
    max_tokens=30, # Maximum sequence length
    verbosity=1 # Print particles at each step
)

# Show the inferred posterior distribution over complete UTF-8 decodable sequences.
sequences.decoded_posterior
# Example output:
# {
#   '{"title": "The Lord of the Rings", "pages": 1200, "genre": "fiction"}': 0.5008318164809697,
#   '{"title": "The Great Gatsby", "pages": 178, "genre": "fiction"}': 0.49916818351903025,
# }
```

### More examples

See [getting_started.md](getting_started.md) to get an overview of the full range of features, including how to specify custom potential functions.

## Development

See [DEVELOPING.md](https://github.com/genlm/genlm-control/tree/main/DEVELOPING.md) for details on how to install the project locally.
