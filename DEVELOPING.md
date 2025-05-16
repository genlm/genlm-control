# Developer's Guide

This guide describes how to complete various tasks you'll encounter when working
on the `genlm-control` codebase.

## Requirements

- Python >= 3.11
- The core dependencies listed in the `pyproject.toml` file of the repository.

## Installation

Clone the repository:
```bash
git clone git@github.com:genlm/genlm-control.git
cd genlm-control
```
and install with pip:

```bash
pip install -e ".[test,docs]"
```

This installs the dependencies needed for testing (test) and documentation (docs).

For faster and less error-prone installs, consider using [`uv`](https://github.com/astral-sh/uv):

```bash
uv pip install -e ".[test,docs]"
```

It is also recommended to use a dedicated environment.

With conda:
```bash
conda create -n genlm python=3.11
conda activate genlm
uv pip install -e ".[test,docs]"
```

With uv:
```bash
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[test,docs]"
```


## Testing

When test dependencies are installed, the test suite can be run via:

```bash
pytest tests
```

## Documentation

Documentation is generated using [mkdocs](https://www.mkdocs.org/) and hosted on GitHub Pages. To build the documentation, run:

```bash
mkdocs build
```

To serve the documentation locally, run:

```bash
mkdocs serve
```

## Commit Hooks

We use [pre-commit](https://pre-commit.com/) to manage a series of git
pre-commit hooks for the project; for example, each time you commit code, the
hooks will make sure that your python is formatted properly. If your code isn't,
the hook will format it, so when you try to commit the second time you'll get
past the hook.

All hooks are defined in `.pre-commit-config.yaml`. To install these hooks,
install `pre-commit` if you don't yet have it. I prefer using
[pipx](https://github.com/pipxproject/pipx) so that `pre-commit` stays globally
available.

```bash
pipx install pre-commit
```

Then install the hooks with this command:

```bash
pre-commit install
```

Now they'll run on every commit. If you want to run them manually, run the
following command:

```bash
pre-commit run --all-files
```
