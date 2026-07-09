# Jaxtree Extensions

`jaxtree-extensions` is a Python library that extends JAX's native `jax.tree_util` with additional tree utilities. It is designed to provide clean, reusable helper functions for mapping, filtering, and manipulating PyTree structures in common JAX workflows.

## Installation

You can install the package locally using `uv`:

```bash
uv pip install -e .
```

## Quick Start

```python
import jax.numpy as jnp
import jaxtree_extensions as jte

# Example tree structure
tree = {
    "a": jnp.ones((3, 3)),
    "b": jnp.zeros((3,)),
}
```

## Running Tests

To run the test suite:

```bash
uv run pytest
```
