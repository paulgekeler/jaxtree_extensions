# Jaxtree Extensions

`jaxtree-extensions` extends JAX's native `jax.tree_util` with extra tree utilities for JAX, Equinox, and other pytree frameworks.

## Core Utilities

| Function | Description | Comparison |
| :--- | :--- | :--- |
| `tree_map_prune` | Map and filter leaves by predicate, collapsing containers. | Similar to `jax.tree_util.tree_map` but drops/collapses leaves. |
| `tree_select_by_path` / `tree_at_path` | Mask or edit nodes matching wildcard path strings (e.g. `**/bias`). | Similar to `equinox.tree_at` but using glob/wildcard patterns. |
| `tree_zip_mismatched` | Zip mismatched trees via intersection/padding and broadcast scalars. | Extends `jax.tree_util.tree_map` to handle structure/shape mismatches. |
| `tree_map_enumerate` | Map over a tree passing a sequential flat index to each leaf. | Similar to `jax.tree_util.tree_map` but provides automatic indices `0, 1, 2...` |
| `tree_compare` | Compare two trees and generate a detailed mismatch diagnostic report. | Similar to `chex.assert_trees_all_close` but returns a diagnostic `TreeDiff`. |
| `tree_reduce_with_path` | Reduce a tree passing key-path tuples of each leaf. | Extends `jax.tree_util.tree_reduce` with path-aware context. |

## Installation

Install the package via PyPI:

```bash
pip install jaxtree-extensions
```

Or install locally for development:

```bash
uv pip install -e .
```

## Quick Start

```python
import jaxtree_extensions as jte
import jax.numpy as jnp

# Pruning elements
pruned, _ = jte.tree_map_prune(lambda x: x, {"a": [1, 2], "b": 3}, keep_fn=lambda x: x > 1)
# pruned = {"a": [2], "b": 3}

# Path-based editing
model = {"layer1": {"bias": jnp.ones(2)}, "layer2": {"bias": jnp.ones(2)}}
zero_bias = jte.tree_at_path(model, "**/bias", replace=0.0)
```

## Running Tests

To run the test suite:

```bash
uv run pytest
```
