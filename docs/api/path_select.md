# Path Selection API Reference

Select and modify nodes in a PyTree using glob/wildcard path pattern matching.

## Comparison
* **Equinox**: Similar to `equinox.tree_at`, which replaces values/calls functions at specified nodes, but supports wildcard selectors (`*`, `**`, `?`) instead of static lambda accessors, allowing bulk modification of attributes across layers.
* **JAX**: Complements JAX's key paths, offering wildcard matching over path string structures.

## Example
```python
import jaxtree_extensions as jte
import jax.numpy as jnp

model = {
    "layer1": {"weight": jnp.ones((2, 2)), "bias": jnp.ones(2)},
    "layer2": {"weight": jnp.ones((2, 2)), "bias": jnp.ones(2)},
}

# Select all biases in all layers
mask = jte.tree_select_by_path(model, "**/bias")
# mask = {
#     "layer1": {"weight": False, "bias": True},
#     "layer2": {"weight": False, "bias": True},
# }

# Zero out all biases
zero_bias_model = jte.tree_at_path(model, "**/bias", replace=0.0)
```

::: jaxtree_extensions.tree_select_by_path

::: jaxtree_extensions.tree_at_path
