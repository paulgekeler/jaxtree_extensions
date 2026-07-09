# Comparison API Reference

Compare two PyTrees structure, types, and array closeness, generating a detailed difference report.

## Comparison
* **Chex**: Similar to `chex.assert_trees_all_close` or other validation libraries, but instead of raising an immediate assertion error on the first mismatch, `tree_compare` performs a complete traversal and returns a structured `TreeDiff` diagnostic object describing all mismatches in detail.

## Example
```python
import jaxtree_extensions as jte
import jax.numpy as jnp

tree_a = {"a": [1, 2], "b": jnp.ones((2, 2))}
tree_b = {"a": [1.0, 2], "b": jnp.ones((2, 2)) + 0.1}

diff = jte.tree_compare(tree_a, tree_b, rtol=1e-3, atol=1e-3)
if diff.has_changes:
    print(diff.report())
# Output shows the type mismatch at 'a/0' and the value difference at 'b'
```

::: jaxtree_extensions.tree_compare

::: jaxtree_extensions.TreeDiff
