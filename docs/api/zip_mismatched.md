# Mismatched Zip API Reference

Zip and map a function over multiple PyTrees that have mismatched structure or shape.

## Comparison
* **JAX**: Extends `jax.tree_util.tree_map` (when mapping over multiple trees), but handles mismatched structures (different dictionary keys, different list/tuple lengths) via intersection or padding rather than raising shape/structure errors. Supports broadcasting of scalar leaves to the whole tree structure.

## Example
```python
import jaxtree_extensions as jte

tree1 = {"a": [1, 2], "b": 3}
tree2 = {"a": [10, 20, 30], "c": 4}

# Zip matching elements via intersection
res = jte.tree_zip_mismatched(
    lambda x, y: x + y,
    tree1,
    tree2,
    mode="intersection"
)
# res = {"a": [11, 22]}

# Zip with padding and fallback fill value
res_pad = jte.tree_zip_mismatched(
    lambda x, y: x + y,
    tree1,
    tree2,
    mode="padding",
    fill_value=0
)
# res_pad = {"a": [11, 22, 30], "b": 3, "c": 4}
```

::: jaxtree_extensions.tree_zip_mismatched
