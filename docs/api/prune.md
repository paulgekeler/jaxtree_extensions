# Pruning API Reference

Traverse a PyTree structure and prune out leaves that do not satisfy a predicate.

## Comparison
* **JAX**: Similar to `jax.tree_util.tree_map`, but dynamically removes elements from the PyTree structure instead of keeping a 1:1 mapping.
* **Equinox**: Allows selective filtering of components, but unlike Equinox's filtering which replaces leaves with `None` or leaves the tree structure unchanged, `tree_map_prune` completely removes (collapses) elements from lists, tuples, and dictionaries.

## Example
```python
import jaxtree_extensions as jte

tree = {"a": [1, 2, 3], "b": 4}
# Map x * 10, keeping only values that are not divisible by 20
pruned, treedef = jte.tree_map_prune(
    lambda x: x * 10,
    tree,
    keep_fn=lambda x: x % 20 != 0
)

# pruned = {'a': [10, 30], 'b': None}

# Reconstruct the original structure
original = treedef.unflatten(pruned, fill_value=0)
# original = {'a': [10, 0, 30], 'b': 0}
```

::: jaxtree_extensions.tree_map_prune

::: jaxtree_extensions.PrunedTreeDef
