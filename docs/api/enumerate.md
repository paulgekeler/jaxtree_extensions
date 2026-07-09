# Map Enumerate API Reference

Map a function over a PyTree, passing a sequential flat index to each leaf.

## Comparison
* **JAX**: Identical to `jax.tree_util.tree_map`, but provides a running flat leaf index starting from `0` to the mapping function `f(index, leaf)`, saving the boilerplate of manually tracking state in a mutable list or closure.

## Example
```python
import jaxtree_extensions as jte

tree = {"a": [10, 20], "b": 30}

# Tag each leaf with its flat index
res = jte.tree_map_enumerate(lambda idx, x: (idx, x), tree)
# res = {"a": [(0, 10), (1, 20)], "b": (2, 30)}
```

::: jaxtree_extensions.tree_map_enumerate
