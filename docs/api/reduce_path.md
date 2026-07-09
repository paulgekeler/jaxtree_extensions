# Path Reduction API Reference

Reduce a PyTree to a single value, passing the key path of each leaf to the reduction function.

## Comparison
* **JAX**: Extends `jax.tree_util.tree_reduce` by using path-aware traversal. Similar to how `jax.tree_util.tree_map_with_path` extends `jax.tree_util.tree_map`, `tree_reduce_with_path` allows tracking leaf context paths (e.g. key-path tuples) during the aggregation phase.

## Example
```python
import jaxtree_extensions as jte

tree = {"a": [10, 20], "b": 30}

# Collect string representations of all leaf paths
paths = jte.tree_reduce_with_path(
    lambda acc, path, leaf: acc + [jte.path_to_str(path)],
    tree,
    initializer=[]
)
# paths = ["/a/0", "/a/1", "/b"]
```

::: jaxtree_extensions.tree_reduce_with_path
