"""Reduction utility with leaf key-paths for JAX PyTrees."""

from collections.abc import Callable
from typing import Any

import jax.tree_util as jtu


def tree_reduce_with_path(
    f: Callable[[Any, tuple[Any, ...], Any], Any],
    tree: Any,
    initializer: Any = ...,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Reduce a PyTree over its leaves, passing their key paths to the reduction function.

    Args:
        f: The reduction function, called as `f(accumulator, path, leaf)`.
        tree: The PyTree to reduce.
        initializer: The initial accumulator value. If not specified, the first
            leaf value is used as the initial accumulator.
        is_leaf: A predicate called on each node to determine if it should be
            treated as a leaf.

    Returns:
        The reduced accumulator value.

    Raises:
        ValueError: If the PyTree is empty and no initializer is provided.
    """
    leaves_with_path, _ = jtu.tree_flatten_with_path(tree, is_leaf=is_leaf)

    if not leaves_with_path:
        if initializer is not ...:
            return initializer
        raise ValueError(
            "tree_reduce_with_path() on empty PyTree requires an initializer"
        )

    if initializer is ...:
        val = leaves_with_path[0][1]
        start_idx = 1
    else:
        val = initializer
        start_idx = 0

    for path, leaf in leaves_with_path[start_idx:]:
        val = f(val, path, leaf)

    return val
