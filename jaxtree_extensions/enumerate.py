"""Utility to map functions over PyTrees with flat leaf indices."""

from collections.abc import Callable
from typing import Any

import jax.tree_util as jtu


def tree_map_enumerate(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Map a function over leaves, passing the flat leaf index as the first argument.

    Args:
        f: The function to map, called as `f(idx, leaf, *rest_leaves)`.
        tree: The primary PyTree.
        *rest: Additional matching PyTrees.
        is_leaf: A predicate called on each node to determine if it should be
            treated as a leaf.

    Returns:
        The mapped PyTree.
    """
    leaves, treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
    leaves_rest = [jtu.tree_flatten(r, is_leaf=is_leaf)[0] for r in rest]

    mapped_leaves = [
        f(idx, l, *lr)
        for idx, (l, *lr) in enumerate(zip(leaves, *leaves_rest, strict=True))
    ]

    return treedef.unflatten(mapped_leaves)  # type: ignore[attr-defined]
