"""Pruning utilities for JAX PyTrees."""

from collections.abc import Callable
from typing import Any

import jax.tree_util as jtu


class KeptLeaf:
    """Sentinel representing a leaf that was kept during pruning."""

    def __repr__(self) -> str:
        return "Kept"


class PrunedLeaf:
    """Sentinel representing a leaf that was pruned."""

    def __repr__(self) -> str:
        return "Pruned"


def _reconstruct_one_level(node: Any, new_children: list[Any]) -> Any:
    """Reconstruct a custom container of the same type as node with new children.

    Args:
        node: The original container node.
        new_children: The new children list to unflatten into the container.

    Returns:
        The reconstructed container node.
    """
    registry = jtu.default_registry
    flat_one = registry.flatten_one_level(node)
    if flat_one is None:
        return node
    children, _ = flat_one
    child_ids = {id(c) for c in children}
    _, level_treedef = jtu.tree_flatten(node, is_leaf=lambda x: id(x) in child_ids)
    return level_treedef.unflatten(new_children)  # type: ignore[attr-defined]


class PrunedTreeDef:
    """Structure definition for reconstructing pruned JAX PyTrees.

    Allows restoring the original PyTree structure from a pruned tree, filling
    pruned leaves with a specific value or a matching template PyTree.

    Args:
        template: The template PyTree representing the original structure.
        is_leaf_fn: A predicate called on each node to determine if it should
            be treated as a leaf.
    """

    def __init__(
        self, template: Any, is_leaf_fn: Callable[[Any], bool] | None = None
    ) -> None:
        self.template = template
        self.is_leaf_fn = is_leaf_fn

    def unflatten(self, pruned_tree: Any, fill_value: Any = None) -> Any:
        """Reconstruct the original PyTree structure from a pruned tree.

        Args:
            pruned_tree: The pruned PyTree returned by `tree_map_prune`.
            fill_value: The value to fill pruned leaf positions with. Can be a
                single value (like `None`), or a PyTree matching the original
                structure.

        Returns:
            The reconstructed PyTree with the original structure.
        """
        registry = jtu.default_registry

        def recurse(tpl: Any, prn: Any, fill: Any) -> Any:
            if isinstance(tpl, KeptLeaf):
                return prn
            if isinstance(tpl, PrunedLeaf):
                return fill

            flat_tpl = registry.flatten_one_level(tpl)
            if flat_tpl is None:
                # Fallback if somehow a leaf is not a sentinel
                return prn

            tpl_children, tpl_aux = flat_tpl

            is_dict = isinstance(tpl, dict)
            is_list = isinstance(tpl, list)
            is_tuple = isinstance(tpl, tuple) and not hasattr(tpl, "_fields")

            flat_fill = None
            if fill is not None:
                flat_fill = registry.flatten_one_level(fill)

            if is_dict:
                keys = tpl_aux
                pruned_dict = prn if prn is not None else {}
                res_dict = {}
                for idx, key in enumerate(keys):
                    tpl_child = tpl_children[idx]
                    prn_child = pruned_dict.get(key, None)
                    fill_child = None
                    if flat_fill is not None and isinstance(fill, dict) and key in fill:
                        fill_child = fill[key]
                    else:
                        fill_child = fill

                    res_dict[key] = recurse(tpl_child, prn_child, fill_child)
                return res_dict

            elif is_list or is_tuple:
                tpl_child_keeps = []
                for tc in tpl_children:

                    def has_kept(x: Any) -> bool:
                        if isinstance(x, KeptLeaf):
                            return True
                        if isinstance(x, PrunedLeaf):
                            return False
                        fo = registry.flatten_one_level(x)
                        if fo is None:
                            return False
                        return any(has_kept(c) for c in fo[0])

                    tpl_child_keeps.append(has_kept(tc))

                prn_children_iter = iter(prn) if prn is not None else iter([])
                res_children = []
                for idx, keep in enumerate(tpl_child_keeps):
                    tpl_child = tpl_children[idx]
                    prn_child = next(prn_children_iter) if keep else None
                    fill_child = None
                    if (
                        flat_fill is not None
                        and (isinstance(fill, list) or isinstance(fill, tuple))
                        and idx < len(flat_fill[0])
                    ):
                        fill_child = flat_fill[0][idx]
                    else:
                        fill_child = fill

                    res_children.append(recurse(tpl_child, prn_child, fill_child))
                return list(res_children) if is_list else tuple(res_children)

            else:
                # Non-collapsable container (namedtuple, custom class)
                prn_children = None
                if prn is not None:
                    flat_prn = registry.flatten_one_level(prn)
                    if flat_prn is not None:
                        prn_children = flat_prn[0]

                if prn_children is None:
                    prn_children = [None] * len(tpl_children)

                if flat_fill is not None and type(fill) is type(tpl):
                    fill_children = flat_fill[0]
                else:
                    fill_children = [fill] * len(tpl_children)

                res_children = []
                for idx in range(len(tpl_children)):
                    res_children.append(
                        recurse(
                            tpl_children[idx], prn_children[idx], fill_children[idx]
                        )
                    )
                return _reconstruct_one_level(tpl, res_children)

        return recurse(self.template, pruned_tree, fill_value)


def tree_map_prune(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    keep_fn: Callable[[Any], bool],
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[Any, PrunedTreeDef]:
    """Map a function over a PyTree, pruning leaves that don't match a predicate.

    For collapsable PyTree containers (dicts, lists, tuples), pruned leaves are
    completely filtered out. For non-collapsable or custom PyTree containers,
    pruned leaves are replaced with `None`.

    Args:
        f: The function to map over PyTree leaves.
        tree: The primary PyTree to traverse.
        *rest: Additional matching PyTrees to map over in parallel.
        keep_fn: A predicate called on each mapped leaf to determine if it
            should be kept. Returns `True` to keep, `False` to prune.
        is_leaf: A predicate called on each node to determine if it should be
            treated as a leaf instead of traversed.

    Returns:
        A tuple containing:
            pruned_tree: The mapped and pruned PyTree.
            treedef: A `PrunedTreeDef` structure definition that can reconstruct
                the original tree structure (e.g. replacing pruned nodes with
                custom values).
    """
    leaves, _ = jtu.tree_flatten(tree, is_leaf=is_leaf)
    leaves_rest = [jtu.tree_flatten(r, is_leaf=is_leaf)[0] for r in rest]
    mapped_leaves = [f(l, *lr) for l, *lr in zip(leaves, *leaves_rest, strict=True)]
    keep_mask = [keep_fn(ml) for ml in mapped_leaves]

    mapped_iter = iter(mapped_leaves)
    keep_iter = iter(keep_mask)

    registry = jtu.default_registry

    def process_node(node: Any) -> tuple[Any, bool, Any]:
        if is_leaf is not None and is_leaf(node):
            val = next(mapped_iter)
            keep = next(keep_iter)
            tpl = KeptLeaf() if keep else PrunedLeaf()
            return val, keep, tpl

        flat_one = registry.flatten_one_level(node)
        if flat_one is None:
            val = next(mapped_iter)
            keep = next(keep_iter)
            tpl = KeptLeaf() if keep else PrunedLeaf()
            return val, keep, tpl

        children, aux = flat_one
        child_res = [process_node(c) for c in children]

        is_dict = isinstance(node, dict)
        is_list = isinstance(node, list)
        is_tuple = isinstance(node, tuple) and not hasattr(node, "_fields")

        if is_dict:
            keys = aux
            pruned_dict = {}
            tpl_dict = {}
            for key, (c_val, c_keep, c_tpl) in zip(keys, child_res, strict=True):
                tpl_dict[key] = c_tpl
                if c_keep:
                    pruned_dict[key] = c_val
            has_kept = any(r[1] for r in child_res)
            return pruned_dict, has_kept, tpl_dict

        elif is_list:
            pruned_list = []
            tpl_list = []
            for c_val, c_keep, c_tpl in child_res:
                tpl_list.append(c_tpl)
                if c_keep:
                    pruned_list.append(c_val)
            has_kept = any(r[1] for r in child_res)
            return pruned_list, has_kept, tpl_list

        elif is_tuple:
            pruned_list = []
            tpl_list = []
            for c_val, c_keep, c_tpl in child_res:
                tpl_list.append(c_tpl)
                if c_keep:
                    pruned_list.append(c_val)
            has_kept = any(r[1] for r in child_res)
            return tuple(pruned_list), has_kept, tuple(tpl_list)

        else:
            # Non-collapsable container
            pruned_children = []
            tpl_children = []
            for c_val, c_keep, c_tpl in child_res:
                tpl_children.append(c_tpl)
                pruned_children.append(c_val if c_keep else None)

            reconstructed_pruned = _reconstruct_one_level(node, pruned_children)
            reconstructed_tpl = _reconstruct_one_level(node, tpl_children)
            has_kept = any(r[1] for r in child_res)
            return reconstructed_pruned, has_kept, reconstructed_tpl

    pruned_tree, _, template = process_node(tree)
    return pruned_tree, PrunedTreeDef(template, is_leaf)
