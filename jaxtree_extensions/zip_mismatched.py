"""Utility to map over mismatched PyTree structures."""

from collections.abc import Callable
from typing import Any

import jax.tree_util as jtu


class _MissingSentinel:
    """Internal sentinel representing a missing node in a mismatched PyTree."""

    def __repr__(self) -> str:
        return "<Missing>"


_missing = _MissingSentinel()


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


def tree_zip_mismatched(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    mode: str = "intersection",
    fill_value: Any = None,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Zip multiple PyTrees by mapping `f` over their leaves, handling mismatches.

    If structures mismatch, the behavior is determined by the `mode` parameter.
    If a leaf is encountered in one tree and a container in another, the leaf
    is broadcasted to match the container's structure.

    Args:
        f: The function to map over aligned leaves.
        tree: The primary PyTree.
        *rest: Additional PyTrees to align and map.
        mode: How to handle mismatched keys or elements:
            - `"intersection"`: Drop unmatched keys or elements from lists/tuples.
            - `"padding"`: Pad unmatched elements/branches using `fill_value`.
        fill_value: The value used to fill missing elements in `"padding"` mode.
        is_leaf: A predicate called on each node to determine if it should be
            treated as a leaf.

    Returns:
        The zipped PyTree.
    """
    registry = jtu.default_registry
    trees = (tree,) + rest

    def check_is_leaf(x: Any) -> bool:
        if x is _missing:
            return True
        if is_leaf is not None and is_leaf(x):
            return True
        return registry.flatten_one_level(x) is None

    def recurse(nodes: tuple[Any, ...]) -> tuple[Any, bool]:
        # If all nodes are missing, return missing
        if all(n is _missing for n in nodes):
            return _missing, False

        # In intersection mode, if any node is missing, drop this branch
        if mode == "intersection" and any(n is _missing for n in nodes):
            return _missing, False

        # If all nodes are leaves
        if all(check_is_leaf(n) for n in nodes):
            replaced_nodes = [fill_value if n is _missing else n for n in nodes]
            return f(*replaced_nodes), True

        # Find the first valid container to determine structure
        container_node = None
        for n in nodes:
            if n is not _missing and not check_is_leaf(n):
                container_node = n
                break

        assert container_node is not None
        container_type = type(container_node)

        is_dict = isinstance(container_node, dict)
        is_list = isinstance(container_node, list)
        is_tuple = isinstance(container_node, tuple) and not hasattr(
            container_node, "_fields"
        )

        def get_children(
            node: Any, template_flat: Any, template_type: type
        ) -> tuple[list[Any], Any]:
            if node is _missing:
                return [_missing] * len(template_flat[0]), None

            flat = registry.flatten_one_level(node)
            if flat is None:
                # Leaf broadcasting
                return [node] * len(template_flat[0]), None

            if type(node) is not template_type:
                # Type mismatch, treat container as missing
                return [_missing] * len(template_flat[0]), None

            children, aux = flat
            return list(children), aux

        flat_template = registry.flatten_one_level(container_node)
        assert flat_template is not None

        if is_dict:
            # Align dict by keys
            all_keys: set[Any] = set()
            for n in nodes:
                if isinstance(n, dict):
                    all_keys.update(n.keys())

            if mode == "intersection":
                keys_list = None
                for n in nodes:
                    if isinstance(n, dict):
                        if keys_list is None:
                            keys_list = set(n.keys())
                        else:
                            keys_list &= set(n.keys())
                keys = keys_list if keys_list is not None else set()
            else:
                keys = all_keys

            res_dict = {}
            for key in sorted(keys):
                child_nodes = []
                for n in nodes:
                    if n is _missing:
                        child_nodes.append(_missing)
                    elif isinstance(n, dict):
                        child_nodes.append(n.get(key, _missing))
                    elif check_is_leaf(n):
                        child_nodes.append(n)
                    else:
                        child_nodes.append(_missing)

                val, keep = recurse(tuple(child_nodes))
                if keep:
                    res_dict[key] = val
            return res_dict, len(res_dict) > 0 or mode == "padding"

        elif is_list or is_tuple:
            lengths = [
                len(n)
                for n in nodes
                if (isinstance(n, list) or isinstance(n, tuple))
                and not hasattr(n, "_fields")
            ]
            if mode == "intersection":
                length = min(lengths) if lengths else 0
            else:
                length = max(lengths) if lengths else 0

            res_children = []
            for i in range(length):
                child_nodes = []
                for n in nodes:
                    if n is _missing:
                        child_nodes.append(_missing)
                    elif (isinstance(n, list) or isinstance(n, tuple)) and not hasattr(
                        n, "_fields"
                    ):
                        child_nodes.append(n[i] if i < len(n) else _missing)
                    elif check_is_leaf(n):
                        child_nodes.append(n)
                    else:
                        child_nodes.append(_missing)

                val, keep = recurse(tuple(child_nodes))
                if keep:
                    res_children.append(val)
                elif mode == "padding":
                    res_children.append(val if val is not _missing else fill_value)

            res_container = list(res_children) if is_list else tuple(res_children)
            return res_container, len(res_container) > 0 or mode == "padding"

        else:
            # Custom container or namedtuple
            tpl_children, _ = flat_template

            child_results = []
            for idx in range(len(tpl_children)):
                child_nodes = []
                for n in nodes:
                    n_children, _ = get_children(n, flat_template, container_type)
                    child_nodes.append(n_children[idx])

                val, keep = recurse(tuple(child_nodes))
                if keep:
                    child_results.append(val)
                else:
                    child_results.append(fill_value)

            reconstructed = _reconstruct_one_level(container_node, child_results)
            return reconstructed, True

    result, _ = recurse(trees)
    return fill_value if result is _missing else result
