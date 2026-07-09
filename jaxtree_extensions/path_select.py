"""Path selection and modification utilities for JAX PyTrees."""

import re
from collections.abc import Callable
from typing import Any

import jax.tree_util as jtu


def _key_to_str(key: Any) -> str:
    """Convert a JAX PyTree path key to a simple string segment.

    Args:
        key: The JAX path key object (e.g. DictKey, SequenceKey).

    Returns:
        The converted string segment.
    """
    if hasattr(key, "key"):
        return str(key.key)
    if hasattr(key, "name"):
        return str(key.name)
    if hasattr(key, "idx"):
        return str(key.idx)
    return str(key)


def path_to_str(path: tuple[Any, ...]) -> str:
    """Convert a JAX PyTree path tuple to a standard forward-slash separated path string.

    Args:
        path: The JAX path key tuple.

    Returns:
        The slash-separated path string (e.g. "/layer1/bias").
    """
    parts = [_key_to_str(k) for k in path]
    return "/" + "/".join(parts)


def glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a glob pattern to a compiled regex pattern matching slash-separated paths.

    Supports:
        *   matching any characters except a slash (one path segment)
        **  matching any characters recursively (multiple path segments)
        ?   matching a single character except a slash

    If the pattern starts with a slash, it matches the absolute path from the root.
    Otherwise, it matches any suffix (subpath) ending with the pattern.

    Args:
        pattern: The glob-like path pattern.

    Returns:
        A compiled regular expression matching valid string paths.
    """
    regex_parts = []
    i = 0
    n = len(pattern)
    while i < n:
        char = pattern[i]
        if char == "*":
            if i + 1 < n and pattern[i + 1] == "*":
                if i + 2 < n and pattern[i + 2] == "/":
                    regex_parts.append("(?:.*/)?")
                    i += 3
                else:
                    regex_parts.append(".*")
                    i += 2
            else:
                regex_parts.append("[^/]*")
                i += 1
        elif char == "?":
            regex_parts.append("[^/]")
            i += 1
        elif char == "/":
            regex_parts.append("/")
            i += 1
        else:
            regex_parts.append(re.escape(char))
            i += 1

    regex_str = "".join(regex_parts)
    if pattern.startswith("/"):
        return re.compile(f"^{regex_str}$")
    else:
        return re.compile(f"^(?:.*/)?{regex_str}$")


def tree_select_by_path(
    tree: Any, pattern: str, is_leaf: Callable[[Any], bool] | None = None
) -> Any:
    """Return a boolean mask PyTree matching a glob-like path pattern.

    Leaves matching the pattern are True, others are False.

    Args:
        tree: The PyTree to inspect.
        pattern: A glob-like path pattern (e.g. "**/weight").
        is_leaf: A predicate called on each node to determine if it should be
            treated as a leaf.

    Returns:
        A boolean mask PyTree matching the original structure.
    """
    rx = glob_to_regex(pattern)

    def select_fn(path: tuple[Any, ...], leaf: Any) -> bool:
        path_str = path_to_str(path)
        return bool(rx.match(path_str))

    return jtu.tree_map_with_path(select_fn, tree, is_leaf=is_leaf)


def tree_at_path(
    tree: Any,
    pattern: str,
    replace: Any = None,
    replace_fn: Callable[[Any], Any] | None = None,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Modify a PyTree by replacing leaves matching a path pattern.

    Args:
        tree: The PyTree to modify.
        pattern: A glob-like path pattern (e.g. "**/bias").
        replace: The static replacement value for matched leaves. Ignored if
            `replace_fn` is provided.
        replace_fn: A function mapped over matched leaves to compute their new
            values.
        is_leaf: A predicate called on each node to determine if it should be
            treated as a leaf.

    Returns:
        A modified PyTree with replaced values at matched paths.
    """
    rx = glob_to_regex(pattern)

    def update_fn(path: tuple[Any, ...], leaf: Any) -> Any:
        path_str = path_to_str(path)
        if rx.match(path_str):
            if replace_fn is not None:
                return replace_fn(leaf)
            return replace
        return leaf

    return jtu.tree_map_with_path(update_fn, tree, is_leaf=is_leaf)
